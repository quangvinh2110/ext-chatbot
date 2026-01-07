"""
FastAPI service for SQL Search API
"""
import json
import os
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AnyMessage
from sqlalchemy import create_engine
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from src.sql_assistant_v4.full_pipeline import build_sql_assistant_without_answer_generation
from src.tools.sqlite_database import SQLiteDatabase
from src.utils.client import (
    get_openai_llm_model,
    get_infinity_embeddings,
)
from src.router.llm_router import route_conversation

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "env", "internal.env"))

# Global variables for pipeline, database, and chat model
sql_assistant_pipeline = None
database = None
chat_model = None
langfuse = None


class SQLSearchRequest(BaseModel):
    """Request model for SQL search query"""
    current_message: str = Field(..., description="Current user message to process")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional conversation history in format [{'role': 'user'|'assistant', 'content': '...'}]"
    )


class RouteRequest(BaseModel):
    """Request model for routing conversation to a data source"""
    current_message: str = Field(..., description="Current user message to process")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional conversation history in format [{'role': 'user'|'assistant', 'content': '...'}]"
    )


class SQLSearchResponse(BaseModel):
    """Response model for SQL search query"""
    sql_query: str = Field(..., description="Generated SQL query")
    db_output: List[Dict[str, Any]] = Field(..., description="Database query results")
    ddl_schema: str = Field(..., description="Linked schema information in string format")


class RouteResponse(BaseModel):
    """Response model for routing decision"""
    data_source: str = Field(..., description="Chosen data source e.g. sql or vector or none")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


def initialize_pipeline(
    db_path: str,
    faiss_dir: str,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_kwargs: Dict[str, Any],
    embed_model: str,
    embed_base_url: str,
) -> None:
    """
    Initialize the SQL assistant pipeline with database and models.
    
    Args:
        db_path: Path to SQLite database file
        faiss_dir: Directory containing FAISS indexes
        llm_model: LLM model name
        llm_base_url: LLM API base URL
        llm_api_key: LLM API key
        embed_model: Embedding model name
        embed_base_url: Embedding API base URL
    """
    global sql_assistant_pipeline, database, chat_model
    
    # Initialize embeddings
    embeddings = get_infinity_embeddings(
        model=embed_model,
        infinity_api_url=embed_base_url
    )
    
    # Initialize database
    engine = create_engine(f"sqlite:///{db_path}")
    database = SQLiteDatabase(
        engine,
        faiss_dir=faiss_dir,
        embeddings=embeddings,
        concurrency_limit=10
    )
    if database is None:
        raise ValueError("Failed to initialize database")
    
    # Initialize LLM
    chat_model = get_openai_llm_model(
        model=llm_model,
        base_url=llm_base_url,
        api_key=SecretStr(llm_api_key),
        **llm_kwargs
    )
    if chat_model is None:
        raise ValueError("Failed to initialize chat model")
    
    # Build pipeline
    sql_assistant_pipeline = build_sql_assistant_without_answer_generation(
        chat_model=chat_model,
        database=database
    )


def convert_to_conversation(
    current_message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> List[AnyMessage]:
    """
    Convert current message and optional history to conversation format.
    
    Args:
        current_message: Current user message
        conversation_history: Optional conversation history
        
    Returns:
        List of AnyMessage objects
    """
    from langchain_core.messages import AIMessage
    
    messages: List[AnyMessage] = []
    
    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
    
    # Add current message
    messages.append(HumanMessage(content=current_message))
    
    return messages


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app initialization and cleanup"""
    # Startup: Initialize pipeline
    db_path = os.getenv("DATABASE_PATH", "/Users/vinhnguyen/Projects/ext-chatbot/resources/database/batdongsan.db")
    if not db_path or not os.path.exists(db_path):
        raise ValueError("DATABASE_PATH is not set or does not exist")
    faiss_dir = os.getenv("FAISS_DIR", "/Users/vinhnguyen/Projects/ext-chatbot/resources/faiss/batdongsan/")
    if not faiss_dir or not os.path.exists(faiss_dir):
        raise ValueError("FAISS_DIR is not set or does not exist")
    llm_model = os.getenv("LLM_MODEL")
    if not llm_model:
        raise ValueError("LLM_MODEL is not set")
    llm_base_url = os.getenv("LLM_BASE_URL")
    if not llm_base_url:
        raise ValueError("LLM_BASE_URL is not set")
    llm_api_key = os.getenv("LLM_API_KEY")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY is not set")
    llm_kwargs = json.loads(os.getenv("LLM_KWARGS", "{}"))
    embed_model = os.getenv("EMBED_MODEL")
    if not embed_model:
        raise ValueError("EMBED_MODEL is not set")
    embed_base_url = os.getenv("EMBED_BASE_URL")
    if not embed_base_url:
        raise ValueError("EMBED_BASE_URL is not set")
    
    if not all([llm_model, llm_base_url, llm_api_key, embed_model, embed_base_url]):
        raise ValueError(
            "Missing required environment variables: "
            "LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, EMBED_MODEL, EMBED_BASE_URL"
        )
    
    # Ensure embed_base_url ends with /v1 if not already
    if embed_base_url and not embed_base_url.endswith("/v1"):
        embed_base_url = f"{embed_base_url.rstrip('/')}/v1"
    
    print("Initializing SQL Search Pipeline...")
    print(f"Database: {db_path}")
    print(f"FAISS Directory: {faiss_dir}")
    print(f"LLM Model: {llm_model}")
    print(f"Embedding Model: {embed_model}")
    
    # Initialize Langfuse client
    global langfuse
    langfuse = get_client()
    
    # Verify Langfuse connection
    if langfuse.auth_check():
        print("Langfuse client is authenticated and ready!")
    else:
        print("Warning: Langfuse authentication failed. Please check your credentials and host.")
    
    initialize_pipeline(
        db_path=db_path,
        faiss_dir=faiss_dir,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_kwargs=llm_kwargs,
        embed_model=embed_model,
        embed_base_url=embed_base_url,
    )
    
    print("SQL Search Pipeline initialized successfully!")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down SQL Search Pipeline...")


# Create FastAPI app
app = FastAPI(
    title="SQL Search API",
    description="API for SQL Search Pipeline without answer generation",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if sql_assistant_pipeline is None or database is None or chat_model is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return HealthResponse(
        status="healthy",
        message="SQL Search Pipeline is running"
    )


@app.post("/sql_search", response_model=SQLSearchResponse)
async def sql_search(request: SQLSearchRequest):
    """
    Process a user message through the SQL search pipeline.
    
    Args:
        request: SQL search request with current message and optional conversation history
        
    Returns:
        SQLSearchResponse with SQL queries, database results, and metadata
    """
    if sql_assistant_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Please check server logs."
        )
    
    try:
        # Convert message to conversation format
        conversation = convert_to_conversation(
            current_message=request.current_message,
            conversation_history=request.conversation_history
        )
        
        # Initialize state
        initial_state = {
            "conversation": conversation,
            "sql_queries": [],
            "db_output": [],
        }
        
        # Initialize Langfuse CallbackHandler for tracing
        langfuse_handler = CallbackHandler()
        
        # Run pipeline with Langfuse tracing
        final_state = await sql_assistant_pipeline.ainvoke(
            initial_state,
            config={"callbacks": [langfuse_handler]}
        )
        if database:
            linked_schema = final_state.get("linked_schema", {})
            if linked_schema:
                ddl_schema = "\n\n".join([
                    database.get_table_info_no_throw(
                        table_name,
                        get_col_comments=True,
                        allowed_col_names=list(col_types.keys()),
                        sample_count=5,
                    )
                    for table_name, col_types in linked_schema.items()
                ])
            else:
                ddl_schema = "\n\n".join([
                    database.get_table_info_no_throw(
                        table_name,
                        get_col_comments=True,
                        sample_count=3,
                    )
                    for table_name in database.get_usable_table_names()
                ])
        else:
            ddl_schema = ""
        sql_queries = final_state.get("sql_queries", [])
        
        # Extract results
        return SQLSearchResponse(
            sql_query=sql_queries[-1] if sql_queries else "",
            db_output=final_state.get("db_output", {}).get("result", [])[:5],
            ddl_schema=ddl_schema,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/route", response_model=RouteResponse)
async def route_api(request: RouteRequest):
    """
    Route a conversation to the appropriate data source.
    """
    if chat_model is None or database is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Please check server logs."
        )
    try:
        conversation = convert_to_conversation(
            current_message=request.current_message,
            conversation_history=request.conversation_history
        )
        
        # Initialize Langfuse CallbackHandler for tracing
        langfuse_handler = CallbackHandler()
        
        # Call route_conversation with Langfuse tracing
        data_source = await route_conversation(
            conversation=conversation,
            chat_model=chat_model,
            database=database,
            config={"callbacks": [langfuse_handler]}
        )
        return RouteResponse(data_source=data_source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SQL Search API",
        "version": "1.0.0",
        "description": "API for SQL Search Pipeline without answer generation",
        "endpoints": {
            "health": "/health",
            "sql_search": "/sql_search",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True
    )


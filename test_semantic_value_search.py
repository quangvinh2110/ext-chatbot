import json
from typing import List, Dict

from sqlalchemy import create_engine

from src.utils.client import get_infinity_embeddings
from src.tools.table.sqlite_database import SQLiteDatabase
from dotenv import load_dotenv
import os

load_dotenv()


def main():
    # Hard-coded paths as requested
    db_path = "/Users/vinhnguyen/Projects/ext-chatbot/resources/database/batdongsan.db"
    faiss_dir = "/Users/vinhnguyen/Projects/ext-chatbot/resources/faiss/batdongsan"

    # Initialize InfinityEmbeddings from .env via helper
    embeddings = get_infinity_embeddings(model=os.getenv("EMBED_MODEL"), infinity_api_url=os.getenv("EMBED_BASE_URL"))

    # Create SQLAlchemy engine and wrapped SQLiteDatabase with FAISS indexes loaded
    engine = create_engine(f"sqlite:///{db_path}")
    db = SQLiteDatabase(engine, faiss_dir=faiss_dir, embeddings=embeddings)

    # Triplets to test
    items: List[Dict[str, str]] = [
        {
            "table": "BĐS Cho thuê 500",
            "column": "Địa chỉ_đường",
            "original_value": "Láng",
            "operator": "IN",
        },
        {
            "table": "BĐS Cho thuê 500",
            "column": "Địa chỉ_đường",
            "original_value": "Cầu Giấy",
            "operator": "IN",
        },
    ]
    results = []
    for item in items:
        sims = db.search_similar_value(
            table_name=item["table"],
            column_name=item["column"],
            cell_value=item["original_value"],
            k=5,
        )
        results.append(
            {
                "input": item,
                "matches": sims,
            }
        )

    # Pretty-print as JSON so it is easy to inspect / pipe elsewhere
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



"""
Recall@k metric for search engine evaluation.

Recall@k measures the proportion of relevant documents that were retrieved 
in the top k results.
"""


def recall_at_k(
    predictions: list[str], 
    targets: list[str], 
    k: int, 
    required_n: int | None = None
) -> float:
    """
    Calculate Recall@k score.
    
    Standard Recall@k = (number of relevant documents in top k) / (total number of relevant documents)
    
    If required_n is specified, uses threshold-based recall:
    - If n or more relevant results found: recall = 1.0
    - Otherwise: recall = relevant_count / required_n
    
    Args:
        predictions: List of predicted document IDs (ordered by relevance)
        targets: List of relevant document IDs (ground truth)
        k: Number of top results to consider
        required_n: Optional threshold. If set, recall is 1.0 when n relevant results are found,
                    otherwise scales linearly (relevant_count / required_n). Defaults to None.
        
    Returns:
        Recall@k score (float between 0.0 and 1.0)
        
    Examples:
        >>> recall_at_k(['doc1', 'doc2', 'doc3'], ['doc1', 'doc3', 'doc4'], k=2)
        0.3333333333333333
        >>> recall_at_k(['doc1', 'doc2', 'doc3'], ['doc1', 'doc3', 'doc4'], k=3)
        0.6666666666666666
        >>> recall_at_k(['doc1', 'doc2', 'doc3'], ['doc1', 'doc3', 'doc4'], k=3, required_n=2)
        1.0
        >>> recall_at_k(['doc1', 'doc2', 'doc4'], ['doc1', 'doc3', 'doc4'], k=3, required_n=2)
        1.0
        >>> recall_at_k(['doc1', 'doc2', 'doc5'], ['doc1', 'doc3', 'doc4'], k=3, required_n=2)
        0.5
    """
    if not targets:
        # If there are no relevant documents, recall is undefined
        # Convention: return 1.0 if no targets (perfect recall for empty set)
        return 1.0
    
    if not predictions:
        return 0.0
    
    # Convert targets to set for O(1) lookup
    target_set = set(targets)
    
    # Take top k predictions
    top_k_predictions = predictions[:k]
    
    # Count relevant documents in top k
    relevant_count = sum(1 for doc_id in top_k_predictions if doc_id in target_set)
    
    # If required_n is specified, use threshold-based recall
    if required_n is not None:
        if required_n <= 0:
            raise ValueError("required_n must be a positive integer")
        # If required_n is greater than k, use k
        required_n = min(required_n, k)
        # If we found n or more relevant results, recall is 1.0
        if relevant_count >= required_n:
            return 1.0
        # Otherwise, scale linearly: relevant_count / required_n
        return relevant_count / required_n
    
    # Standard Recall@k = relevant_count / total_relevant
    return relevant_count / len(targets)


def recall_at_k_batch(
    predictions_list: list[list[str]], 
    targets_list: list[list[str]], 
    k: int,
    required_n: int | None = None
) -> float:
    """
    Calculate average Recall@k across multiple queries.
    
    Args:
        predictions_list: List of prediction lists, one per query
        targets_list: List of target lists, one per query
        k: Number of top results to consider
        required_n: Optional threshold for threshold-based recall. See recall_at_k for details.
        
    Returns:
        Average Recall@k score across all queries
    """
    if len(predictions_list) != len(targets_list):
        raise ValueError(
            f"Length mismatch: predictions_list has {len(predictions_list)} items, "
            f"targets_list has {len(targets_list)} items"
        )
    
    if not predictions_list:
        return 0.0
    
    recalls = [
        recall_at_k(pred, tgt, k, required_n=required_n) 
        for pred, tgt in zip(predictions_list, targets_list)
    ]
    
    return sum(recalls) / len(recalls)


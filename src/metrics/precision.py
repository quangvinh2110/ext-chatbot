"""
Precision@k metric for search engine evaluation.

Precision@k measures the proportion of relevant documents in the top k retrieved results.
"""


def precision_at_k(predictions: list[str], targets: list[str], k: int) -> float:
    """
    Calculate Precision@k score.
    
    Precision@k = (number of relevant documents in top k) / k
    
    Args:
        predictions: List of predicted document IDs (ordered by relevance)
        targets: List of relevant document IDs (ground truth)
        k: Number of top results to consider
        
    Returns:
        Precision@k score (float between 0.0 and 1.0)
        
    Examples:
        >>> precision_at_k(['doc1', 'doc2', 'doc3'], ['doc1', 'doc3'], k=2)
        0.5
        >>> precision_at_k(['doc1', 'doc2', 'doc3'], ['doc1', 'doc3'], k=3)
        0.6666666666666666
    """
    if k == 0:
        return 0.0
    
    if not predictions:
        return 0.0
    
    # Convert targets to set for O(1) lookup
    target_set = set(targets)
    
    # Take top k predictions
    top_k_predictions = predictions[:k]
    
    # Count relevant documents in top k
    relevant_count = sum(1 for doc_id in top_k_predictions if doc_id in target_set)
    
    # Precision@k = relevant_count / k
    return relevant_count / k


def precision_at_k_batch(
    predictions_list: list[list[str]], 
    targets_list: list[list[str]], 
    k: int
) -> float:
    """
    Calculate average Precision@k across multiple queries.
    
    Args:
        predictions_list: List of prediction lists, one per query
        targets_list: List of target lists, one per query
        k: Number of top results to consider
        
    Returns:
        Average Precision@k score across all queries
    """
    if len(predictions_list) != len(targets_list):
        raise ValueError(
            f"Length mismatch: predictions_list has {len(predictions_list)} items, "
            f"targets_list has {len(targets_list)} items"
        )
    
    if not predictions_list:
        return 0.0
    
    precisions = [
        precision_at_k(pred, tgt, k) 
        for pred, tgt in zip(predictions_list, targets_list)
    ]
    
    return sum(precisions) / len(precisions)


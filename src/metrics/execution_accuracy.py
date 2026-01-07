"""
Execution accuracy metric for text-to-SQL evaluation.

Execution accuracy measures the proportion of correctly retrieved tables
by comparing predicted results with target results through column matching.
"""

from typing import List, Tuple, Set


def _normalize_value(value) -> str:
    """
    Normalize a value for comparison.
    
    Handles None and converts to string for comparison.
    More conservative normalization - only handles common SQL edge cases.
    """
    if value is None:
        return ""
    # Convert to string, but preserve the original representation
    # This handles cases where SQL returns numbers as strings or vice versa
    return str(value)


def _columns_match(pred_col: List, target_col: List) -> bool:
    """
    Check if two columns have all values equal.
    
    Args:
        pred_col: List of values from prediction column
        target_col: List of values from target column
        
    Returns:
        True if all values are equal (same length and same values in same order)
    """
    if len(pred_col) != len(target_col):
        return False
    
    # Try exact equality first (fast path)
    if all(p == t for p, t in zip(pred_col, target_col)):
        return True
    
    # Try normalized comparison (handles None and type differences like int vs str)
    # This handles cases where SQLite returns numbers as int/float but they're compared as strings
    try:
        return all(_normalize_value(p) == _normalize_value(t) for p, t in zip(pred_col, target_col))
    except Exception:
        # If normalization fails, fall back to exact equality
        return False


def _map_columns(prediction: List[Tuple], target: List[Tuple]) -> Set[Tuple[int, int]]:
    """
    Map columns between prediction and target tables.
    
    Two columns match if all values are equal.
    Uses a greedy matching algorithm that tries to maximize matches.
    
    Args:
        prediction: List of tuples representing prediction table (rows)
        target: List of tuples representing target table (rows)
        
    Returns:
        Set of (pred_col_idx, target_col_idx) tuples representing matched columns
    """
    if not prediction or not target:
        return set()
    
    num_pred_cols = len(prediction[0])
    num_target_cols = len(target[0])
    
    matched_columns: Set[Tuple[int, int]] = set()
    used_target_cols = set()
    
    # Build all possible matches first
    possible_matches = []
    for pred_col_idx in range(num_pred_cols):
        pred_col = [row[pred_col_idx] for row in prediction]
        for target_col_idx in range(num_target_cols):
            target_col = [row[target_col_idx] for row in target]
            if _columns_match(pred_col, target_col):
                possible_matches.append((pred_col_idx, target_col_idx))
    
    # Greedily match columns (one-to-one mapping)
    # Sort by prediction column index to ensure deterministic matching
    possible_matches.sort()
    
    for pred_col_idx, target_col_idx in possible_matches:
        if pred_col_idx not in {m[0] for m in matched_columns} and target_col_idx not in used_target_cols:
            matched_columns.add((pred_col_idx, target_col_idx))
            used_target_cols.add(target_col_idx)
    
    return matched_columns


def _execution_accuracy_one(
    prediction: List[Tuple],
    target: List[Tuple],
    threshold: float = 0.8,
    debug: bool = False
) -> bool:
    """
    Calculate execution accuracy for a single sample.
    
    A sample is considered correct if:
    1. Number of rows match (len(prediction) == len(target))
    2. The ratio of matched columns to minimum number of columns exceeds threshold
    
    Args:
        prediction: List of tuples representing prediction table (rows)
        target: List of tuples representing target table (rows)
        threshold: Minimum ratio of matched columns (default: 0.5)
        
    Returns:
        True if the sample is considered correct, False otherwise
        
    Examples:
        >>> pred = [('A', 1), ('B', 2)]
        >>> tgt = [('A', 1), ('B', 2)]
        >>> _execution_accuracy_one(pred, tgt, threshold=0.5)
        True
        >>> pred = [('A', 1), ('B', 2)]
        >>> tgt = [('A', 3), ('B', 4)]  # Different values in second column
        >>> _execution_accuracy_one(pred, tgt, threshold=0.5)
        False
    """
    # Check if number of rows match
    if len(prediction) != len(target):
        return False
    
    # Handle empty tables
    if not prediction or not target:
        # Both empty -> correct, otherwise incorrect
        return len(prediction) == len(target)
    
    # Check if number of columns match (for safety)
    if len(prediction[0]) != len(target[0]):
        # Still try to map columns even if counts differ
        pass
    
    # Map columns between prediction and target
    matched_columns = _map_columns(prediction, target)
    num_matched = len(matched_columns)
    
    # Calculate ratio
    num_pred_cols = len(prediction[0])
    num_target_cols = len(target[0])
    min_cols = min(num_pred_cols, num_target_cols)
    
    if min_cols == 0:
        return True  # Both have no columns, consider correct
    
    ratio = num_matched / min_cols
    
    if debug:
        print(f"Rows: pred={len(prediction)}, target={len(target)}")
        print(f"Columns: pred={num_pred_cols}, target={num_target_cols}, min={min_cols}")
        print(f"Matched: {num_matched}/{min_cols} = {ratio:.3f}")
        print(f"Threshold: {threshold}, Result: {ratio > threshold}")
        if num_matched < min_cols:
            print(f"Unmatched prediction columns: {set(range(num_pred_cols)) - {m[0] for m in matched_columns}}")
            print(f"Unmatched target columns: {set(range(num_target_cols)) - {m[1] for m in matched_columns}}")
    
    return ratio > threshold


def execution_accuracy(
    predictions: List[List[Tuple]],
    targets: List[List[Tuple]],
    threshold: float = 0.8,
    debug: bool = False
) -> float:
    """
    Calculate average execution accuracy across multiple samples.
    
    Args:
        predictions: List of prediction tables, each is a list of tuples (rows)
        targets: List of target tables, each is a list of tuples (rows)
        threshold: Minimum ratio of matched columns (default: 0.5)
        
    Returns:
        Average execution accuracy score (float between 0.0 and 1.0)
        
    Raises:
        ValueError: If predictions and targets have different lengths
        
    Examples:
        >>> preds = [[('A', 1), ('B', 2)], [('X', 3)]]
        >>> tgts = [[('A', 1), ('B', 2)], [('X', 3)]]
        >>> execution_accuracy(preds, tgts)
        1.0
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: predictions has {len(predictions)} items, "
            f"targets has {len(targets)} items"
        )
    
    if not predictions:
        return 0.0
    
    correct_count = sum(
        _execution_accuracy_one(pred, tgt, threshold, debug=debug)
        for pred, tgt in zip(predictions, targets)
    )
    
    return correct_count / len(predictions)


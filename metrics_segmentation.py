import numpy as np

def validate_mask(mask: np.ndarray) -> bool:
    """Validate that the mask is binary (only contains 0s and 255s)."""
    if not np.array_equal(mask, mask.astype(np.uint8)):
        print("Mask contains invalid values.")
        return False
    return True

def calculate_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    if not validate_mask(predicted_mask) or not validate_mask(ground_truth_mask):
        return 0.0

    intersection = np.logical_and(predicted_mask == 255, ground_truth_mask == 255)
    union = np.logical_or(predicted_mask == 255, ground_truth_mask == 255)
    
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou

def calculate_precision(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    if not validate_mask(predicted_mask) or not validate_mask(ground_truth_mask):
        return 0.0

    TP = np.sum(np.logical_and(predicted_mask == 255, ground_truth_mask == 255))
    FP = np.sum(np.logical_and(predicted_mask == 255, ground_truth_mask == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

def calculate_recall(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    if not validate_mask(predicted_mask) or not validate_mask(ground_truth_mask):
        return 0.0

    TP = np.sum(np.logical_and(predicted_mask == 255, ground_truth_mask == 255))
    FN = np.sum(np.logical_and(predicted_mask == 0, ground_truth_mask == 255))
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall

def calculate_f1_score(precision: float, recall: float) -> float:
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

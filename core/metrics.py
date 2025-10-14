# core/metrics.py

def accuracy_score_torch(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    return correct / total

def f1_score_torch(y_true, y_pred, num_classes):
    f1_per_class = []
    for cls in range(num_classes):
        # True positives
        tp = ((y_pred == cls) & (y_true == cls)).sum().item()
        # False positives
        fp = ((y_pred == cls) & (y_true != cls)).sum().item()
        # False negatives
        fn = ((y_pred != cls) & (y_true == cls)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_per_class.append(f1)
    return sum(f1_per_class) / num_classes  # Macro average

def precision_score_torch(y_true, y_pred, num_classes):
    precisions = []
    for cls in range(num_classes):
        tp = ((y_pred == cls) & (y_true == cls)).sum().item()
        fp = ((y_pred == cls) & (y_true != cls)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        precisions.append(precision)
    return sum(precisions) / num_classes

def recall_score_torch(y_true, y_pred, num_classes):
    recalls = []
    for cls in range(num_classes):
        tp = ((y_pred == cls) & (y_true == cls)).sum().item()
        fn = ((y_pred != cls) & (y_true == cls)).sum().item()
        recall = tp / (tp + fn + 1e-8)
        recalls.append(recall)
    return sum(recalls) / num_classes

METRICS = {
    "Accuracy": accuracy_score_torch,
    "F1": f1_score_torch,
    "Precision": precision_score_torch,
    "Recall": recall_score_torch,
}
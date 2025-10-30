# Import necessary libraries
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy_score_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
    """
    Compute the accuracy score of a model.
    
    Args:
        y_true (torch.Tensor): Ground truth labels
        y_pred (torch.Tensor): Predicted labels
        
    Returns:
        float: Accuracy score
    """
    
    y_pred = y_pred_logits.argmax(dim=1)

    # Compute correct predictions and total number of samples
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    
    # Return accuracy score
    return correct / total

def topk_accuracy_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor, k: int = 5) -> float:
    """
    Compute the Top-k accuracy of a model.
    
    Args:
        y_true (torch.Tensor): Ground truth labels, shape (N,)
        y_pred_logits (torch.Tensor): Model output logits or probabilities, shape (N, C)
        k (int): Value of k for Top-k accuracy
        
    Returns:
        float: Top-k accuracy score
    """
    # Get top-k predicted class indices
    topk_pred = torch.topk(y_pred_logits, k=k, dim=1).indices  # (N, k)
    
    # Check if true label is among the top-k predictions
    correct = topk_pred.eq(y_true.view(-1, 1)).any(dim=1).sum().item()
    total = y_true.size(0)
    return correct / total

def f1_score_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int) -> float:
    """
    Compute the F1 score of a model.
    
    Args:
        y_true (torch.Tensor): Ground truth labels
        y_pred (torch.Tensor): Predicted labels
        num_classes (int): Number of classes
        
    Returns:
        float: F1 score
    """

    y_pred = y_pred_logits.argmax(dim=1)
    
    # Initialize list to store F1 scores for each class
    f1_per_class = []
    
    # Iterate over classes
    for cls in range(num_classes):
        
        # Compute true positives, false positives, and false negatives
        tp = ((y_pred == cls) & (y_true == cls)).sum().item()
        fp = ((y_pred == cls) & (y_true != cls)).sum().item()
        fn = ((y_pred != cls) & (y_true == cls)).sum().item()
        
        # Compute precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # Compute F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Append F1 score to list
        f1_per_class.append(f1)
    
    # Return macro average of F1 scores
    return sum(f1_per_class) / num_classes

def precision_score_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int) -> float:
    """
    Compute the precision score of a model.
    
    Args:
        y_true (torch.Tensor): Ground truth labels
        y_pred (torch.Tensor): Predicted labels
        num_classes (int): Number of classes
        
    Returns:
        float: Precision score
    """

    y_pred = y_pred_logits.argmax(dim=1)
    
    # Initialize list to store precision scores for each class
    precisions = []
    
    # Iterate over classes
    for cls in range(num_classes):
        
        # Compute true positives and false positives
        tp = ((y_pred == cls) & (y_true == cls)).sum().item()
        fp = ((y_pred == cls) & (y_true != cls)).sum().item()
        
        # Compute precision score
        precision = tp / (tp + fp + 1e-8)
        
        # Append precision score to list
        precisions.append(precision)
    
    # Return macro average of precision scores
    return sum(precisions) / num_classes

def recall_score_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int) -> float:
    """
    Compute the recall score of a model.
    
    Args:
        y_true (torch.Tensor): Ground truth labels
        y_pred (torch.Tensor): Predicted labels
        num_classes (int): Number of classes
        
    Returns:
        float: Recall score
    """

    y_pred = y_pred_logits.argmax(dim=1)
    
    # Initialize list to store recall scores for each class
    recalls = []
    
    # Iterate over classes
    for cls in range(num_classes):
        
        # Compute true positives and false negatives
        tp = ((y_pred == cls) & (y_true == cls)).sum().item()
        fn = ((y_pred != cls) & (y_true == cls)).sum().item()
        
        # Compute recall score
        recall = tp / (tp + fn + 1e-8)
        
        # Append recall score to list
        recalls.append(recall)
    
    # Return macro average of recall scores
    return sum(recalls) / num_classes

def confusion_matrix_torch(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute the confusion matrix of a model.
    
    Args:
        y_true (torch.Tensor): Ground truth labels
        y_pred (torch.Tensor): Predicted labels
        num_classes (int): Number of classes
        
    Returns:
        torch.Tensor: Confusion matrix
    """
    
    # Initialize tensor to store confusion matrix
    indices = num_classes * y_true + y_pred
    cm = torch.bincount(indices, minlength=num_classes*num_classes)
    cm = cm.reshape(num_classes, num_classes)
    
    return cm

def plot_confusion_matrix(cm: torch.Tensor, class_names: list) -> None:
    """
    Plot the confusion matrix of a model.
    
    Args:
        cm (torch.Tensor): Confusion matrix
        class_names (list): List of class names
        
    Returns:
        None
    """
    
    # Create figure and axis
    plt.figure(figsize=(8,6))
    
    # Use seaborn to plot heatmap
    sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    # Set labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Show plot
    plt.show()
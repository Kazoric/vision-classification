from typing import Callable, Dict, List, Optional, Tuple

import torch

# Registry : nom utilisable dans la config -> fonction (y_true, logits, **params) -> float
METRICS: Dict[str, Callable] = {}


def register_metric(fn: Callable) -> Callable:
    if fn.__name__ in METRICS:
        raise ValueError(f"Metric '{fn.__name__}' already registered")
    METRICS[fn.__name__] = fn
    return fn


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _confusion_from_labels(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    indices = num_classes * y_true.long() + y_pred.long()
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def confusion_matrix_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Confusion matrix (rows = true class, columns = predicted class).
    Not a scalar metric: use it after training, not through the config.
    """
    return _confusion_from_labels(y_true, y_pred_logits.argmax(dim=1), num_classes)


def _tp_fp_fn(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class true positives, false positives and false negatives (one pass)."""
    cm = confusion_matrix_torch(y_true, y_pred_logits, num_classes).float()
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    return tp, fp, fn


# ----------------------------------------------------------------------
# Scalar metrics (usable from the config)
# ----------------------------------------------------------------------

@register_metric
def accuracy_score_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
    """Top-1 accuracy."""
    y_pred = y_pred_logits.argmax(dim=1)
    return (y_true == y_pred).sum().item() / y_true.size(0)


@register_metric
def topk_accuracy_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy."""
    if k > y_pred_logits.size(1):
        raise ValueError(f"k={k} > number of classes ({y_pred_logits.size(1)})")
    topk_pred = torch.topk(y_pred_logits, k=k, dim=1).indices                # (N, k)
    correct = topk_pred.eq(y_true.view(-1, 1)).any(dim=1).sum().item()
    return correct / y_true.size(0)


@register_metric
def precision_score_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged precision."""
    tp, fp, _ = _tp_fp_fn(y_true, y_pred_logits, num_classes)
    return (tp / (tp + fp).clamp(min=1)).mean().item()


@register_metric
def recall_score_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged recall (equals balanced accuracy)."""
    tp, _, fn = _tp_fp_fn(y_true, y_pred_logits, num_classes)
    return (tp / (tp + fn).clamp(min=1)).mean().item()


@register_metric
def f1_score_torch(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged F1 score."""
    tp, fp, fn = _tp_fp_fn(y_true, y_pred_logits, num_classes)
    return (2 * tp / (2 * tp + fp + fn).clamp(min=1)).mean().item()


# ----------------------------------------------------------------------
# Plotting (heavy imports kept local)
# ----------------------------------------------------------------------

def plot_confusion_matrix(
    cm: torch.Tensor,
    class_names: List[str],
    normalize: bool = False,
    save_path: Optional[str] = None,
):
    """Plot a confusion matrix. If save_path is given, save instead of showing."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = cm.cpu().float()
    if normalize:                                   # each row sums to 1 (per-class recall)
        data = data / data.sum(dim=1, keepdim=True).clamp(min=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.numpy(), annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig
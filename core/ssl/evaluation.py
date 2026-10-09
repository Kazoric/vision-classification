import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(backbone, loader, device: str, amp: bool = True):
    """Extracts L2-normalized backbone features in evaluation mode (without augmentation)."""
    was_training = backbone.training
    backbone.eval()
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    feats, labels = [], []
    for images, y in loader:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            f = backbone(images.to(device, non_blocking=True))
        feats.append(F.normalize(f.float(), dim=1))
        labels.append(y.to(device))
    backbone.train(was_training)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_accuracy(train_f, train_y, val_f, val_y, num_classes: int,
                 k: int = 20, temperature: float = 0.1, chunk: int = 1000) -> float:
    """Computes kNN accuracy weighted by exp(sim / T), processing chunks to reduce memory usage."""
    correct = 0
    for i in range(0, val_f.size(0), chunk):
        sim = val_f[i:i + chunk] @ train_f.T  # (chunk, N_train)
        s, idx = sim.topk(k, dim=1)
        votes = torch.zeros(sim.size(0), num_classes, device=sim.device)
        votes.scatter_add_(1, train_y[idx], (s / temperature).exp())
        correct += (votes.argmax(dim=1) == val_y[i:i + chunk]).sum().item()
    return correct / val_f.size(0)


def make_knn_monitor(train_loader, val_loader, device, num_classes,
                     k: int = 20, temperature: float = 0.1):
    def monitor(method) -> float:
        tf, ty = extract_features(method.backbone, train_loader, device)
        vf, vy = extract_features(method.backbone, val_loader, device)
        return knn_accuracy(tf, ty, vf, vy, num_classes, k, temperature)
    return monitor
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from core.optim import build_optimizer, build_scheduler


class SSLTrainer:
    """Self-supervised pretraining: two views, no labels, kNN for monitoring."""

    def __init__(self, method, optimizer, device: str,
                 scheduler: Optional[LRScheduler] = None,
                 monitor_fn: Optional[Callable] = None, monitor_every: int = 10,
                 on_best: Optional[Callable[[int, float], None]] = None,
                 amp: bool = True, config: Optional[Any] = None,
                 run_id: Optional[str] = None):
        self.method = method.to(device)
        self.optimizer, self.device, self.scheduler = optimizer, device, scheduler
        self.monitor_fn, self.monitor_every = monitor_fn, monitor_every
        self.on_best, self.amp, self.config = on_best, amp, config
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.run_id = run_id or f"ssl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.history: Dict[str, List[float]] = {"loss": [], "emb_std": [], "tau": [], "lr": []}
        self.knn_history: List[Tuple[int, float]] = []  # (epoch, accuracy)
        self.best_knn = float("-inf")
        self.start_epoch = 0

    @classmethod
    def from_config(cls, method, config, device=None, monitor_fn=None, on_best=None, run_id=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        method = method.to(device)
        optimizer = build_optimizer(method.parameters(), config)  # Ignore the target (requires_grad=False).
        scheduler = build_scheduler(optimizer, config)
        return cls(method, optimizer, device, scheduler, monitor_fn,
                   monitor_every=config.ssl.monitor.knn_every, on_best=on_best,
                   amp=config.ssl.mixed_precision, config=config, run_id=run_id)

    def _run_monitor(self, epoch: int) -> None:
        acc = self.monitor_fn(self.method)
        self.knn_history.append((epoch, acc))
        tag = " (random initialization)" if epoch == 0 else ""
        print(f"[kNN] epoch {epoch}{tag}: {acc:.4f}")
        if acc > self.best_knn:
            self.best_knn = acc
            if self.on_best is not None and epoch > 0:
                self.on_best(epoch, acc)

    def train(self, loader, epochs: int) -> None:
        steps = len(loader)
        total, it = epochs * steps, self.start_epoch * steps

        if self.monitor_fn and self.start_epoch == 0:
            self._run_monitor(0)  # Baseline: kNN accuracy of an untrained backbone.

        for epoch in range(self.start_epoch, epochs):
            lr = self.optimizer.param_groups[0]["lr"]
            self.method.train()
            loss_sum = torch.zeros((), device=self.device)
            std_sum = torch.zeros((), device=self.device)

            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for i, ((v1, v2), _) in enumerate(pbar):  # Labels are ignored.
                v1 = v1.to(self.device, non_blocking=True)
                v2 = v2.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16,
                                    enabled=self.amp):
                    out = self.method(v1, v2)
                out["loss"].backward()
                self.optimizer.step()
                self.method.after_step(it / total)
                it += 1

                loss_sum += out["loss"].detach()
                std_sum += F.normalize(out["z"].float(), dim=1).std(dim=0).mean()
                if i % 50 == 0:
                    pbar.set_postfix({"loss": f"{out['loss'].item():.4f}"})

            loss = loss_sum.item() / steps
            emb_std = std_sum.item() / steps
            tau = getattr(self.method, "tau", float("nan"))
            for key, val in zip(("loss", "emb_std", "tau", "lr"), (loss, emb_std, tau, lr)):
                self.history[key].append(val)
            print(f"{'Train':<12} | Loss: {loss:.4f} | emb_std: {emb_std:.4f} "
                  f"| tau: {tau:.4f} | LR: {lr:.2e}")

            if self.scheduler is not None:
                self.scheduler.step()

            if self.monitor_fn and self.monitor_every > 0 and \
                    ((epoch + 1) % self.monitor_every == 0 or epoch + 1 == epochs):
                self._run_monitor(epoch + 1)
            print()

    def resume_from(self, epoch: int, best_knn: float) -> None:
        self.start_epoch = epoch
        self.best_knn = best_knn
        if self.scheduler is not None:
            for _ in range(epoch):  # Restore the LR to its value at the start of epoch `epoch`.
                self.scheduler.step()

    def save_hyperparams(self, extra_results: Optional[Dict] = None) -> None:
        if self.config is None:
            return
        meta = asdict(self.config) if is_dataclass(self.config) else dict(self.config)
        meta["experiment"]["run_id"] = self.run_id
        meta["results"] = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "best_knn": self.best_knn, **(extra_results or {})}
        path = os.path.join("experiments", self.run_id, "meta.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Configuration saved: {path}")
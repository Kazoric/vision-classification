import os
import json
from datetime import datetime
from dataclasses import is_dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import functools
import inspect
from core.metrics import METRICS

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import core.metrics as core_metrics
from core.optim import build_optimizer, build_scheduler


class Trainer:
    """Boucle d'entraînement / validation pour la classification d'images."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optimizer: torch.optim.Optimizer,
        device: str,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[LRScheduler] = None,
        metrics_config: Optional[Any] = None,
        on_best_model: Optional[Callable[[int, float], None]] = None,
        config: Optional[Any] = None,
        run_id: Optional[str] = None,
        amp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion if criterion is not None else model.criterion
        self.scheduler = scheduler
        self.on_best_model = on_best_model
        self.config = config
        self.amp = amp
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"

        # --- Run ID : celui du modèle en priorité ---
        if run_id is None:
            run_id = getattr(model, "run_id", None)
        if run_id is None and config is not None:
            run_id = getattr(config.experiment, "run_id", None)
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # --- Métriques : {nom: (fonction, params)} ---
        self.num_classes = num_classes if num_classes is not None else getattr(model, "num_classes", None)
        self.metrics = self._resolve_metrics(metrics_config, self.num_classes)
        if metrics_config is not None:
            self.monitor_metric = metrics_config.monitor_metric
            self.monitor_mode = metrics_config.monitor_mode.lower()
        else:
            self.monitor_metric, self.monitor_mode = "val_loss", "min"
        assert self.monitor_mode in ("max", "min"), "monitor_mode must be 'max' or 'min'"
        if self.monitor_metric not in ("loss", "val_loss") and self.monitor_metric not in self.metrics:
            raise ValueError(f"monitor_metric '{self.monitor_metric}' absent de metrics.configs")

        # --- Historiques ---
        self.train_loss: List[float] = []
        self.valid_loss: List[float] = []
        self.lr_history: List[float] = []
        self.train_metrics: Dict[str, List[float]] = {n: [] for n in self.metrics}
        self.valid_metrics: Dict[str, List[float]] = {n: [] for n in self.metrics}

        # --- État interne ---
        self.start_epoch = 0
        self.best_metric_value = float("-inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch_metrics: dict = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: Any,
        device: Optional[str] = None,
        on_best_model: Optional[Callable[[int, float], None]] = None,
        metrics_config: Optional[Any] = None,
    ) -> "Trainer":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Le gel du backbone (linear probe) est déjà appliqué dans Model.__init__ :
        # build_optimizer ne reçoit donc que les paramètres entraînables.
        optimizer = build_optimizer(model.parameters(), config)
        scheduler = build_scheduler(optimizer, config)

        return cls(
            model=model,
            num_classes=config.model.num_classes,
            optimizer=optimizer,
            device=device,
            criterion=getattr(model, "criterion", None),
            scheduler=scheduler,
            metrics_config=metrics_config if metrics_config is not None else config.metrics,
            on_best_model=on_best_model,
            config=config,
            run_id=getattr(config.experiment, "run_id", None),
            amp=getattr(config.training, "amp", False),
        )

    @staticmethod
    def _resolve_metrics(metrics_config: Optional[Any], num_classes: Optional[int]) -> Dict[str, Callable]:
        """Config names -> ready-to-call functions f(y_true, logits)."""
        if metrics_config is None:
            return {}
        resolved = {}
        for name, spec in metrics_config.configs.items():
            func, params = spec
            params = dict(params or {})
            if isinstance(func, str):
                if func not in METRICS:
                    raise KeyError(f"Unknown metric '{func}'. Available: {sorted(METRICS)}")
                func = METRICS[func]
            if "num_classes" in inspect.signature(func).parameters:
                if num_classes is None:
                    raise ValueError(f"Metric '{name}' needs num_classes")
                params.setdefault("num_classes", num_classes)
            resolved[name] = functools.partial(func, **params)
        return resolved

    # ------------------------------------------------------------------
    # Entraînement et validation
    # ------------------------------------------------------------------

    def _autocast(self):
        return torch.autocast(device_type=self.device_type, dtype=torch.bfloat16, enabled=self.amp)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> None:
        for epoch in range(self.start_epoch, epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_history.append(current_lr)

            self.model.train()
            loss_sum, n_seen = 0.0, 0
            all_logits, all_labels = [], []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                bs = labels.size(0)
                loss_sum += loss.item() * bs
                n_seen += bs
                all_logits.append(logits.detach().float())
                all_labels.append(labels)
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

            epoch_loss = loss_sum / n_seen
            self.train_loss.append(epoch_loss)

            metric_outputs = self._compute_metrics(torch.cat(all_labels), torch.cat(all_logits))
            for name, value in metric_outputs.items():
                self.train_metrics[name].append(value)

            print(f"{'Train':<12} | Loss: {epoch_loss:.4f}{self._fmt(metric_outputs)} | LR: {current_lr:.2e}")

            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader)
                self._maybe_save_best(epoch, val_loss, val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()
            print()

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        loss_sum, n_seen = 0.0, 0
        all_logits, all_labels = [], []

        for images, labels in data_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with self._autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            bs = labels.size(0)
            loss_sum += loss.item() * bs
            n_seen += bs
            all_logits.append(logits.float())
            all_labels.append(labels)

        val_loss = loss_sum / n_seen
        self.valid_loss.append(val_loss)

        metric_outputs = self._compute_metrics(torch.cat(all_labels), torch.cat(all_logits))
        for name, value in metric_outputs.items():
            self.valid_metrics[name].append(value)

        print(f"{'Validation':<12} | Loss: {val_loss:.4f}{self._fmt(metric_outputs)}")
        return val_loss, metric_outputs

    # ------------------------------------------------------------------
    # Métriques et suivi du meilleur modèle
    # ------------------------------------------------------------------

    def _compute_metrics(self, y_true: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
        outputs = {}
        for name, func in self.metrics.items():
            score = func(y_true, logits)
            outputs[name] = score.item() if torch.is_tensor(score) else float(score)
        return outputs

    @torch.no_grad()
    def collect_predictions(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        labels_all, logits_all = [], []
        for images, labels in data_loader:
            with self._autocast():
                out = self.model(images.to(self.device, non_blocking=True))
            logits_all.append(out.float().cpu())
            labels_all.append(labels)
        return torch.cat(labels_all), torch.cat(logits_all)

    @staticmethod
    def _fmt(metric_outputs: Dict[str, float]) -> str:
        return "".join(f" | {n}: {v:.4f}" for n, v in metric_outputs.items())

    def _is_better(self, value: float) -> bool:
        if self.monitor_mode == "max":
            return value > self.best_metric_value
        return value < self.best_metric_value

    def _maybe_save_best(self, epoch: int, val_loss: float, val_metrics: dict) -> None:
        if self.monitor_metric in ("loss", "val_loss"):
            current_value = val_loss
        else:
            current_value = val_metrics[self.monitor_metric]

        if self._is_better(current_value):
            self.best_metric_value = current_value
            self.best_epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": self.train_loss[-1],
                "val_loss": val_loss,
                "train_metrics": {n: h[-1] for n, h in self.train_metrics.items() if h},
                "val_metrics": val_metrics,
                "monitor_metric": self.monitor_metric,
                "monitor_value": current_value,
            }
            if self.on_best_model is not None:
                self.on_best_model(epoch + 1, current_value)

    def get_final_metrics(self) -> dict:
        return self.best_epoch_metrics

    def resume_from(self, epoch: int, best_metric_value: float) -> None:
        self.start_epoch = epoch
        self.best_metric_value = best_metric_value

    # ------------------------------------------------------------------
    # Métadonnées
    # ------------------------------------------------------------------

    def save_hyperparams(self, extra_results: Optional[Dict] = None) -> None:
        if self.config is None:
            print("[WARN] Aucune config liée au Trainer : hyperparamètres non sauvegardés.")
            return

        meta = asdict(self.config) if is_dataclass(self.config) else dict(self.config)
        if isinstance(meta.get("experiment"), dict):
            meta["experiment"]["run_id"] = self.run_id

        results = extra_results if extra_results is not None else self.best_epoch_metrics
        if results:
            meta["results"] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "best_validation_results": results,
            }

        path = os.path.join("experiments", self.run_id, "meta.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Configuration saved: {path}")
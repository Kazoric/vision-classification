# core/trainer.py

import numpy as np
import torch
from tqdm import tqdm

from core.metrics import METRICS

class Trainer:
    def __init__(self, model, optimizer, criterion, device, save=False, checkpoint_fn=None, scheduler=None, metrics=None, num_classes=None):
        """
        model: le modèle PyTorch (nn.Module)
        optimizer: optimiseur (ex: Adam)
        criterion: fonction de perte (ex: CrossEntropyLoss)
        device: 'cuda' ou 'cpu'
        save: booléen, indique si on sauvegarde le meilleur modèle
        checkpoint_fn: fonction de sauvegarde externe (ex: model.save_checkpoint)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save = save
        self.save_checkpoint = checkpoint_fn
        self.scheduler = scheduler
        self.num_classes = num_classes

        self.train_loss = []
        self.valid_loss = []

        self.best_val_loss = float('inf')
        self.start_epoch = 0

        if metrics:
            self.metrics = {
                name: METRICS[name] for name in metrics if name in METRICS
            }
        else:
            self.metrics = {}
        self.train_metrics = {name: [] for name in self.metrics}
        self.valid_metrics = {name: [] for name in self.metrics}

    def train(self, train_loader, val_loader=None, epochs=10):
        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []

            for images, labels in tqdm(train_loader, desc=f"🎯 Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.append(predicted)
                all_labels.append(labels)

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            self.train_loss.append(running_loss)

            metric_outputs = {}
            for name, fn in self.metrics.items():
                if fn.__code__.co_argcount == 3:
                    score = fn(all_labels, all_preds, self.num_classes)
                else:
                    score = fn(all_labels, all_preds)
                self.train_metrics[name].append(score)
                metric_outputs[name] = score

            metrics_str = " | ".join(f"{name}: {value:.4f}" for name, value in metric_outputs.items())
            print(f"{'Train':<7} | Loss: {running_loss:.4f} | {metrics_str}")

            if val_loader:
                val_loss = self.evaluate(val_loader)

                if self.save and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.save_checkpoint:
                        self.save_checkpoint(epoch + 1, val_loss)
            
            if self.scheduler is not None:
                self.scheduler.step()

            print()

    def evaluate(self, data_loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.append(predicted)
                all_labels.append(labels)

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

        self.valid_loss.append(running_loss)

        metric_outputs = {}
        for name, fn in self.metrics.items():
            if fn.__code__.co_argcount == 3:
                score = fn(all_labels, all_preds, self.num_classes)
            else:
                score = fn(all_labels, all_preds)
            self.valid_metrics[name].append(score)
            metric_outputs[name] = score

        metrics_str = " | ".join(f"{name}: {value:.4f}" for name, value in metric_outputs.items())
        print(f"{'Validation':<7} | Loss: {running_loss:.4f} | {metrics_str}")

        return running_loss

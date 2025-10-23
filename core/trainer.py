# Import necessary libraries
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Optional

class Trainer:
    """
    Class for training a PyTorch model.
    
    Attributes:
        model (nn.Module): Trained model to use for prediction
        optimizer (torch.optim.Optimizer): Optimizer to use for training
        criterion (torch.nn.Module): Loss function to use for training
        device (str): Device to use for training (e.g., 'cuda' or 'cpu')
        save (bool): Whether to save the best model during training
        checkpoint_fn (function): External checkpoint saving function
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler to use
        num_classes (int): Number of classes in the dataset
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: str,
        save: bool = False,
        checkpoint_fn: Callable[[int, float], None] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        metrics: Optional[dict] = None,
        num_classes: Optional[int] = None
    ) -> None:
        """
        Initialize the Trainer object.
        
        Args:
            model (nn.Module): Trained model to use for prediction
            optimizer (torch.optim.Optimizer): Optimizer to use for training
            criterion (torch.nn.Module): Loss function to use for training
            device (str): Device to use for training
            save (bool): Whether to save the best model during training
            checkpoint_fn (function): External checkpoint saving function
            scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler to use
            metrics (list): List of evaluation metrics to track
            num_classes (int): Number of classes in the dataset
        
        Returns:
            None
        """
        
        # Set model and optimizer attributes
        self.model = model
        self.optimizer = optimizer
        
        # Set criterion attribute
        self.criterion = criterion
        
        # Set device attribute
        self.device = device
        
        # Set save attribute
        self.save = save
        
        # Set checkpoint saving function attribute
        self.save_checkpoint = checkpoint_fn
        
        # Set scheduler attribute
        self.scheduler = scheduler
        
        # Set number of classes attribute
        self.num_classes = num_classes

        # Initialize loss and metric tracking lists
        self.train_loss = []
        self.valid_loss = []

        # Initialize best validation loss and start epoch attributes
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        # Initialize metrics attribute
        if metrics:
            self.metrics = metrics
            # self.metrics = {
            #     name: METRICS[name] for name in metrics if name in METRICS
            # }
        else:
            self.metrics = {}
        
        # Initialize train and validation metric tracking dictionaries
        self.train_metrics = {name: [] for name in self.metrics}
        self.valid_metrics = {name: [] for name in self.metrics}

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 10) -> None:
        """
        Train the model on a given dataset.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader (optional)
            epochs (int): Number of training epochs
        
        Returns:
            None
        """
        
        # Iterate over training epochs
        for epoch in range(self.start_epoch, epochs):
            
            # Set model to training mode
            self.model.train()
            
            # Initialize running loss and prediction tracking lists
            running_loss = 0.0
            all_outputs = []
            all_labels = []

            # Iterate over training data loader
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                
                # Move input to specified device
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                # Zero gradients and make predictions
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Update running loss and prediction tracking lists
                running_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(labels)

            # Update train loss and metric tracking lists
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            self.train_loss.append(running_loss)

            # Compute and store metrics
            metric_outputs = self._compute_metrics(all_labels, all_outputs)
            for name, value in metric_outputs.items():
                self.train_metrics[name].append(value)

            # Print training metrics
            metrics_str = " | ".join(f"{name}: {value:.4f}" for name, value in metric_outputs.items())
            print(f"{'Train':<12} | Loss: {running_loss:.4f} | {metrics_str}")

            # Evaluate model on validation set if available
            if val_loader:
                val_loss = self.evaluate(val_loader)

                # Save best model if validation loss improves
                if self.save and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.save_checkpoint:
                        self.save_checkpoint(epoch + 1, val_loss)
            
            # Update learning rate scheduler if available
            if self.scheduler is not None:
                self.scheduler.step()

            print()

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the model on a given dataset.
        
        Args:
            data_loader (DataLoader): Data loader to use for evaluation
        
        Returns:
            float: Validation loss
        """
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize running loss and prediction tracking lists
        running_loss = 0.0
        all_outputs = []
        all_labels = []

        # Iterate over data loader
        with torch.no_grad():
            for images, labels in data_loader:
                
                # Move input to specified device
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                # Make predictions
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Update running loss and prediction tracking lists
                running_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(labels)

        # Update validation loss and metric tracking lists
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        self.valid_loss.append(running_loss)

        # Compute and store metrics
        metric_outputs = self._compute_metrics(all_labels, all_outputs)
        for name, value in metric_outputs.items():
            self.valid_metrics[name].append(value)

        # Print validation metrics
        metrics_str = " | ".join(f"{name}: {value:.4f}" for name, value in metric_outputs.items())
        print(f"{'Validation':<12} | Loss: {running_loss:.4f} | {metrics_str}")
        
        return running_loss

    def _compute_metrics(self, y_true, y_pred_logits):
        metric_outputs = {}
        for name, (func, params) in self.metrics.items():
            score = func(y_true, y_pred_logits, **params)
            metric_outputs[name] = score
        return metric_outputs



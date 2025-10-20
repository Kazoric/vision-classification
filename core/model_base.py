# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Tuple

# Import custom modules
from core.trainer import Trainer
from core.predictor import Predictor
from core.checkpoint import CheckpointManager

class Model(ABC):
    """
    Base class for all models. Provides common functionality and attributes.
    
    Attributes:
        device (str): Device to use for training and inference (e.g., 'cuda' or 'cpu')
        lr (float): Initial learning rate
        save (bool): Whether to save model checkpoints during training
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset used to train the model
        run_id (str): Unique identifier for the current experiment
        optimizer_cls (class): Class of the optimizer to use
        optimizer_params (dict): Parameters for the optimizer
        scheduler_cls (class): Class of the scheduler to use
        scheduler_params (dict): Parameters for the scheduler
        metrics (list): List of evaluation metrics to track
        num_classes (int): Number of classes in the dataset
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        lr: float = 0.001,
        save: bool = False,
        model_name: str = "model",
        dataset_name: str = "unknown",
        run_id: Optional[str] = None,
        optimizer_cls: Any = None,
        optimizer_params: Optional[Dict[str, Any]] = None,
        scheduler_cls: Any = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        num_classes: Optional[int] = None
    ) -> None:
        """
        Initialize the Model object.
        
        Args:
            device (str): Device to use for training and inference
            lr (float): Initial learning rate
            save (bool): Whether to save model checkpoints during training
            model_name (str): Name of the model
            dataset_name (str): Name of the dataset used to train the model
            run_id (str): Unique identifier for the current experiment
            optimizer_cls (class): Class of the optimizer to use
            optimizer_params (dict): Parameters for the optimizer
            scheduler_cls (class): Class of the scheduler to use
            scheduler_params (dict): Parameters for the scheduler
            metrics (list): List of evaluation metrics to track
            num_classes (int): Number of classes in the dataset
        """
        
        # Set device to CPU or GPU if available
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate run ID if not provided
        if run_id is None:
            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = f"{model_name}_{dataset_name}_{date}"
        
        self.run_id = run_id

        # Set number of classes
        self.num_classes = num_classes
        
        # Build and move model to device
        self.model = self.build_model().to(self.device)
        
        # Initialize criterion (cross-entropy loss with label smoothing)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0)
        
        # Set metrics
        self.metrics = metrics

        # Initialize optimizer (default is Adam)
        if optimizer_cls is None:
            optimizer_cls = Adam
        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer_name = optimizer_cls.__name__
        self.optimizer_params = optimizer_params

        # Initialize learning rate and optimizer
        self.lr = lr
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr, **optimizer_params)

        # Initialize scheduler (optional)
        if scheduler_cls is not None:
            self.scheduler_name = scheduler_cls.__name__
            if scheduler_params is None:
                scheduler_params = {}
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_params)
            self.scheduler_params = scheduler_params
        else:
            self.scheduler = None
            self.scheduler_name = None
            self.scheduler_params = {}

        # Initialize checkpoint manager
        self.checkpoint = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            run_id=run_id,
            model_name=model_name
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            save=save,
            checkpoint_fn=self.checkpoint.save,
            scheduler=self.scheduler,
            metrics=metrics,
            num_classes=self.num_classes
        )

        # Initialize predictor
        self.predictor = Predictor(self.model, self.device)

        # Initialize best validation loss and checkpoint path
        self.best_val_loss = float('inf')
        self.checkpoint_path = "./checkpoints/"
        self.start_epoch = 0

    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        Abstract method to build the model. Must be implemented by subclasses.
        
        Returns:
            nn.Module: The built model
        """
        pass

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10) -> None:
        """
        Train the model for a specified number of epochs.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader (optional)
            epochs (int): Number of epochs to train for
        """
        
        self.trainer.train(train_loader, val_loader, epochs)

    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model on a validation set.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Model performance metric (e.g., accuracy or loss)
        """
        
        return self.trainer.evaluate(val_loader)

    def predict(self, inputs: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Make predictions on a given input.
        
        Args:
            inputs (torch.Tensor): Input tensor
            return_probs (bool): Whether to return class probabilities instead of labels
            
        Returns:
            torch.Tensor: Predicted labels or class probabilities
        """
        
        return self.predictor.predict(inputs, return_probs=return_probs)
    
    def predict_on_loader(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on a data loader.
        
        Args:
            dataloader (DataLoader): Data loader to make predictions on
            
        Returns:
            tuple: Tuple containing true labels and predicted labels
        """
        
        all_preds = []
        all_labels = []

        for inputs, labels in dataloader:
            preds = self.predict(inputs)
            all_preds.append(preds)
            all_labels.append(labels)

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        return y_true, y_pred.cpu()

    def load_checkpoint(self) -> None:
        """
        Load the latest checkpoint.
        
        Returns:
            bool: Whether loading was successful
        """
        
        success = self.checkpoint.load_latest()
        if success:
            self.trainer.start_epoch = self.checkpoint.start_epoch
            self.trainer.best_val_loss = self.checkpoint.best_val_loss

    def save_hyperparams(
        self,
        optimizer_name: str,
        optimizer_params: Dict[str, Any],
        scheduler_name: str,
        scheduler_params: Dict[str, Any],
        batch_size: int,
        num_epochs: int
    ) -> None:
        """
        Save hyperparameters to a JSON file.
        
        Args:
            optimizer_name (str): Name of the optimizer
            optimizer_params (dict): Parameters for the optimizer
            scheduler_name (str): Name of the scheduler
            scheduler_params (dict): Parameters for the scheduler
            batch_size (int): Batch size used during training
            num_epochs (int): Number of epochs trained for
        """
        
        meta = {
            "model_name": self.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "batch_size": batch_size,
            "learning_rate": self.lr,
            "num_epochs": num_epochs,
            "optimizer": optimizer_name,
            "optimizer_params": optimizer_params,
            "scheduler": scheduler_name,
            "scheduler_params": scheduler_params,
            "metrics": self.metrics,
            "model_params": self.get_model_specific_params()
        }

        path = os.path.join(f"experiments/{self.run_id}", "meta.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=4)
        print(f"Hyperparameters saved to file: {path}")

    def get_model_specific_params(self) -> Dict[str, Any]:
        """
        Abstract method to retrieve model-specific parameters. Must be implemented by subclasses.
        
        Returns:
            dict: Model-specific parameters
        """
        
        raise NotImplementedError("Each model must define get_model_specific_params()")
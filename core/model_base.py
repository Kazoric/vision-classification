import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from core.trainer import Trainer
from core.predictor import Predictor
from core.checkpoint import CheckpointManager

class Model(ABC):
    def __init__(self, device=None, 
                 lr=0.001, 
                 save=False, 
                 model_name="model",
                 checkpoint_dir=None, 
                 optimizer_cls=None, 
                 optimizer_params=None,
                 scheduler_cls=None,
                 scheduler_params=None,
                 metrics=None,
                 num_classes=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_classes = num_classes
        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        if optimizer_cls is None:
            from torch.optim import Adam
            optimizer_cls = Adam
        if optimizer_params is None:
            optimizer_params = {}

        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr, **optimizer_params)

        if scheduler_cls is not None:
            if scheduler_params is None:
                scheduler_params = {}
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None

        self.checkpoint = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_dir=checkpoint_dir,
            model_name=model_name
        )

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

        self.predictor = Predictor(self.model, self.device)

        self.best_val_loss = float('inf')
        self.checkpoint_path = "./checkpoints/"
        self.start_epoch = 0

    @abstractmethod
    def build_model(self):
        """À implémenter dans les sous-classes, retourne un nn.Module."""
        pass

    def train(self, train_loader, val_loader=None, epochs=10):
        self.trainer.train(train_loader, val_loader, epochs)

    def evaluate(self, val_loader):
        return self.trainer.evaluate(val_loader)

    def predict(self, inputs, return_probs=False):
        return self.predictor.predict(inputs, return_probs=return_probs)

    def load_checkpoint(self):
        success = self.checkpoint.load_latest()
        if success:
            self.trainer.start_epoch = self.checkpoint.start_epoch
            self.trainer.best_val_loss = self.checkpoint.best_val_loss
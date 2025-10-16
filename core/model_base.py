import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

from core.trainer import Trainer
from core.predictor import Predictor
from core.checkpoint import CheckpointManager
from core.logger import TensorBoardLogger

class Model(ABC):
    def __init__(self, device=None, 
                 lr=0.001, 
                 save=False, 
                 model_name="model",
                 run_id=None, 
                 optimizer_cls=None, 
                 optimizer_params=None,
                 scheduler_cls=None,
                 scheduler_params=None,
                 metrics=None,
                 num_classes=None,
                 use_logger=True, 
                 log_dir="runs"):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if run_id is None:
            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = f"{model_name}_{date}"
        self.run_id = run_id

        self.num_classes = num_classes
        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = metrics

        if optimizer_cls is None:
            optimizer_cls = Adam
        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer_name = optimizer_cls.__name__
        self.optimizer_params = optimizer_params

        self.lr = lr
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr, **optimizer_params)

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

        self.checkpoint = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            run_id=run_id,
            model_name=model_name
        )

        if use_logger:
            self.logger = TensorBoardLogger(log_dir=log_dir, experiment_name=self.run_id)
        else:
            self.logger = None

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            save=save,
            checkpoint_fn=self.checkpoint.save,
            scheduler=self.scheduler,
            metrics=metrics,
            num_classes=self.num_classes,
            logger=self.logger
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

    def save_hyperparams(self, optimizer_name, optimizer_params, scheduler_name, scheduler_params, batch_size, num_epochs):
        """
        Sauvegarde des hyperparamètres dans meta.json (communs + spécifiques au modèle)
        """
        meta = {
            "model_name": self.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "batch_size": batch_size,
            "learning_rate": self.lr,
            "num_epochs": num_epochs,
            "optimizer": self.optimizer_name,
            "optimizer_params": self.optimizer_params,
            "scheduler": self.scheduler_name,
            "scheduler_params": self.scheduler_params,
            "metrics": self.metrics,
            "model_params": self.get_model_specific_params()  # appelé sur la classe enfant
        }

        path = os.path.join(f"experiments/{self.run_id}", "meta.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=4)
        print(f"✅ Hyperparamètres sauvegardés dans {path}")

    def get_model_specific_params(self):
        """
        Doit être redéfini dans la sous-classe pour retourner un dictionnaire.
        """
        raise NotImplementedError("Chaque modèle doit définir get_model_specific_params()")
    
    def close_logger(self):
        if self.logger:
            self.logger.close()

# core/checkpoint.py

import torch
import os
from datetime import datetime

class CheckpointManager:
    def __init__(self, model, optimizer, checkpoint_dir=None, model_name="model"):
        """
        model: nn.Module à sauvegarder
        optimizer: optimiseur à sauvegarder
        checkpoint_dir: dossier où stocker les checkpoints
        model_name: préfixe pour nommer les fichiers
        """
        self.model = model
        self.optimizer = optimizer
        self.model_name = model_name
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        if checkpoint_dir is None:
            run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            checkpoint_dir = f"./checkpoints/{model_name}_{run_id}"
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, epoch, val_loss):
        """
        Sauvegarde un checkpoint si la perte de validation est la meilleure.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }

        filename = f"{self.model_name}_epoch-{epoch}_loss-{val_loss:.4f}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")

        # Mettre à jour les métadonnées
        self.best_val_loss = val_loss
        self.start_epoch = epoch

    def load_latest(self):
        """
        Charge le dernier checkpoint (basé sur val_loss minimum).
        """
        if not os.path.exists(self.checkpoint_dir):
            print("ℹ️ No checkpoint directory found.")
            return False

        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt") and self.model_name in f
        ]
        if not checkpoint_files:
            print("ℹ️ No checkpoint files found.")
            return False

        # Trier les fichiers par val_loss (extrait du nom de fichier)
        checkpoint_files.sort(key=lambda x: float(x.split("loss-")[-1].replace(".pt", "")))

        best_checkpoint = checkpoint_files[0]
        path = os.path.join(self.checkpoint_dir, best_checkpoint)
        checkpoint = torch.load(path, map_location=self.model.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']

        print(f"📦 Loaded checkpoint: {best_checkpoint} (epoch {self.start_epoch}, val_loss {self.best_val_loss:.4f})")
        return True

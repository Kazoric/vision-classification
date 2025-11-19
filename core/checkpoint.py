# Import necessary libraries
import torch
import os

class CheckpointManager:
    """
    Class for managing model checkpoints.
    
    Attributes:
        model (nn.Module): Model to save and load checkpoints from
        optimizer (torch.optim.Optimizer): Optimizer to save and load checkpoints from
        run_id (str): Unique identifier for the current experiment
        model_name (str): Prefix for naming checkpoint files
        best_val_loss (float): Best validation loss seen so far
        start_epoch (int): Starting epoch number for training
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        run_id: str = None, 
        model_name: str = "model"
    ) -> None:
        """
        Initialize the CheckpointManager object.
        
        Args:
            model (nn.Module): Model to save and load checkpoints from
            optimizer (torch.optim.Optimizer): Optimizer to save and load checkpoints from
            run_id (str): Unique identifier for the current experiment
            model_name (str): Prefix for naming checkpoint files
        
        Returns:
            None
        """
        
        # Set model and optimizer attributes
        self.model = model
        self.optimizer = optimizer
        
        # Set model name attribute
        self.model_name = model_name
        
        # Initialize best validation loss and start epoch attributes
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = f"./experiments/{run_id}/checkpoints"
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, epoch: int, val_loss: float) -> None:
        """
        Save a checkpoint if the validation loss is the best seen so far.
        
        Args:
            epoch (int): Current epoch number
            val_loss (float): Validation loss for the current epoch
        
        Returns:
            None
        """
        
        # Create checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }

        # Create filename and path for saving checkpoint
        filename = f"{self.model_name}_epoch-{epoch}_loss-{val_loss:.4f}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        # Save checkpoint to file
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

        # Update metadata
        self.best_val_loss = val_loss
        self.start_epoch = epoch

    def load_latest(self, load_optimizer = True) -> bool:
        """
        Load the latest checkpoint (based on val_loss minimum).
        
        Returns:
            bool: Whether a checkpoint was loaded successfully
        """
        
        # Check if checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            print("No checkpoint directory found.")
            return False

        # Get list of checkpoint files in the directory
        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt") and self.model_name in f
        ]
        
        # Check if any checkpoint files exist
        if not checkpoint_files:
            print("No checkpoint files found.")
            return False
        
        # Sort checkpoint files by val_loss (extracted from filename)
        checkpoint_files.sort(key=lambda x: float(x.split("loss-")[-1].replace(".pt", "")))

        # Load the best checkpoint
        best_checkpoint = checkpoint_files[0]
        path = os.path.join(self.checkpoint_dir, best_checkpoint)
        checkpoint = torch.load(path, map_location=self.model.device)

        # Load model and optimizer state dictionaries
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Update metadata
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']

        print(f"Loaded checkpoint: {best_checkpoint} (epoch {self.start_epoch}, val_loss {self.best_val_loss:.4f})")
        return True

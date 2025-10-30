import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional
import torch

from core.trainer import Trainer

class Visualizer:
    """
    Class for visualizing training metrics.
    
    Attributes:
        None
    """
    
    def __init__(self) -> None:
        """
        Initialize the Visualizer object.
        
        Args:
            None
        """
        
        pass

    def plot_metrics(self, trainer: Trainer, run_id: str, save: Optional[bool] = True):
        """
        Plot training and validation loss, as well as each metric separately.
        
        Args:
            trainer (Trainer): Trainer object containing training metrics
            run_id (str): Unique identifier for the current experiment
            save (bool): Whether to save plots to file (default: True)
        
        Returns:
            None
        """
        
        # Get epochs range
        epochs = range(1, len(trainer.train_loss) + 1)

        # Plot loss
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, trainer.train_loss, label='Train Loss')
        plt.plot(epochs, trainer.valid_loss, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Plot each metric separately
        for metric_name in trainer.train_metrics:
            plt.figure(figsize=(8, 5))

            train_values = trainer.train_metrics[metric_name]
            valid_values = trainer.valid_metrics.get(metric_name, [])

            plt.plot(epochs, train_values, label=f'Train {metric_name}')
            if valid_values:
                plt.plot(epochs, valid_values, label=f'Val {metric_name}')

            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'Training and Validation {metric_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save plot to file if save=True
            if save:
                save_path = f"experiments/{run_id}/plots"
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f'{metric_name}')
                plt.savefig(save_path)
                print(f"Training curves saved to {save_path}")
        print()
        plt.show()

    def plot_confusion_matrix(self, cm: torch.Tensor, class_names: list, run_id: str, save: Optional[bool] = True):
        """
        Plot and optionally save the confusion matrix.
        
        Args:
            cm (torch.Tensor): Confusion matrix
            class_names (list): List of class names
            run_id (str): Unique identifier for the experiment
            save (bool, optional): Whether to save the plot (default: True)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        if save:
            save_dir = f"experiments/{run_id}/plots"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "confusion_matrix.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Confusion matrix saved to {save_path}")

        plt.show()
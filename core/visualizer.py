# core/visualizer.py

import matplotlib.pyplot as plt
import os

class Visualizer:
    def __init__(self):
        pass

    def plot_metrics(self, trainer, run_id, save=True):
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

            if save:
                save_path = f"experiments/{run_id}/plots"
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f'{metric_name}')
                plt.savefig(save_path)
                print(f"Training curves saved to {save_path}")
        print()
        plt.show()

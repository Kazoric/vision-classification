# core/visualizer.py

import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        pass

    # def plot_metrics(self, train_acc, val_acc, train_loss, val_loss, save_path=None):
    #     """
    #     Affiche (et optionnellement sauvegarde) les courbes d'accuracy et de loss.

    #     Args:
    #         train_acc (list of float): accuracy d'entraînement par epoch
    #         val_acc (list of float): accuracy de validation par epoch
    #         train_loss (list of float): perte d'entraînement par epoch
    #         val_loss (list of float): perte de validation par epoch
    #         save_path (str): chemin pour sauvegarder la figure (PNG), sinon affiche seulement
    #     """
    #     epochs = range(1, len(train_acc) + 1)

    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))

    #     # Accuracy
    #     ax1.plot(epochs, train_acc, label='Train Accuracy')
    #     ax1.plot(epochs, val_acc, label='Val Accuracy')
    #     ax1.set_title('Model Accuracy')
    #     ax1.set_xlabel('Epoch')
    #     ax1.set_ylabel('Accuracy (%)')
    #     ax1.legend()
    #     ax1.grid(True)

    #     # Loss
    #     ax2.plot(epochs, train_loss, label='Train Loss')
    #     ax2.plot(epochs, val_loss, label='Val Loss')
    #     ax2.set_title('Model Loss')
    #     ax2.set_xlabel('Epoch')
    #     ax2.set_ylabel('Loss')
    #     ax2.legend()
    #     ax2.grid(True)

    #     plt.tight_layout()

    #     if save_path:
    #         plt.savefig(save_path)
    #         print(f"📊 Training curves saved to {save_path}")
    #     else:
    #         plt.show()


    # def plot_metrics(self, trainer):

    #     epochs = range(1, len(trainer.train_loss) + 1)

    #     plt.figure(figsize=(12, 6))
    #     plt.plot(epochs, trainer.train_loss, label='Train Loss')
    #     plt.plot(epochs, trainer.valid_loss, label='Val Loss')

    #     for metric_name, values in trainer.train_metrics.items():
    #         plt.plot(epochs, values, label=f'Train {metric_name}')

    #     for metric_name, values in trainer.valid_metrics.items():
    #         plt.plot(epochs, values, label=f'Val {metric_name}')

    #     plt.xlabel('Epoch')
    #     plt.ylabel('Score')
    #     plt.legend()
    #     plt.title('Training and Validation Metrics')
    #     plt.show()


    def plot_metrics(self, trainer):
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
        
        plt.show()

import math
import matplotlib.pyplot as plt


def plot_ssl_curves(trainer, proj_dim: int, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(trainer.history["loss"])
    axes[0].set(title="Loss BYOL", xlabel="epoch")

    axes[1].plot(trainer.history["emb_std"], label="std des embeddings")
    axes[1].axhline(1 / math.sqrt(proj_dim), ls="--", c="gray", label="1/sqrt(d) (sain)")
    axes[1].set(title="Suivi du collapse", xlabel="epoch")
    axes[1].legend()

    if trainer.knn_history:
        epochs, acc = zip(*trainer.knn_history)
        axes[2].plot(epochs, acc, marker="o")
    axes[2].set(title="kNN (validation)", xlabel="epoch")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
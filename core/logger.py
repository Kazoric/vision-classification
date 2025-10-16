# core/logger.py
from torch.utils.tensorboard import SummaryWriter
import os

class TensorBoardLogger:
    def __init__(self, log_dir="runs", experiment_name="default"):
        self.log_path = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(log_dir=self.log_path)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_prefix, scalars: dict, step: int):
        for key, value in scalars.items():
            self.writer.add_scalar(f"{tag_prefix}/{key}", value, step)

    def close(self):
        self.writer.close()

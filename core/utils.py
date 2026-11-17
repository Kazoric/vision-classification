import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_run_id(model_name, dataset_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return f"{model_name}_{dataset_name}_{timestamp}"
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR


def build_optimizer(params, config):
    params = [p for p in params if p.requires_grad]   # indispensable pour le linear probe
    optimizer_cls = getattr(optim, config.optimizer.type)
    return optimizer_cls(params, lr=config.training.lr, **config.optimizer.params)


def build_scheduler(optimizer, config):
    warmup_epochs = config.training.warm_up_epochs
    has_main = config.scheduler.type is not None
    if warmup_epochs == 0 and not has_main:
        return None

    warmup = LinearLR(optimizer, start_factor=0.05, end_factor=1.0,
                      total_iters=warmup_epochs) if warmup_epochs > 0 else None
    main = None
    if has_main:
        scheduler_cls = getattr(optim.lr_scheduler, config.scheduler.type)
        main = scheduler_cls(optimizer, **config.scheduler.params)

    if warmup and main:
        return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])
    return warmup or main
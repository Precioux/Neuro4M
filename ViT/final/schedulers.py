import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):
    """
    Cosine LR schedule with warmup steps.
    """
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]

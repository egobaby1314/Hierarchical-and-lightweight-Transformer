# hooks/prune_hook.py
"""
Structured channel pruning hook (placeholder).

Real implementation would analyse BN gamma or weight magnitude after N epochs
and set `requires_grad=False` (or zero‐out) on the selected channels, then
re-initialize the optimizer. Here we do a no-op but leave a full API skeleton.
"""

from mmengine.hooks import Hook
from mmengine.runner import Runner

class StructuredPruningHook(Hook):
    """Stub for channel‐pruning; does nothing unless `prune_epoch` reached."""

    priority = 'NORMAL'

    def __init__(self, prune_epoch: int = 8, sparsity: float = 0.3):
        self.prune_epoch = prune_epoch
        self.sparsity = sparsity

    def after_train_epoch(self, runner: Runner):
        cur_epoch = runner.epoch + 1
        if cur_epoch == self.prune_epoch:
            runner.logger.info(
                f'[PruneHook] Would prune {self.sparsity*100:.0f}% channels here.'
            )
            # TODO: implement actual pruning logic

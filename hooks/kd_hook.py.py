# Copyright (c) 2025, Your-Lab
# Licensed under the Apache 2.0 License.
#
# A production-grade knowledge-distillation hook for MMDetection.
# --------------------------------------------------------------
from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample


class DetectionKDLossHook(Hook):
    """Online knowledge-distillation for one-stage detectors.

    Args
    ----
    teacher_cfg : str
        Path to the teacher config.
    teacher_ckpt : str
        Path to the teacher checkpoint.
    distill_weight_cls : float
        Weight for the KL divergence on classification logits.
    distill_weight_bbox : float
        Weight for the regression distillation loss.
    temperature : float, default=1.0
        Softmax temperature. When >1, softens probability distribution.
    """

    priority = 'NORMAL'  # after default losses are computed

    def __init__(
        self,
        teacher_cfg: str,
        teacher_ckpt: str,
        distill_weight_cls: float = 0.5,
        distill_weight_bbox: float = 0.5,
        temperature: float = 2.0,
    ) -> None:
        self.teacher_cfg = teacher_cfg
        self.teacher_ckpt = teacher_ckpt
        self.dw_cls = distill_weight_cls
        self.dw_bbox = distill_weight_bbox
        self.T = temperature

    # --------------------------------------------------------------------- #
    #                    runner-lifecycle interface                          #
    # --------------------------------------------------------------------- #
    def before_run(self, runner: Runner) -> None:
        """Load teacher once; put on same device and eval-only."""
        self.teacher = init_detector(
            self.teacher_cfg,
            self.teacher_ckpt,
            device=runner.model.device,
        )
        self.teacher.eval()
        # Disable grad for teacher entirely
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        """Compute KD loss and add it to runner's loss dict."""
        if (self.dw_cls == 0) and (self.dw_bbox == 0):
            return

        # ---------------- teacher forward (no grad) ---------------- #
        with torch.no_grad():
            teacher_out: Dict[str, torch.Tensor] = self.teacher(
                **data_batch, mode='loss'
            )

        kd_losses: List[torch.Tensor] = []

        # ---------- (1) KL loss on class logits (temperature scaled) -------- #
        if self.dw_cls > 0 and 'loss_cls' in outputs and 'loss_cls' in teacher_out:
            # teacher and student loss dicts contain sum-reduced CE; we need logits
            st_logits = outputs['loss_cls'].pop('logits', None)
            te_logits = teacher_out['loss_cls'].pop('logits', None)
            if st_logits is not None and te_logits is not None:
                kd_cls = F.kl_div(
                    F.log_softmax(st_logits / self.T, dim=-1),
                    F.softmax(te_logits / self.T, dim=-1),
                    reduction='batchmean',
                ) * (self.T**2)
                kd_losses.append(self.dw_cls * kd_cls)

        # ---------- (2) Smooth-L1 on bbox deltas --------------------------- #
        if self.dw_bbox > 0 and 'loss_bbox' in outputs and 'loss_bbox' in teacher_out:
            st_bbox = outputs['loss_bbox'].pop('bbox_pred', None)
            te_bbox = teacher_out['loss_bbox'].pop('bbox_pred', None)
            if st_bbox is not None and te_bbox is not None:
                kd_reg = F.smooth_l1_loss(st_bbox, te_bbox, reduction='mean')
                kd_losses.append(self.dw_bbox * kd_reg)

        # ---------- aggregate & inject into runner ------------------------- #
        if kd_losses:
            kd_total = torch.stack(kd_losses).sum()
            # MMEngine expects losses to be attached to runner.outputs
            outputs['loss'] += kd_total
            outputs['loss_kd'] = kd_total.detach()


__all__ = ['DetectionKDLossHook']

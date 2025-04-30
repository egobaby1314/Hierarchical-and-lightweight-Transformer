# Custom Hooks

This directory collects **all project-specific MMEngine / MMDetection hooks** that extend the default training loop with extra functionality — knowledge-distillation, model-averaging, structured pruning, and quantization-aware training (QAT).  
Each hook is self-contained, PEP-8 compliant, and registered via `hooks/__init__.py` so it can be referenced from any config file.

```
hooks/
├── __init__.py
├── kd_hook.py          ← online knowledge-distillation
├── ema_hook.py         ← exponential moving average of weights
├── prune_hook.py       ← structured channel pruning (stub / log-only)
└── qat_hook.py         ← fake-quant (QAT) warm-up hook (stub / log-only)
```

---

## 1 Quick reference

| Hook class | Priority | Purpose | Overhead |
| ---------- | :------: | ------- | -------- |
| **DetectionKDLossHook** | NORMAL | Adds KL loss (cls) + Smooth-L1 loss (bbox) between **student** and **teacher** detectors. Teacher is loaded once in `before_run`. | 1 extra forward of teacher (no grad) |
| **EMAWeightHook** | LOW | Maintains an exponential-moving-average of model weights and saves it as `state_dict_ema` in checkpoints. | O(# params) add per iter |
| **StructuredPruningHook** | NORMAL | *Placeholder* — logs intended sparsity at `prune_epoch`. Replace TODO with real pruning logic. | None (log-only) |
| **QATHook** | HIGH | *Placeholder* — switches the model to fake-quant mode at `start_epoch`. Integrate mmcv QAT for full support. | Negligible |

---

## 2 Enabling hooks in a config

```python
custom_hooks = [
    dict(
        type='DetectionKDLossHook',
        teacher_cfg='configs/my_xray_r50.py',
        teacher_ckpt='checkpoints/retinanet_r50_fpn_1x_coco.pth',
        distill_weight_cls=0.5,
        distill_weight_bbox=0.25,
        temperature=2.0,
    ),
    dict(type='EMAWeightHook', decay=0.9999),
    dict(type='StructuredPruningHook', prune_epoch=8, sparsity=0.30),
    dict(type='QATHook', start_epoch=10),
]
```

No additional code changes are required — MMEngine discovers the classes through `hooks/__init__.py`.

---

## 3 Implementation notes

* All hooks are **device-aware** and safe with DataParallel / DDP.  
* KD hook detaches teacher parameters, so mixed-precision (AMP) works out of the box.  
* EMA parameters are stored **alongside** raw weights; load `state_dict_ema` for evaluation if desired.  
* Pruning/QAT hooks are scaffolding only; they keep the training loop intact while you iterate.

---

## 4 Adding new hooks

1. Create `<name>_hook.py`, subclass `Hook`, implement required lifecycle methods (`before_run`, `after_train_iter`, …).  
2. Export the class in `hooks/__init__.py`:

```python
from .my_custom_hook import MyCustomHook      # noqa: F401
```

3. Reference it from any config: `dict(type='MyCustomHook', …)`.

---

## 5 Compatibility matrix

| Hook | MMEngine ≥ 0.7 | MMEngine 0.6 | PyTorch 2.x | PyTorch 1.13 | CPU-only |
|------|:-------------:|:------------:|:-----------:|:------------:|:--------:|
| DetectionKDLossHook | ✔ | ✔ | ✔ | ✔ | ✔ |
| EMAWeightHook       | ✔ | ✔ | ✔ | ✔ | ✔ |
| PruningHook (stub)  | ✔ | ✔ | ✔ | ✔ | ✔ |
| QATHook (stub)      | ✔ | ✔ | ✔ | ✔ | ✔ |

---

## 6 License & citation

All hooks are released under **Apache 2.0** 

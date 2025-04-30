"""
Register all custom hooks so MMEngine can discover them.
"""

from .kd_hook import DetectionKDLossHook          # noqa: F401
from .ema_hook import EMAWeightHook               # noqa: F401
from .prune_hook import StructuredPruningHook     # noqa: F401
from .qat_hook import QATHook                     # noqa: F401

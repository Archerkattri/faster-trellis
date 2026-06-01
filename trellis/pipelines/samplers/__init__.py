from .base import Sampler
# Stock TRELLIS flow-matching samplers.
from .flow_euler import (
    FlowEulerSampler,
    FlowEulerCfgSampler,
    FlowEulerGuidanceIntervalSampler,
)
# Our accelerated drop-ins: HiCache (Hermite velocity forecast) and the full
# HiCache + Adaptive-Guidance stack. Resolved by name from pipeline.json or
# via pipeline.enable_faster_mode(...).
from .faster_samplers import (
    FlowEulerGuidanceIntervalSampler_hicache,
    FlowEulerGuidanceIntervalSampler_faster as FlowEulerGuidanceIntervalSampler_fasterstack,
)

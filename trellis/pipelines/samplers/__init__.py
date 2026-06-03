from .base import Sampler
# Standard TRELLIS flow-matching samplers.
from .flow_euler import (
    FlowEulerSampler,
    FlowEulerCfgSampler,
    FlowEulerGuidanceIntervalSampler,
)
# Accelerated drop-in samplers: HiCache (Hermite velocity forecast) and the
# full HiCache + Adaptive-Guidance stack. Resolved by name from pipeline.json
# or via pipeline.enable_faster_mode(...).
from .faster_samplers import (
    FlowEulerGuidanceIntervalSampler_hicache,
    FlowEulerGuidanceIntervalSampler_faster as FlowEulerGuidanceIntervalSampler_fasterstack,
    FlowEulerGuidanceIntervalSampler_adaptive,
)
# Carved SLaT sampler (token carving + delta-cache) for the carved hybrid config
# = HiCache SS + carved SLaT (mirrors faster-trellis2).
from .flow_euler_carved import (
    FlowEulerSampler_carved,
    FlowEulerGuidanceIntervalSampler_carved,
)

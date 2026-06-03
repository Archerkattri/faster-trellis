"""Faster-TRELLIS accelerated Flow-Euler samplers.

Two training-free, drop-in accelerated variants of the TRELLIS v1
``FlowEulerGuidanceIntervalSampler`` are provided here as first-class
registered sampler classes (rather than call-site monkeypatches):

* ``FlowEulerGuidanceIntervalSampler_hicache`` -- HiCache Hermite-polynomial
  velocity forecasting (arXiv:2508.16984). Replaces the model forward pass on
  forecast steps with a dual-scaled physicist's-Hermite extrapolation of the
  cached final velocity.

* ``FlowEulerGuidanceIntervalSampler_faster`` -- the full stack: HiCache plus
  Adaptive Guidance (arXiv:2312.12487). On top of HiCache it also skips the
  unconditional CFG forward pass once the conditional/unconditional predictions
  align (cosine similarity >= gamma_bar), reconstructing the CFG guidance term
  from cached anchors. This is the default ``"faster"`` mode.

Both classes are subclasses of ``FlowEulerGuidanceIntervalSampler`` and
preserve its public ``sample(...)`` contract (CFG strength, guidance interval,
rescale_t, steps). They reuse the forecast math in the sibling ``hicache`` and
``adaptive_cfg`` modules, applied to the instance at construction time, so a
constructed instance is ready to use with no extra enable call. The
acceleration hyper-parameters are exposed as constructor arguments.
"""

from typing import *

from .flow_euler import FlowEulerGuidanceIntervalSampler
from . import hicache as _hicache
from . import adaptive_cfg as _adaptive_cfg


# Default schedule tuned on Toys4K at 25 steps.
_HICACHE_DEFAULTS = dict(
    interval=4,
    max_order=1,
    first_enhance=2,
    end_enhance=None,
    sigma=0.5,
)
_ADAPTIVE_CFG_DEFAULTS = dict(
    gamma_bar=0.94,
    warmup=2,
    max_order=1,
    reuse_guidance=True,
)


class _SinglePatchablePipelineView:
    """Adapter so the module-level ``enable(pipeline, ...)`` helpers, which
    expect a pipeline exposing ``sparse_structure_sampler`` / ``slat_sampler``,
    can be pointed at a single sampler instance.

    ``hicache.enable`` patches whichever of the two attributes is present;
    ``adaptive_cfg.enable`` additionally reads ``*_sampler_params`` for the
    step count. We expose the sampler under ``slat_sampler`` (the SparseTensor
    path) or ``sparse_structure_sampler`` (the dense path) depending on the
    declared substrate, and forward the step count.
    """

    def __init__(self, sampler, is_sparse: bool, steps: int):
        self._sampler = sampler
        if is_sparse:
            self.slat_sampler = sampler
            self.slat_sampler_params = {"steps": steps}
            self.sparse_structure_sampler = None
        else:
            self.sparse_structure_sampler = sampler
            self.sparse_structure_sampler_params = {"steps": steps}
            self.slat_sampler = None


class FlowEulerGuidanceIntervalSampler_hicache(FlowEulerGuidanceIntervalSampler):
    """Guidance-interval Flow-Euler sampler with HiCache velocity forecasting.

    Args:
        sigma_min: flow sigma_min (same as the base sampler).
        is_sparse: True if this sampler operates on ``SparseTensor`` latents
            (the SLaT stage); False for the dense sparse-structure stage.
        steps: nominal step count used to size the cache schedule (the actual
            run length still comes from the ``steps`` passed to ``sample``).
        hicache_kwargs: overrides for the HiCache schedule
            (``interval``/``max_order``/``first_enhance``/``end_enhance``/``sigma``).
    """

    def __init__(
        self,
        sigma_min: float,
        is_sparse: bool = True,
        steps: int = 25,
        **hicache_kwargs,
    ):
        super().__init__(sigma_min)
        self._is_sparse = is_sparse
        self._nominal_steps = int(steps)
        self._hicache_cfg = {**_HICACHE_DEFAULTS, **hicache_kwargs}
        self._install()

    def _install(self):
        view = _SinglePatchablePipelineView(self, self._is_sparse, self._nominal_steps)
        _hicache.enable(
            view,
            patch_slat=self._is_sparse,
            patch_sparse_structure=not self._is_sparse,
            **self._hicache_cfg,
        )


class FlowEulerGuidanceIntervalSampler_faster(FlowEulerGuidanceIntervalSampler):
    """Full-stack accelerated sampler: HiCache + Adaptive Guidance.

    HiCache forecasts the final velocity on skip steps; Adaptive Guidance drops
    the unconditional CFG pass once guidance converges and reconstructs the
    guidance term from cached anchors. The two are orthogonal: Adaptive Guidance
    wraps the per-step CFG forward (``_inference_model``); HiCache wraps the
    whole-step velocity (``sample_once``). Adaptive Guidance is installed first
    so HiCache's compute-step forward calls the AG-wrapped CFG path.

    Args / kwargs mirror the HiCache sampler, plus ``adaptive_cfg_kwargs``
    (``gamma_bar``/``warmup``/``max_order``/``reuse_guidance``).
    """

    def __init__(
        self,
        sigma_min: float,
        is_sparse: bool = True,
        steps: int = 25,
        hicache_kwargs: Optional[dict] = None,
        adaptive_cfg_kwargs: Optional[dict] = None,
    ):
        super().__init__(sigma_min)
        self._is_sparse = is_sparse
        self._nominal_steps = int(steps)
        self._hicache_cfg = {**_HICACHE_DEFAULTS, **(hicache_kwargs or {})}
        self._adaptive_cfg = {**_ADAPTIVE_CFG_DEFAULTS, **(adaptive_cfg_kwargs or {})}
        self._install()

    def _install(self):
        view = _SinglePatchablePipelineView(self, self._is_sparse, self._nominal_steps)
        # Order matters: Adaptive Guidance wraps _inference_model first so that
        # HiCache compute steps (which call _get_model_prediction ->
        # _inference_model) go through the guidance-skipping path.
        _adaptive_cfg.enable(
            view,
            patch_slat=self._is_sparse,
            patch_sparse_structure=not self._is_sparse,
            **self._adaptive_cfg,
        )
        _hicache.enable(
            view,
            patch_slat=self._is_sparse,
            patch_sparse_structure=not self._is_sparse,
            **self._hicache_cfg,
        )


class FlowEulerGuidanceIntervalSampler_adaptive(FlowEulerGuidanceIntervalSampler):
    """Adaptive Guidance only (CFG-skip), WITHOUT HiCache velocity forecasting.

    Drops the unconditional CFG pass once guidance converges and reconstructs the
    guidance term from cached anchors — the speed/quality profile of Adaptive
    Guidance in isolation. Mirrors the full-stack sampler but installs only
    ``adaptive_cfg`` (no ``hicache``).
    """

    def __init__(
        self,
        sigma_min: float,
        is_sparse: bool = True,
        steps: int = 25,
        adaptive_cfg_kwargs: Optional[dict] = None,
    ):
        super().__init__(sigma_min)
        self._is_sparse = is_sparse
        self._nominal_steps = int(steps)
        self._adaptive_cfg = {**_ADAPTIVE_CFG_DEFAULTS, **(adaptive_cfg_kwargs or {})}
        self._install()

    def _install(self):
        view = _SinglePatchablePipelineView(self, self._is_sparse, self._nominal_steps)
        _adaptive_cfg.enable(
            view,
            patch_slat=self._is_sparse,
            patch_sparse_structure=not self._is_sparse,
            **self._adaptive_cfg,
        )

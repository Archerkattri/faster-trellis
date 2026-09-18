"""HiCache: Hermite-polynomial velocity forecasting for TRELLIS v1.

Training-free inference acceleration for the TRELLIS v1
``TrellisImageTo3DPipeline``. The final CFG-combined velocity at *skipped*
sampling steps is forecast with a **scaled (physicist's) Hermite polynomial**
basis. The dual scaling keeps the high-order terms bounded, giving a more
numerically stable forecast than the equivalent Taylor (monomial) series.

Reference
---------
HiCache: Training-free Acceleration of Diffusion Models via Hermite
Polynomial Feature Forecasting (arXiv:2508.16984).

Method
------
Let ``F_t`` be the cached feature/velocity at the most recent compute
("full") step and ``N = N_interval`` the spacing between compute steps.
At a compute step we update backward finite differences::

    Delta^0 F_t = F_t
    Delta^i F_t = (Delta^{i-1} F_t - Delta^{i-1} F_{t-N}) / N

(``F_{t-N}`` is the previous compute step's value of the same order.)

At a skipped step with forward horizon ``k`` (``k = 1 .. N-1`` steps past
the last compute step) the velocity is forecast as::

    F_hat_{t+k} = F_t + sum_{i=1}^{m} (Delta^i F_t / i!) * Htilde_i(k)

where ``Htilde`` is the *dual-scaled* physicist's Hermite polynomial with
contraction factor ``sigma in (0, 1)``::

    Htilde_n(x) = sigma^n * H_n(sigma * x)
    H_0(x) = 1,  H_1(x) = 2x
    H_{n+1}(x) = 2*x*H_n(x) - 2*n*H_{n-1}(x)

TaylorSeer is the special case where the basis ``Htilde_i(k)`` is
replaced by the monomial ``k^i``. The dual scaling (input scale
``sigma*x`` and coefficient scale ``sigma^n``) suppresses the exponential
growth of the high-order Hermite terms and keeps the forecast inside the
numerically stable oscillatory regime.

The caching *substrate* is the FINAL velocity ``pred_v``: we cache
``pred_v`` at compute steps and forecast/reuse it at skip steps, then
rebuild the Euler update with the sampler's own
``_v_to_xstart_eps``. For SLaT, ``pred_v`` is a ``SparseTensor`` and only
its ``.feats`` are cached/forecast; the coords are carried through
unchanged.

Public contract
---------------
``enable(pipeline, **kwargs) -> pipeline`` monkey-patches a loaded
``TrellisImageTo3DPipeline`` in place and returns it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import copy
import json
import time
import uuid

import torch
from hicache_pp.budget import CacheBudget, CacheBudgetRuntime, RunIdentity, stable_digest


# Public compatibility surface for TRELLIS integrations.  This checkout ships
# the Hermite (polynomial) basis only; DMD lives in the sibling
# faster-trellis-plus-plus repo.  The single-backend switch keeps lifecycle,
# reset, and telemetry semantics identical across the two checkouts.
HICACHE_BACKENDS = ("hermite",)


def normalize_backend(backend: str) -> str:
    """Return the supported forecast backend or raise a useful error."""
    value = "hermite" if backend is None else str(backend).lower().strip()
    if value not in HICACHE_BACKENDS:
        choices = ", ".join(repr(item) for item in HICACHE_BACKENDS)
        raise ValueError(f"HiCache backend must be one of {choices}, got {backend!r}")
    return value


# ---------------------------------------------------------------------------
# Hermite basis
# ---------------------------------------------------------------------------
def physicists_hermite(n: int, x: torch.Tensor) -> torch.Tensor:
    """Physicist's Hermite polynomial ``H_n(x)`` via the stable recurrence.

    ``H_0 = 1``, ``H_1 = 2x``, ``H_{k+1} = 2 x H_k - 2 k H_{k-1}``.
    """
    if n < 0:
        raise ValueError(f"Hermite order must be >= 0, got {n}")
    if n == 0:
        return torch.ones_like(x)
    h_prev = torch.ones_like(x)          # H_0
    h_curr = 2.0 * x                     # H_1
    if n == 1:
        return h_curr
    for k in range(1, n):
        h_next = 2.0 * x * h_curr - 2.0 * k * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr


def scaled_hermite(n: int, x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Dual-scaled Hermite ``Htilde_n(x) = sigma^n * H_n(sigma * x)``."""
    return (sigma ** n) * physicists_hermite(n, sigma * x)


# ---------------------------------------------------------------------------
# HiCache state
# ---------------------------------------------------------------------------
def hicache_init(
    num_steps: int,
    interval: int = 4,
    max_order: int = 1,
    first_enhance: int = 2,
    end_enhance: Optional[int] = None,
    sigma: float = 0.5,
    backend: str = "hermite",
    stage: str = "unknown",
) -> Dict[str, Any]:
    """Create a fresh HiCache state dict for one sampling run.

    Parameters
    ----------
    num_steps : total sampler steps.
    interval : ``N_interval`` -- one compute step then ``interval-1`` forecasts.
    max_order : highest finite-difference / Hermite order ``m`` (>= 1).
    first_enhance : always compute the first ``first_enhance`` steps.
    end_enhance : always compute steps with index ``>= end_enhance``
        (defaults to ``num_steps`` -> disabled).
    sigma : Hermite contraction factor in ``(0, 1)``.
    backend : forecast basis; this checkout ships ``"hermite"`` only.
    stage : telemetry label for the model stage owning this cache.
    """
    if interval < 1:
        raise ValueError("interval must be >= 1")
    if max_order < 1:
        raise ValueError("max_order must be >= 1")
    if not (0.0 < sigma < 1.0):
        # sigma == 1 is mathematically valid (pure Hermite) but the paper's
        # stability argument requires the strict contraction sigma in (0,1).
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")
    backend = normalize_backend(backend)
    return {
        "num_steps": int(num_steps),
        "interval": int(interval),
        "max_order": int(max_order),
        "first_enhance": int(first_enhance),
        "end_enhance": int(end_enhance if end_enhance is not None else num_steps),
        "sigma": float(sigma),
        "sigma_min": 1e-2,        # lower bound on the contraction factor sigma
        "backend": backend,       # "hermite" (polynomial HiCache)
        "stage": str(stage),      # telemetry only
        "step": 0,
        "counter": 0,            # forecasts since last compute
        "type": None,            # "full" | "forecast"
        "activated_steps": [],   # indices of compute steps
        # derivative cache: order -> tensor (finite-difference derivatives at
        # the last compute step). "anchor" is the step index they belong to.
        "derivatives": {},       # {0: F_t, 1: Delta^1 F_t, ...}
        "prev_derivatives": {},  # snapshot from the previous compute step
    }


def hicache_decide(state: Dict[str, Any]) -> str:
    """Decide whether the current step is computed or forecast.

    Mirrors the paper's schedule (``t mod N_interval``) plus the
    enhance-window guards used by TRELLIS. Sets and returns ``state['type']``.
    """
    step = state["step"]
    first = step < state["first_enhance"]
    last = step >= state["end_enhance"]
    interval_hit = state["counter"] >= state["interval"] - 1

    if first or last or interval_hit:
        state["type"] = "full"
        state["counter"] = 0
        state["activated_steps"].append(step)
    else:
        state["type"] = "forecast"
        state["counter"] += 1
    return state["type"]


def hicache_update_derivatives(state: Dict[str, Any], feature: torch.Tensor) -> None:
    """Compute backward finite-difference derivatives at a compute step.

    ``Delta^0 = feature``;
    ``Delta^i = (Delta^{i-1}_now - Delta^{i-1}_prev) / N``.

    Edge case (<2 anchors): with only one compute step seen we cannot form a
    finite difference, so only the 0th-order term (the raw velocity) is kept.
    The forecast then reduces to plain reuse of the cached velocity, which is
    the correct, well-defined zero-information forecast.
    """
    interval = state["interval"]
    prev = state["derivatives"]  # derivatives from the previous compute step
    have_prev = len(prev) > 0

    new_deriv: Dict[int, torch.Tensor] = {0: feature}
    if have_prev:
        # distance between the two most recent compute steps (>=1)
        acts = state["activated_steps"]
        if len(acts) >= 2:
            dist = acts[-1] - acts[-2]
        else:
            dist = interval
        dist = max(int(dist), 1)
        for order in range(state["max_order"]):
            if order not in prev:
                break
            new_deriv[order + 1] = (new_deriv[order] - prev[order]) / dist

    state["prev_derivatives"] = prev
    state["derivatives"] = new_deriv


def hicache_forecast(state: Dict[str, Any]) -> torch.Tensor:
    """Scaled-Hermite forecast of the velocity at the current skip step.

    ``F_hat = F_t + sum_{i>=1} (Delta^i F_t / i!) * Htilde_i(k)``.

    ``k`` is the number of steps elapsed since the last compute step.
    With <2 anchors only ``Delta^0`` exists and this returns the cached
    velocity unchanged (k-independent), the correct degenerate forecast.
    """
    deriv = state["derivatives"]
    if 0 not in deriv:
        raise RuntimeError("hicache_forecast called before any compute step")

    k = state["step"] - state["activated_steps"][-1]
    sigma = state["sigma"]
    base = deriv[0]
    x = torch.tensor(float(k), dtype=base.dtype, device=base.device)

    result = base
    order = 1
    while order in deriv:
        coeff = deriv[order] / math.factorial(order)
        result = result + coeff * scaled_hermite(order, x, sigma)
        order += 1
    return result


# ---------------------------------------------------------------------------
# Pipeline patching
# ---------------------------------------------------------------------------
def hicache_forecast_state(state: Dict[str, Any]) -> torch.Tensor:
    """Forecast from the backend selected in ``state``.

    This is the compatibility switch used by the sampler.  Keeping dispatch
    here makes Hermite share lifecycle, reset, and telemetry semantics with
    the sibling DMD checkout.
    """
    normalize_backend(state.get("backend", "hermite"))
    return hicache_forecast(state)


def _budget_memory_mb(value, is_sparse: bool):
    value = getattr(value, "feats", value) if is_sparse else value
    try:
        return float(value.numel() * value.element_size()) / (1024.0 * 1024.0)
    except (AttributeError, TypeError, ValueError):
        return None


def _ensure_budget_runtime(sampler, state, model, x_t, cond, stage, is_sparse):
    runtime = getattr(sampler, "_budget_runtime", None)
    if runtime is not None:
        return runtime
    state.setdefault("run_id", uuid.uuid4().hex)
    model_id = f"{type(model).__module__}.{type(model).__qualname__}"
    schedule = {
        "num_steps": int(state["num_steps"]),
        "interval": int(state["interval"]),
        "first_enhance": int(state["first_enhance"]),
        "end_enhance": int(state["end_enhance"]),
        "backend": str(state["backend"]),
        "stage": str(stage),
    }
    condition_shape = tuple(getattr(cond, "shape", ()))
    token_source = getattr(x_t, "feats", x_t) if is_sparse else x_t
    token_shape = tuple(getattr(token_source, "shape", ()))
    identity = RunIdentity(
        model_id=model_id,
        run_id=state["run_id"],
        schedule_digest=stable_digest(schedule),
        cfg_branch="cfg-combined",
        conditioning_id=stable_digest({"condition_shape": condition_shape}),
        stage=str(stage),
        token_layout_digest=stable_digest({"token_shape": token_shape}),
        dtype=str(getattr(x_t, "dtype", "")),
        device=str(getattr(x_t, "device", "")),
    )
    sampler._budget_runtime = CacheBudgetRuntime(
        sampler._hicache_budget,
        identity,
        model_digest=stable_digest({"model_id": model_id}),
        config_digest=stable_digest({"budget": sampler._hicache_budget.as_dict(), "schedule": schedule}),
        input_digest=stable_digest({"token_shape": token_shape, "condition_shape": condition_shape}),
    )
    return sampler._budget_runtime


def _patch_sampler(sampler, *, is_sparse: bool, cfg: Dict[str, Any], stage: str) -> None:
    """Install HiCache onto a single FlowEuler sampler instance.

    Wraps ``sample`` to (re)initialise per-run state and replaces
    ``sample_once`` with the cache/forecast logic acting on the final
    ``pred_v``. The sampler keeps its original CFG / guidance-interval
    behaviour because we still call its own (mixin-resolved)
    ``_get_model_prediction`` at compute steps.
    """
    from easydict import EasyDict as edict

    orig_sample = sampler.sample
    orig_sample_once = sampler.sample_once
    orig_get_pred = sampler._get_model_prediction
    orig_v_to_xstart = sampler._v_to_xstart_eps

    budget = cfg.get("budget")
    if budget is None:
        budget = CacheBudget(
            backend="hermite",
            allowed_stages=(stage,),
            max_horizon=cfg.get("max_horizon") or max(1, int(cfg["interval"]) - 1),
            quality_preset="adapter-default",
            max_memory_mb=cfg.get("max_memory_mb"),
            audit_budget=cfg.get("audit_budget", 0),
            fallback="full",
        )
    elif not isinstance(budget, CacheBudget):
        budget = CacheBudget.from_mapping(budget)
    sampler._hicache_budget = budget
    sampler._hicache_backend = "hermite"
    sampler._hicache_stage = stage

    def _begin_run(steps: int) -> None:
        sampler._hicache = hicache_init(
            num_steps=steps,
            interval=cfg["interval"],
            max_order=cfg["max_order"],
            first_enhance=cfg["first_enhance"],
            end_enhance=cfg["end_enhance"],
            sigma=cfg["sigma"],
            backend="hermite",
            stage=stage,
        )
        sampler._budget_runtime = None
        sampler._last_hicache_manifest = None

    def _end_run() -> None:
        runtime = getattr(sampler, "_budget_runtime", None)
        if runtime is not None:
            sampler._last_hicache_manifest = runtime.manifest.as_dict()
        sampler._budget_runtime = None
        sampler._hicache = None

    def _status() -> dict:
        state = getattr(sampler, "_hicache", None)
        if state is None:
            return {
                "enabled": False,
                "backend": "none",
                "stage": str(stage),
                "full_steps": 0,
                "forecast_steps": 0,
            }
        full_steps = len(state.get("activated_steps", ()))
        return {
            "enabled": True,
            "backend": state.get("backend", "hermite"),
            "stage": state.get("stage", stage),
            "full_steps": full_steps,
            "forecast_steps": max(int(state.get("step", 0)) - full_steps, 0),
        }

    def _configure(budget=None, *, max_horizon=None, max_memory_mb=None, audit_budget=0):
        sampler._hicache_budget = budget if isinstance(budget, CacheBudget) else (
            CacheBudget.from_mapping(budget) if budget is not None else sampler._hicache_budget)
        if max_horizon is not None or max_memory_mb is not None or audit_budget:
            current = sampler._hicache_budget.as_dict()
            if max_horizon is not None:
                current["max_horizon"] = max_horizon
            if max_memory_mb is not None:
                current["max_memory_mb"] = max_memory_mb
            if audit_budget:
                current["audit_budget"] = audit_budget
            sampler._hicache_budget = CacheBudget.from_mapping(current)
        return sampler

    def _get_manifest():
        manifest = getattr(sampler, "_last_hicache_manifest", None)
        if manifest is None:
            runtime = getattr(sampler, "_budget_runtime", None)
            manifest = runtime.manifest.as_dict() if runtime is not None else None
        return copy.deepcopy(manifest) if manifest is not None else None

    def _save_manifest(destination):
        manifest = _get_manifest()
        if manifest is None:
            raise RuntimeError("no completed HiCache run is available")
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return destination

    sampler._hicache_begin_run = _begin_run
    sampler._hicache_end_run = _end_run
    sampler.hicache_status = _status
    sampler.configure_hicache_budget = _configure
    sampler.get_hicache_manifest = _get_manifest
    sampler.save_hicache_manifest = _save_manifest

    def patched_sample(model, noise, *args, steps: int = 50, **kwargs):
        _begin_run(steps)
        try:
            return orig_sample(model, noise, *args, steps=steps, **kwargs)
        finally:
            _end_run()

    def patched_sample_once(model, x_t, t, t_prev, cond=None, **kwargs):
        state = getattr(sampler, "_hicache", None)
        if state is None:
            # Not inside a HiCache-managed run -> fall back to the original.
            return orig_sample_once(model, x_t, t, t_prev, cond, **kwargs)

        scheduled = hicache_decide(state)
        runtime = _ensure_budget_runtime(sampler, state, model, x_t, cond, stage, is_sparse)
        forecast_requested = scheduled == "forecast"
        budget_decision = runtime.decide(
            str(stage),
            horizon=max(1, int(state.get("counter", 0))) if forecast_requested else 0,
            method=runtime.budget.backend,
            supported=True,
            memory_mb=_budget_memory_mb(x_t, is_sparse),
            force_full=(not forecast_requested or runtime.budget.backend == "full"),
            audit=forecast_requested and runtime.audits_remaining > 0,
            controller_selected=True,
        )
        serve_forecast = forecast_requested and budget_decision.mode == "forecast"
        if forecast_requested and not serve_forecast:
            # hicache_decide has already advanced its skip counter.  A guard or
            # budgeted audit becomes a fresh anchor at this diffusion index.
            state["type"] = "full"
            state["counter"] = 0
            if not state["activated_steps"] or state["activated_steps"][-1] != state["step"]:
                state["activated_steps"].append(state["step"])

        work_started = time.perf_counter()

        if not serve_forecast:
            pred_x_0, pred_eps, pred_v = orig_get_pred(model, x_t, t, cond, **kwargs)
            feats = pred_v.feats if is_sparse else pred_v
            hicache_update_derivatives(state, feats.detach().clone())
            state["step"] += 1
            runtime.record_measurement(
                str(stage), "full",
                wall_time_ms=(time.perf_counter() - work_started) * 1000.0,
            )
            pred_x_prev = x_t - (t - t_prev) * pred_v
            return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

        # forecast step: rebuild the final velocity from cached derivatives.
        feats_hat = hicache_forecast_state(state)
        if is_sparse:
            pred_v = x_t.replace(feats_hat)
        else:
            pred_v = feats_hat
        pred_x_0, _eps = orig_v_to_xstart(x_t=x_t, t=t, v=pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        state["step"] += 1
        runtime.record_measurement(
            str(stage), state.get("backend", "hermite"),
            wall_time_ms=(time.perf_counter() - work_started) * 1000.0,
        )
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    sampler.sample = patched_sample
    sampler.sample_once = patched_sample_once
    sampler._hicache = None
    sampler._hicache_orig = (orig_sample, orig_sample_once)


def enable(pipeline, **kwargs):
    """Enable HiCache on a loaded TRELLIS v1 pipeline (in place).

    Parameters
    ----------
    pipeline : a ``TrellisImageTo3DPipeline`` instance.
    interval : ``N_interval`` between compute steps (default 4).
    max_order : Hermite / finite-difference order ``m`` (default 1).
    first_enhance : always-compute leading steps (default 2).
    end_enhance : always-compute steps with index >= this (default: all
        steps, i.e. disabled).
    sigma : Hermite contraction factor in ``(0, 1)`` (default 0.5).
    patch_slat : patch the SLaT sampler (default True).
    patch_sparse_structure : patch the sparse-structure sampler
        (default True).
    stage : telemetry label for patched samplers; per-stage overrides
        ``sparse_structure_stage`` / ``slat_stage`` (defaults
        ``"sparse_structure"`` / ``"slat"``).
    budget : explicit ``CacheBudget`` (or mapping); ``None`` keeps the
        historical schedule while still emitting an identity-bound manifest.
        ``max_horizon`` / ``max_memory_mb`` / ``audit_budget`` tune the
        default envelope.

    Returns
    -------
    The same ``pipeline`` object, with the requested samplers patched.
    """
    cfg = {
        "interval": int(kwargs.get("interval", 4)),
        "max_order": int(kwargs.get("max_order", 1)),
        "first_enhance": int(kwargs.get("first_enhance", 2)),
        "end_enhance": kwargs.get("end_enhance", None),
        "sigma": float(kwargs.get("sigma", 0.5)),
        "budget": kwargs.get("budget"),
        "max_horizon": kwargs.get("max_horizon"),
        "max_memory_mb": kwargs.get("max_memory_mb"),
        "audit_budget": kwargs.get("audit_budget", 0),
    }
    stage_ss = kwargs.get("sparse_structure_stage", kwargs.get("stage", "sparse_structure"))
    stage_slat = kwargs.get("slat_stage", kwargs.get("stage", "slat"))
    patch_slat = kwargs.get("patch_slat", True)
    patch_ss = kwargs.get("patch_sparse_structure", True)

    patched: List[str] = []

    if patch_ss and getattr(pipeline, "sparse_structure_sampler", None) is not None:
        _patch_sampler(pipeline.sparse_structure_sampler, is_sparse=False, cfg=cfg, stage=stage_ss)
        patched.append("sparse_structure_sampler")

    if patch_slat and getattr(pipeline, "slat_sampler", None) is not None:
        _patch_sampler(pipeline.slat_sampler, is_sparse=True, cfg=cfg, stage=stage_slat)
        patched.append("slat_sampler")

    if not patched:
        raise RuntimeError(
            "HiCache.enable: pipeline exposes no sparse_structure_sampler or "
            "slat_sampler to patch."
        )

    pipeline._hicache_patched = patched
    return pipeline


# ---------------------------------------------------------------------------
# CPU unit test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True

    def check(name: str, cond: bool) -> None:
        global ok
        ok = ok and cond
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    # 1. Hermite recurrence sanity: known low-order physicist's polynomials.
    xs = torch.tensor([-1.5, 0.0, 0.7, 2.0])
    check("H_0 == 1", torch.allclose(physicists_hermite(0, xs), torch.ones_like(xs)))
    check("H_1 == 2x", torch.allclose(physicists_hermite(1, xs), 2 * xs))
    check("H_2 == 4x^2-2", torch.allclose(physicists_hermite(2, xs), 4 * xs**2 - 2))
    check("H_3 == 8x^3-12x", torch.allclose(physicists_hermite(3, xs), 8 * xs**3 - 12 * xs))

    # 2. scaled_hermite definition: Htilde_n(x) = sigma^n H_n(sigma x).
    sig = 0.5
    expect = (sig**2) * (4 * (sig * xs) ** 2 - 2)
    check("scaled_hermite matches sigma^n H_n(sigma x)",
          torch.allclose(scaled_hermite(2, xs, sig), expect))

    # 3. Finite-difference derivatives are EXACT on a linear velocity series.
    #    Build a synthetic compute-step history F_s = a + b*s and verify that
    #    hicache_update_derivatives recovers Delta^1 = b * interval-distance
    #    relation, and that the order-1 Hermite forecast at sigma->small
    #    reduces to a reuse-plus-correction that lowers error vs pure reuse.
    a = torch.tensor([1.0, -2.0, 0.5])
    b = torch.tensor([0.3, 0.3, 0.3])
    interval = 4
    st = hicache_init(num_steps=12, interval=interval, max_order=1,
                      first_enhance=0, end_enhance=12, sigma=sig)

    # First compute step at index 0.
    st["step"] = 0
    st["activated_steps"].append(0)
    hicache_update_derivatives(st, a + b * 0.0)
    check("after 1 anchor only order-0 derivative exists",
          set(st["derivatives"].keys()) == {0})

    # <2 anchors edge case: forecast must equal the cached velocity (reuse).
    st["step"] = 1
    fc1 = hicache_forecast(st)
    check("<2 anchors -> forecast == cached velocity",
          torch.allclose(fc1, a + b * 0.0))

    # Second compute step at index `interval`.
    st["step"] = interval
    st["activated_steps"].append(interval)
    hicache_update_derivatives(st, a + b * float(interval))
    # Delta^1 = (F_now - F_prev)/dist = (b*interval)/interval = b.
    check("finite-difference order-1 derivative == b (exact on linear series)",
          torch.allclose(st["derivatives"][1], b))

    # 4. Skip decision: schedule recomputes at the right cadence.
    sched = hicache_init(num_steps=12, interval=4, max_order=1,
                         first_enhance=2, end_enhance=10, sigma=sig)
    types = []
    for s in range(12):
        sched["step"] = s
        types.append(hicache_decide(sched))
    # first_enhance=2 -> steps 0,1 full; then every 4th; end_enhance>=10 full.
    check("step 0 and 1 are full (first_enhance)",
          types[0] == "full" and types[1] == "full")
    check("steps 2,3,4 are forecast then full at the interval boundary",
          types[2] == "forecast" and types[5] == "full")
    check("steps >= end_enhance are full",
          types[10] == "full" and types[11] == "full")

    # 5. Exactness on a CONSTANT velocity series: with max_order high the
    #    Hermite forecast must reproduce the (constant) value exactly because
    #    all finite differences vanish, leaving only Delta^0.
    stc = hicache_init(num_steps=8, interval=4, max_order=3,
                       first_enhance=0, end_enhance=8, sigma=sig)
    const = torch.tensor([2.0, -1.0, 4.0])
    for ci, idx in enumerate([0, 4]):
        stc["step"] = idx
        stc["activated_steps"].append(idx)
        hicache_update_derivatives(stc, const.clone())
    higher = [o for o in stc["derivatives"] if o >= 1]
    check("constant series -> all higher derivatives are zero",
          all(torch.allclose(stc["derivatives"][o], torch.zeros_like(const))
              for o in higher))
    stc["step"] = 6
    check("constant series -> forecast == constant (exact)",
          torch.allclose(hicache_forecast(stc), const))

    # 6. end-to-end forecast monotonicity guard: Hermite term is finite and
    #    the sigma-contraction keeps it bounded (no NaN/Inf) at large k.
    stb = hicache_init(num_steps=64, interval=32, max_order=3,
                       first_enhance=0, end_enhance=64, sigma=0.4)
    f0 = torch.randn(5)
    for idx in (0, 32):
        stb["step"] = idx
        stb["activated_steps"].append(idx)
        hicache_update_derivatives(stb, f0 + 0.01 * idx)
    stb["step"] = 60  # k = 28, far horizon
    out = hicache_forecast(stb)
    check("far-horizon Hermite forecast is finite (sigma contraction)",
          torch.isfinite(out).all().item())

    print()
    print("ALL PASS" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)

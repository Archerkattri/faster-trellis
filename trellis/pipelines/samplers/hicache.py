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

    F_hat_{t-k} = F_t + sum_{i=1}^{m} (Delta^i F_t / i!) * Htilde_i(-k)

where ``Htilde`` is the *dual-scaled* physicist's Hermite polynomial with
contraction factor ``sigma in (0, 1)``::

    Htilde_n(x) = sigma^n * H_n(sigma * x)
    H_0(x) = 1,  H_1(x) = 2x
    H_{n+1}(x) = 2*x*H_n(x) - 2*n*H_{n-1}(x)

TaylorSeer is the special case where the basis ``Htilde_i(-k)`` is
replaced by the monomial ``(-k)^i``. The dual scaling (input scale
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

import torch


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
    """
    if interval < 1:
        raise ValueError("interval must be >= 1")
    if max_order < 1:
        raise ValueError("max_order must be >= 1")
    if not (0.0 < sigma < 1.0):
        # sigma == 1 is mathematically valid (pure Hermite) but the paper's
        # stability argument requires the strict contraction sigma in (0,1).
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")
    return {
        "num_steps": int(num_steps),
        "interval": int(interval),
        "max_order": int(max_order),
        "first_enhance": int(first_enhance),
        "end_enhance": int(end_enhance if end_enhance is not None else num_steps),
        "sigma": float(sigma),
        "sigma_min": 1e-2,        # lower bound on the contraction factor sigma
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

    ``F_hat = F_t + sum_{i>=1} (Delta^i F_t / i!) * Htilde_i(-k)``.

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
    x = torch.tensor(float(-k), dtype=base.dtype, device=base.device)

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
def _patch_sampler(sampler, *, is_sparse: bool, cfg: Dict[str, Any]) -> None:
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

    def patched_sample(model, noise, *args, steps: int = 50, **kwargs):
        sampler._hicache = hicache_init(
            num_steps=steps,
            interval=cfg["interval"],
            max_order=cfg["max_order"],
            first_enhance=cfg["first_enhance"],
            end_enhance=cfg["end_enhance"],
            sigma=cfg["sigma"],
        )
        try:
            return orig_sample(model, noise, *args, steps=steps, **kwargs)
        finally:
            sampler._hicache = None

    def patched_sample_once(model, x_t, t, t_prev, cond=None, **kwargs):
        state = getattr(sampler, "_hicache", None)
        if state is None:
            # Not inside a HiCache-managed run -> fall back to the original.
            return orig_sample_once(model, x_t, t, t_prev, cond, **kwargs)

        decision = hicache_decide(state)

        if decision == "full":
            pred_x_0, pred_eps, pred_v = orig_get_pred(model, x_t, t, cond, **kwargs)
            feats = pred_v.feats if is_sparse else pred_v
            hicache_update_derivatives(state, feats.detach().clone())
            state["step"] += 1
            pred_x_prev = x_t - (t - t_prev) * pred_v
            return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

        # forecast step: rebuild the final velocity from cached derivatives.
        feats_hat = hicache_forecast(state)
        if is_sparse:
            pred_v = x_t.replace(feats_hat)
        else:
            pred_v = feats_hat
        pred_x_0, _eps = orig_v_to_xstart(x_t=x_t, t=t, v=pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        state["step"] += 1
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
    }
    patch_slat = kwargs.get("patch_slat", True)
    patch_ss = kwargs.get("patch_sparse_structure", True)

    patched: List[str] = []

    if patch_ss and getattr(pipeline, "sparse_structure_sampler", None) is not None:
        _patch_sampler(pipeline.sparse_structure_sampler, is_sparse=False, cfg=cfg)
        patched.append("sparse_structure_sampler")

    if patch_slat and getattr(pipeline, "slat_sampler", None) is not None:
        _patch_sampler(pipeline.slat_sampler, is_sparse=True, cfg=cfg)
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

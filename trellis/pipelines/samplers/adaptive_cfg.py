"""Adaptive-CFG: training-free acceleration for TRELLIS v1.

Implements *Adaptive Guidance* (Castillo et al., arXiv:2312.12487) on top of
the TRELLIS v1 ``FlowEulerGuidanceIntervalSampler`` (the guidance-interval
mixin path). The cached/forecast quantity is velocity-space: specifically the
CFG **guidance term** ``g = w * (v_cond - v_uncond)``.

------------------------------------------------------------------------------
The paper, exactly
------------------------------------------------------------------------------
Classifier-free guidance combines two model evaluations per step:

    v_cfg = (1 + w) * v_cond - w * v_uncond                                  (1)

where ``w == cfg_strength``, ``v_cond = model(x_t, t, cond)`` and
``v_uncond = model(x_t, t, neg_cond)``. The unconditional pass doubles the
cost.

Adaptive Guidance observes that the conditional and unconditional predictions
become increasingly *aligned* as sampling proceeds, measured by the cosine
similarity (the paper's gamma_t):

    gamma_t = <v_cond, v_uncond> / (||v_cond|| * ||v_uncond||)               (2)

with ``lim_{t->0} gamma_t = 1``. Once ``gamma_t >= gamma_bar`` (threshold in
[0, 1]) the unconditional forward pass carries no new directional information,
so AG drops it and proceeds with cheaper conditional-only updates. The vanilla
paper rule replaces ``v_cfg`` by ``v_cond`` on skipped steps. This halves the
CFG cost on every aligned step.

------------------------------------------------------------------------------
Caching substrate
------------------------------------------------------------------------------
Vanilla AG discards guidance entirely on skip steps (``v_cfg -> v_cond``). We
keep a strictly-superior, faithful reconstruction: the CFG *guidance term*

    g_t = v_cfg - v_cond = w * (v_cond - v_uncond)                           (3)

is a smooth function of the diffusion step. At a *compute* step we evaluate
both passes, cache ``g`` (anchored at the step index). At a *skip* step we do
only the conditional pass and **forecast** ``g_t`` from the cached anchors
instead of zeroing it, then return ``v_cond + g_forecast``. With the
held-constant (0th-order) forecast this reduces to vanilla AG; with >= 2
anchors we use a finite-difference (BDF2 / Newton-divided-difference)
extrapolation that is *exact* for polynomial guidance signals, which is what
the unit test asserts.

State lives in a plain dict; the cached object is the guidance term ``g`` (a
linear function of the final velocity), and skip decisions are driven by the
cosine similarity of the conditional and unconditional predictions -- the
correct substrate for *CFG* skipping specifically.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# tensor / SparseTensor helpers                                               #
# --------------------------------------------------------------------------- #
def _is_sparse(x: Any) -> bool:
    return hasattr(x, "feats") and hasattr(x, "replace")


def _flat(x: Any) -> torch.Tensor:
    """Return a 1-D float view of the numeric payload (handles SparseTensor)."""
    t = x.feats if _is_sparse(x) else x
    return t.reshape(-1).float()


def _payload(x: Any) -> torch.Tensor:
    return x.feats if _is_sparse(x) else x


def _with_payload(template: Any, feats: torch.Tensor) -> Any:
    """Rebuild an object of the same kind as ``template`` carrying ``feats``."""
    if _is_sparse(template):
        return template.replace(feats)
    return feats


def cosine_sim(a: Any, b: Any, eps: float = 1e-12) -> float:
    """gamma_t from Eq. (2): cosine similarity of two (possibly sparse) preds."""
    fa, fb = _flat(a), _flat(b)
    num = torch.dot(fa, fb)
    den = fa.norm() * fb.norm() + eps
    return float((num / den).item())


# --------------------------------------------------------------------------- #
# forecast math                                                               #
# --------------------------------------------------------------------------- #
def forecast_guidance(
    anchors: List[Tuple[int, torch.Tensor]],
    step: int,
    max_order: int = 1,
) -> torch.Tensor:
    """Forecast the guidance term ``g`` at integer ``step`` from cached anchors.

    ``anchors`` is an ordered list ``[(step_i, g_i), ...]`` of *computed*
    guidance terms. We build the Newton forward-difference / divided-difference
    extrapolator through the most recent ``max_order + 1`` anchors. This is the
    unique polynomial of degree ``<= max_order`` interpolating those anchors and
    is therefore *exact* whenever the true guidance series is polynomial of that
    degree -- the property asserted by the unit test.

    Edge cases (explicit, no silent fallback):
      * 0 anchors -> ValueError (caller must guarantee >= 1).
      * 1 anchor  -> 0th-order hold (returns the single cached value); this is
                     exactly the vanilla-AG behaviour (g held constant).
      * >= 2 anchors with max_order >= 1 -> divided-difference extrapolation.
    """
    if len(anchors) == 0:
        raise ValueError("forecast_guidance requires at least one anchor")

    # 0th order or a single anchor -> hold the most recent value constant.
    if len(anchors) == 1 or max_order < 1:
        return anchors[-1][1].clone()

    # Use the last (max_order + 1) anchors.
    used = anchors[-(max_order + 1):]
    xs = [float(s) for s, _ in used]
    ys = [g.clone() for _, g in used]
    n = len(used)

    # Newton's divided differences: dd[0] = ys (copied above), then
    # dd[k][i] = (dd[k-1][i+1] - dd[k-1][i]) / (xs[i+k] - xs[i]).
    coeffs = [ys[0]]                     # leading coefficient column
    col = ys
    for k in range(1, n):
        new_col = []
        for i in range(n - k):
            denom = xs[i + k] - xs[i]
            new_col.append((col[i + 1] - col[i]) / denom)
        col = new_col
        coeffs.append(col[0])

    # Evaluate the Newton polynomial at ``step`` (Horner-like, ascending basis).
    x = float(step)
    result = coeffs[-1].clone()
    for k in range(n - 2, -1, -1):
        result = result * (x - xs[k]) + coeffs[k]
    return result


# --------------------------------------------------------------------------- #
# state                                                                       #
# --------------------------------------------------------------------------- #
def adaptive_cfg_init(
    num_steps: int,
    gamma_bar: float = 0.94,
    warmup: int = 2,
    max_order: int = 1,
    reuse_guidance: bool = True,
) -> Dict[str, Any]:
    """Create per-``sample()`` state for one diffusion trajectory.

    Args:
        num_steps:      total Euler steps (for bookkeeping / final-step force).
        gamma_bar:      cosine-similarity threshold from the paper, in [0, 1].
                        Higher -> more conservative (fewer skips).
        warmup:         force full CFG for the first ``warmup`` steps so the
                        forecast has anchors before any skip is allowed.
        max_order:      polynomial order of the guidance forecast (1 = BDF2-like
                        linear extrapolation; 0 = vanilla-AG hold).
        reuse_guidance: if True, skip steps return ``v_cond + forecast(g)``; if
                        False, they return ``v_cond`` (literal paper AG).
    """
    if not (0.0 <= gamma_bar <= 1.0):
        raise ValueError(f"gamma_bar must be in [0,1], got {gamma_bar}")
    return {
        "num_steps": int(num_steps),
        "gamma_bar": float(gamma_bar),
        "warmup": int(warmup),
        "max_order": int(max_order),
        "reuse_guidance": bool(reuse_guidance),
        "step": 0,
        "anchors": [],          # list[(step, g)] cached guidance terms
        "last_gamma": None,
        "n_full": 0,
        "n_skip": 0,
    }


def adaptive_cfg_decide(state: Dict[str, Any], gamma: Optional[float]) -> bool:
    """Return True if the current step must run the full (uncond) CFG pass.

    A skip requires ALL of: past warmup, at least one cached anchor, not the
    final step, and the last measured cosine similarity above threshold.
    """
    step = state["step"]
    if step < state["warmup"]:
        return True
    if step >= state["num_steps"] - 1:          # always anchor the final step
        return True
    if len(state["anchors"]) == 0:              # <2-anchor edge case: no skip
        return True
    if gamma is None:
        return True
    return gamma < state["gamma_bar"]


# --------------------------------------------------------------------------- #
# the patched inference path                                                  #
# --------------------------------------------------------------------------- #
def _make_inference_model(sampler, state_holder):
    """Build the replacement ``_inference_model`` bound onto ``sampler``.

    Signature mirrors ``GuidanceIntervalSamplerMixin._inference_model`` so it is
    a drop-in along the guidance-interval mixin path. ``super()._inference_model``
    of the mixin is the plain ``FlowEulerSampler._inference_model``; we reach it
    through the saved original conditional path.
    """
    def _inference_model(self, model, x_t, t, cond, neg_cond,
                         cfg_strength, cfg_interval, **kwargs):
        state = state_holder["state"]

        # Outside the guidance interval: behave exactly like the stock mixin
        # (single conditional pass, no CFG, no caching) and do NOT advance the
        # adaptive step counter -- those steps are not CFG steps.
        if not (cfg_interval[0] <= t <= cfg_interval[1]):
            return self._adaptive_base_inference(model, x_t, t, cond, **kwargs)

        # Lazily init state on first CFG step of this trajectory.
        if state is None:
            state = adaptive_cfg_init(
                num_steps=state_holder["num_steps"],
                **state_holder["kwargs"],
            )
            state_holder["state"] = state

        v_cond = self._adaptive_base_inference(model, x_t, t, cond, **kwargs)

        run_full = adaptive_cfg_decide(state, state["last_gamma"])

        if run_full:
            v_uncond = self._adaptive_base_inference(
                model, x_t, t, neg_cond, **kwargs)
            # gamma for the NEXT decision (Eq. 2).
            state["last_gamma"] = cosine_sim(v_cond, v_uncond)
            # guidance term g = w * (v_cond - v_uncond)  (Eq. 3, payload space).
            g = cfg_strength * (_payload(v_cond) - _payload(v_uncond))
            state["anchors"].append((state["step"], g))
            # bound memory: keep only what the forecast order needs.
            keep = state["max_order"] + 2
            if len(state["anchors"]) > keep:
                state["anchors"] = state["anchors"][-keep:]
            v_payload = (1.0 + cfg_strength) * _payload(v_cond) \
                - cfg_strength * _payload(v_uncond)
            out = _with_payload(v_cond, v_payload)
            state["n_full"] += 1
        else:
            # Skip the unconditional pass. Reconstruct guidance.
            if state["reuse_guidance"]:
                g = forecast_guidance(
                    state["anchors"], state["step"], state["max_order"])
                v_payload = _payload(v_cond) + g
                out = _with_payload(v_cond, v_payload)
            else:
                out = v_cond                      # literal paper AG: v_cfg<-v_cond
            state["n_skip"] += 1

        state["step"] += 1
        return out

    return _inference_model


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
def _patch_sampler(sampler, num_steps: int, **kwargs) -> None:
    """Monkey-patch one ``FlowEulerGuidanceIntervalSampler`` instance in place."""
    if getattr(sampler, "_adaptive_cfg_patched", False):
        # idempotent: refresh config, reset trajectory state.
        sampler._adaptive_cfg_holder["kwargs"] = kwargs
        sampler._adaptive_cfg_holder["num_steps"] = num_steps
        sampler._adaptive_cfg_holder["state"] = None
        return

    # The base single-pass evaluator (no mixin). On a
    # FlowEulerGuidanceIntervalSampler the mixin overrides _inference_model and
    # calls super()._inference_model -> FlowEulerSampler._inference_model. We
    # capture that base bound method so skip steps cost exactly one pass.
    from trellis.pipelines.samplers.flow_euler import FlowEulerSampler
    base = FlowEulerSampler._inference_model

    def _adaptive_base_inference(self, model, x_t, t, cond, **kw):
        return base(self, model, x_t, t, cond, **kw)

    sampler._adaptive_base_inference = _adaptive_base_inference.__get__(
        sampler, type(sampler))

    holder = {"state": None, "num_steps": num_steps, "kwargs": kwargs}
    sampler._adaptive_cfg_holder = holder

    new_inf = _make_inference_model(sampler, holder)
    sampler._adaptive_cfg_orig_inference = sampler._inference_model
    sampler._inference_model = new_inf.__get__(sampler, type(sampler))
    sampler._adaptive_cfg_patched = True


def enable(pipeline, **kwargs):
    """Enable Adaptive-CFG on a loaded TRELLIS v1 ``TrellisImageTo3DPipeline``.

    Monkey-patches the SLaT sampler and (optionally) the sparse-structure
    sampler so each one's guidance-interval CFG path skips the unconditional
    forward pass on aligned steps, reconstructing the guidance term from cached
    final-velocity guidance anchors.

    Args:
        pipeline: a loaded ``TrellisImageTo3DPipeline``.

    Keyword Args:
        gamma_bar (float):      cosine-sim threshold in [0,1] (default 0.94).
        warmup (int):           full-CFG warm-up steps (default 2).
        max_order (int):        guidance forecast order (default 1).
        reuse_guidance (bool):  reconstruct guidance on skip (True, default) vs
                                literal paper drop-to-conditional (False).
        patch_slat (bool):      patch ``slat_sampler`` (default True).
        patch_sparse_structure (bool): patch ``sparse_structure_sampler``
                                (default True).

    Returns:
        the same ``pipeline``, mutated in place.
    """
    patch_slat = kwargs.pop("patch_slat", True)
    patch_ss = kwargs.pop("patch_sparse_structure", True)

    targets = []
    if patch_ss:
        ss = getattr(pipeline, "sparse_structure_sampler", None)
        if ss is not None:
            steps = pipeline.sparse_structure_sampler_params.get("steps", 50)
            targets.append((ss, steps))
    if patch_slat:
        slat = getattr(pipeline, "slat_sampler", None)
        if slat is not None:
            steps = pipeline.slat_sampler_params.get("steps", 50)
            targets.append((slat, steps))

    if not targets:
        raise RuntimeError(
            "adaptive_cfg.enable: pipeline has no slat/sparse_structure sampler")

    for sampler, steps in targets:
        if not hasattr(sampler, "_inference_model"):
            raise TypeError(
                f"{type(sampler).__name__} has no _inference_model; "
                "adaptive_cfg requires a guidance-interval Flow-Euler sampler")
        _patch_sampler(sampler, num_steps=steps, **kwargs)

    return pipeline


def disable(pipeline):
    """Restore the original ``_inference_model`` on any patched samplers."""
    for name in ("sparse_structure_sampler", "slat_sampler"):
        sampler = getattr(pipeline, name, None)
        if sampler is not None and getattr(sampler, "_adaptive_cfg_patched", False):
            sampler._inference_model = sampler._adaptive_cfg_orig_inference
            del sampler._adaptive_cfg_patched
            del sampler._adaptive_cfg_holder
            del sampler._adaptive_cfg_orig_inference
            del sampler._adaptive_base_inference
    return pipeline


# --------------------------------------------------------------------------- #
# CPU unit test (no GPU, no TRELLIS model)                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        global ok
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    D = 32

    # 1) Order-1 (linear) forecast is EXACT for a linear guidance series.
    #    g(s) = A + B * s.  Anchors at s = 3,4 ; predict s = 5,7.
    A = torch.randn(D)
    B = torch.randn(D)
    g = lambda s: A + B * float(s)
    anchors = [(3, g(3)), (4, g(4))]
    for s in (5, 6, 7):
        pred = forecast_guidance(anchors, s, max_order=1)
        check(f"linear forecast exact @ step {s}",
              torch.allclose(pred, g(s), atol=1e-4))

    # 2) Order-2 forecast is EXACT for a quadratic series.
    A2, B2, C2 = torch.randn(D), torch.randn(D), torch.randn(D)
    q = lambda s: A2 + B2 * float(s) + C2 * float(s) ** 2
    anc2 = [(2, q(2)), (3, q(3)), (4, q(4))]
    for s in (5, 7):
        pred = forecast_guidance(anc2, s, max_order=2)
        check(f"quadratic forecast exact @ step {s}",
              torch.allclose(pred, q(s), atol=1e-3))

    # 3) <2-anchor edge cases: 1 anchor -> hold; 0 anchors -> ValueError.
    one = [(5, g(5))]
    check("single-anchor hold == cached value",
          torch.allclose(forecast_guidance(one, 99, max_order=1), g(5)))
    try:
        forecast_guidance([], 0)
        check("zero-anchor raises", False)
    except ValueError:
        check("zero-anchor raises", True)

    # 4) cosine_sim: identical vectors -> 1, anti-parallel -> -1.
    v = torch.randn(1, D)
    check("cosine self == 1", abs(cosine_sim(v, v) - 1.0) < 1e-5)
    check("cosine anti == -1", abs(cosine_sim(v, -v) + 1.0) < 1e-5)

    # 5) Decision logic.
    st = adaptive_cfg_init(num_steps=10, gamma_bar=0.9, warmup=2, max_order=1)
    st["step"] = 0
    check("warmup forces full (step 0)", adaptive_cfg_decide(st, 0.99) is True)
    st["step"] = 3
    # no anchors yet -> must compute even if gamma high
    check("no-anchor forces full", adaptive_cfg_decide(st, 0.99) is True)
    st["anchors"].append((2, torch.zeros(D)))
    check("aligned (gamma>=bar) -> skip",
          adaptive_cfg_decide(st, 0.95) is False)
    check("misaligned (gamma<bar) -> full",
          adaptive_cfg_decide(st, 0.80) is True)
    st["step"] = 9  # final step (num_steps-1)
    check("final step forces full", adaptive_cfg_decide(st, 0.99) is True)

    # 6) End-to-end velocity reconstruction on skip equals true v_cfg when the
    #    guidance term is linear in step (mimics the patched path's math).
    w = 3.0
    # Build two compute anchors of g = w*(v_cond - v_uncond), linear in step.
    gA = lambda s: A + B * float(s)            # this is g already
    state = adaptive_cfg_init(num_steps=10, gamma_bar=0.0, warmup=0, max_order=1)
    state["anchors"] = [(0, gA(0)), (1, gA(1))]
    state["step"] = 2
    v_cond = torch.randn(D)
    v_skip = v_cond + forecast_guidance(state["anchors"], 2, 1)
    v_true = v_cond + gA(2)                     # what full CFG would yield
    check("skip-step v reconstruction exact (linear g)",
          torch.allclose(v_skip, v_true, atol=1e-4))

    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    import sys
    sys.exit(0 if ok else 1)

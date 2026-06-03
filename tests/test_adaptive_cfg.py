"""CPU unit tests for the Adaptive-Guidance forecast math (no GPU, no TRELLIS model).

Run with::

    python -m tests.test_adaptive_cfg
    # or
    python tests/test_adaptive_cfg.py

Asserts Newton divided-difference exactness on linear / quadratic guidance
series, the single-/zero-anchor edge cases, cosine-similarity extremes, the
CFG-skip decision logic, and end-to-end skip-step velocity reconstruction.
"""

import torch

from trellis.pipelines.samplers.adaptive_cfg import (
    forecast_guidance,
    cosine_sim,
    adaptive_cfg_init,
    adaptive_cfg_decide,
)


def main() -> int:
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
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
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

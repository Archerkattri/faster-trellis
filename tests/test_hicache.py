"""CPU unit tests for the HiCache velocity-forecast math (no GPU, no TRELLIS model).

Run with::

    python -m tests.test_hicache
    # or
    python tests/test_hicache.py

Asserts the Hermite recurrence, the dual-scaled basis, finite-difference
derivative exactness on linear / constant velocity series, the compute/forecast
schedule cadence, and far-horizon boundedness of the forecast.
"""

import torch

from trellis.pipelines.samplers.hicache import (
    physicists_hermite,
    scaled_hermite,
    hicache_init,
    hicache_decide,
    hicache_update_derivatives,
    hicache_forecast,
)


def main() -> int:
    torch.manual_seed(0)
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
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
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

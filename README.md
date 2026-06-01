<div align="center">

# ⚡ faster-trellis

**~3.5× faster [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) image-to-3D — no retraining, no weight edits, same output formats.**

`TRELLIS-image-large` · mesh + Gaussian + radiance-field · training-free · single RTX 5090 · MIT

</div>

`faster-trellis` drops two training-free accelerators onto TRELLIS v1's two flow-matching
samplers. They cache and forecast the model's **final CFG-combined velocity**, so fewer
network evaluations run per diffusion trajectory while the weights, decoders, and output
formats stay byte-for-byte identical to stock TRELLIS.

```python
pipeline.enable_faster_mode("faster")     # one line → ~3.5× faster, quality within noise
```

---

## Pick a mode

| `enable_faster_mode(...)` | what it does | speedup | use it when |
|---|---|:--:|---|
| `"faster"` **(default)** | HiCache **+** Adaptive Guidance — the full stack | **3.53×** | always — fastest, quality within noise |
| `"hicache"` | HiCache only (no CFG-skip) | **2.76×** | max-quality safety toggle |
| `"none"` | restore stock TRELLIS samplers | 1.00× | baseline |

> **TL;DR:** keep `"faster"`. Across 40 Toys4K objects the geometry differences between modes
> are within noise — the win is latency, not reconstruction quality.

---

## Quickstart

```bash
git clone --recurse-submodules https://github.com/Archerkattri/faster-trellis
cd faster-trellis
# TRELLIS deps (CUDA toolchain required); see setup.sh for the full option list.
. ./setup.sh --new-env --basic --xformers --flash-attn --spconv --nvdiffrast
```

```python
from trellis.pipelines import TrellisImageTo3DPipeline

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large").cuda()
pipeline.enable_faster_mode("faster")                 # ← the only added line

outputs = pipeline.run(image, formats=["mesh", "gaussian", "radiance_field"])
```

Runnable scripts: `example.py` (`--mode {faster,hicache,none}`) and
`example_faster.py` (minimal, annotated).

<details>
<summary><b>Blackwell (RTX 50-series / sm_120) note</b></summary>

Default flash-attn / sparse-conv kernels may not ship SM_120 builds yet. If sparse
convolution fails to load, force the spconv backend:

```bash
export SPARSE_CONV_BACKEND=spconv
```
</details>

---

## Benchmarks

40 Toys4K objects, **RTX 5090**, 25 steps/stage. CD ↓ lower is better; F1@0.05 / vIoU ↑ higher
is better. `vIoU` = surface-shell occupancy IoU (identical metric across all variants).

| mode | CD ↓ | F1@0.05 ↑ | vIoU ↑ | latency | speedup |
|---|:--:|:--:|:--:|:--:|:--:|
| `none` (vanilla TRELLIS) | 0.236 | 0.343 | 0.052 | 4.04 s | 1.00× |
| TaylorSeer¹ | 0.237 | 0.342 | 0.049 | 1.54 s | 2.61× |
| `hicache` | 0.235 | 0.346 | 0.048 | 1.46 s | 2.76× |
| **`faster`** | 0.231 | 0.350 | 0.049 | 1.14 s | **3.53×** |

<sub>¹ Taylor-basis reference on the same caching substrate — the method `hicache` improves on.
The `faster` row ran 39/40 objects (`hamburger` OOM'd at 25 steps and is excluded). Quality
deltas across modes are within noise; the win is latency.</sub>

---

## How it works

TRELLIS v1 samples a 3D asset in two flow-matching stages (sparse structure, then structured
latent), each with classifier-free guidance. Most of the time goes to redundant model passes —
the velocity field is smooth in the diffusion step, and the conditional / unconditional
predictions converge as sampling proceeds. The two accelerators exploit exactly those.

<details>
<summary><b>① HiCache — Hermite velocity forecast</b> (removes skip-step network calls)</summary>

At a **compute** step we cache the final CFG-combined velocity `F_t` and update backward
finite differences (`N` = compute interval):

```
Δ⁰F_t = F_t
ΔⁱF_t = (Δⁱ⁻¹F_t − Δⁱ⁻¹F_{t−N}) / N
```

At a **skip** step (`k` past the last compute step) the velocity is extrapolated with the
dual-scaled physicist's Hermite basis instead of calling the model:

```
F̂_{t−k} = F_t + Σ_{i=1..m} (ΔⁱF_t / i!) · H̃_i(−k)
H̃_n(x) = σⁿ · H_n(σ·x),   σ ∈ (0,1)
H_0 = 1,  H_1 = 2x,  H_{n+1} = 2x·H_n − 2n·H_{n−1}
```

**Why Hermite > Taylor.** A Taylor (monomial) forecast uses `(−k)ⁱ`, whose terms grow without
bound as the horizon `k` increases — high-order corrections amplify finite-difference noise and
the forecast diverges. The Hermite polynomials are the eigenfunctions of the diffusion operator
governing the velocity field; the dual scaling (input `σx` **and** coefficient `σⁿ`, `σ∈(0,1)`)
contracts high-order terms into the numerically stable oscillatory regime. The forecast stays
bounded and accurate at longer skip horizons, which is what lets us widen the compute interval.
Taylor is the monomial special case. *(arXiv:2508.16984)*
</details>

<details>
<summary><b>② Adaptive Guidance — skip the unconditional pass</b> (halves CFG cost on compute steps)</summary>

CFG costs two forward passes per step: `v_cfg = (1+w)·v_cond − w·v_uncond`. As sampling
proceeds `v_cond` and `v_uncond` align — cosine similarity `γ_t → 1`. Once `γ_t ≥ γ̄` the
uncond pass carries no new directional information and is dropped.

Rather than zeroing guidance on skip steps, we cache the guidance term
`g = w·(v_cond − v_uncond)` at full steps and reconstruct it via Newton divided-difference
extrapolation (exact for polynomial guidance signals) — strictly better than the literal
drop-to-conditional rule, at the same cost. *(arXiv:2312.12487)*

**Composition:** the two are orthogonal — Adaptive Guidance halves each *compute* step's CFG;
HiCache removes the *skip* steps entirely. Stacking both is the `"faster"` mode.
</details>

---

## Tuning

Defaults live in the accelerated sampler config
(`trellis/pipelines/samplers/faster_samplers.py`):

| knob | default | meaning |
|---|:--:|---|
| `interval` | `4` | compute 1 step, then forecast `interval − 1` |
| `sigma` | `0.5` | Hermite contraction `σ ∈ (0,1)` |
| `max_order` | `1` | Hermite / finite-difference order |
| `gamma_bar` | `0.94` | cosine-similarity CFG-skip threshold |
| `warmup` | — | full-CFG warm-up steps before any skip |
| `reuse_guidance` | — | reconstruct vs drop the guidance term on skip |

---

## What changed vs clean TRELLIS

All Microsoft TRELLIS model / decoder code is **unmodified**. Added under
`trellis/pipelines/samplers/`: `hicache.py`, `adaptive_cfg.py`,
`faster_samplers.py` (the accelerated sampler classes), plus the `enable_faster_mode()` API in
`trellis/pipelines/trellis_image_to_3d.py`. The accelerators are independent re-implementations
of the cited papers — **no Fast-TRELLIS code is present**.

---

## Credits & license

| | |
|---|---|
| **TRELLIS** | Xiang et al., *Structured 3D Latents for Scalable and Versatile 3D Generation*, CVPR 2025 — [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) (MIT). `faster-trellis` is a thin acceleration layer on their pipeline and weights. |
| **HiCache** | arXiv:2508.16984 — Hermite-polynomial velocity forecasting |
| **Adaptive Guidance** | Castillo et al., arXiv:2312.12487 — unconditional-pass skipping |

MIT (see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Accelerations © 2026 Krishi Attri;
TRELLIS pipeline / architecture / weights © their authors.

**Krishi Attri** · krishiattriwork@gmail.com · [github.com/Archerkattri](https://github.com/Archerkattri)

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@misc{attri2026fastertrellis,
  title  = {faster-trellis: Training-free Hermite-Cache acceleration for TRELLIS image-to-3D},
  author = {Krishi Attri}, year = {2026},
  howpublished = {\url{https://github.com/Archerkattri/faster-trellis}}
}
@inproceedings{xiang2025trellis,
  title     = {Structured 3D Latents for Scalable and Versatile 3D Generation},
  author    = {Xiang, Jianfeng and others}, booktitle = {CVPR}, year = {2025}
}
@article{hicache2025,
  title   = {HiCache: Training-free Acceleration of Diffusion Models via
             Hermite Polynomial Feature Forecasting},
  journal = {arXiv preprint arXiv:2508.16984}, year = {2025}
}
@article{castillo2023adaptive,
  title   = {Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models},
  author  = {Castillo, Angela and others},
  journal = {arXiv preprint arXiv:2312.12487}, year = {2023}
}
```
</details>

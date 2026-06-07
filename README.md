<div align="center">

<img src="assets/banner.png" alt="faster-trellis" width="680">

# ⚡ faster-trellis

**Training-free acceleration for [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) image-to-3D — faster *and* higher-quality than Fast-TRELLIS, in one line of code.**

`TRELLIS-image-large` · mesh + Gaussian + radiance-field · training-free · single RTX 5090 · MIT

</div>

> **HiCache++ variant:** an exponential (DMD/Prony) forecast variant of this repo lives in [`faster-trellis-plus-plus`](https://github.com/Archerkattri/faster-trellis-plus-plus) — same carved-hybrid, with the sparse-structure velocity forecast on a Dynamic-Mode-Decomposition basis instead of the Hermite polynomial.

## When to use this repo

These repos are **complementary accelerators, not competing solutions** — each speeds up a *different*
base generator, and the `+` / `++` suffix is a **method choice**, not a rival product. Pick by
**(1) which base model you run**, then **(2) which forecast basis you want**:

| base generator | `+` = HiCache (Hermite) | `++` = HiCache++ (DMD) |
|---|---|---|
| Hunyuan3D-2.1 | `hunyuan2.1-plus` | `hunyuan2.1-plus-plus` |
| Hunyuan3D-2 mini | `hunyuan2-plus` | `hunyuan2-plus-plus` |
| SAM 3D Objects | `sam3d-plus` | `sam3d-plus-plus` |
| Fast-SAM3D | `fastsam3d-plus` | `fastsam3d-plus-plus` |
| DiT-XL/2 (ImageNet) | `dit-plus` | `dit-plus-plus` |
| TRELLIS (v1) | `faster-trellis` | `faster-trellis-plus-plus` |
| TRELLIS.2-4B (v2) | `hermit-trellis2` | `hermit-trellis2-plus-plus` |

- **`+` (HiCache / scaled-Hermite):** the *published* polynomial velocity-forecast basis — conservative, reproduces the HiCache paper. Use it to deploy the established method.
- **`++` (HiCache++ / DMD exponential):** our Dynamic-Mode-Decomposition basis — *the same near-lossless quality at wider skip intervals*, where the polynomial diverges. Use it when you push the cache interval for more speed.
- **standalone / model-agnostic:** [`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus) — the forecaster itself, to add DMD caching to *your own* diffusion/flow model.
- **`fast-trellis2`** = the TaylorSeer baseline fork (the upstream "Fast" accel) — the v2 reference point, not a HiCache variant.

> **This repo:** `faster-trellis` — **TRELLIS v1 × HiCache (Hermite)** — carved-hybrid (HiCache SS + token-carved SLaT).

`faster-trellis` is `microsoft/TRELLIS` image-to-3D with a **training-free acceleration stack** built
into the flow-matching sampler. It forecasts and reuses the model's **final CFG-combined velocity**
so the sampler spends far fewer, cheaper network evaluations per asset — the weights, decoders, and
mesh + Gaussian + radiance-field outputs are the base model's, untouched.

```python
pipeline.enable_faster_mode()        # the accelerated config — one line, no tuning
pipeline.enable_faster_mode("none")  # stock TRELLIS sampler (kill-switch)
```

**What it does.** Two accelerations act on the velocity the samplers emit, one per stage class:

- **HiCache** on the **sparse-structure** stage — a dual-scaled physicist's-Hermite forecast of the
  velocity that skips most of the network calls fixing the coarse occupancy volume.
- **Token-carved SLaT** on the **structured-latent** stage — a learned-cadence temporal skip plus
  spatial **token carving** that recomputes only the high-frequency voxels each step.

**Built on TRELLIS and Fast-TRELLIS.** Fast-TRELLIS accelerates the SLaT stage with TaylorSeer
forecasting + token carving; faster-trellis takes that carving substrate and pairs it with a
**Hermite** forecast on the sparse-structure stage — where the monomial TaylorSeer basis diverges —
so both stages accelerate. The result is a **single shipped configuration** (no mode menu) that
**beats Fast-TRELLIS on both speed and quality**: it runs ~6 sparse-latent network evaluations
versus Fast-TRELLIS's ~14, so it is **1.35× faster at equal-or-better F-score** (see
[Results](#results)). `GF_CARVE_RATIO` / `GF_HICACHE_SS_INTERVAL` / `GF_HICACHE_FIRST_ENHANCE`
override the defaults for tuning.

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
pipeline.enable_faster_mode()                         # ← the only added line

outputs = pipeline.run(image, formats=["mesh", "gaussian", "radiance_field"])
```

Runnable scripts: `example.py` (`--mode {faster,none}`) and
`example_faster.py` (minimal, annotated).

> **Mesh output requires the FlexiCubes submodule.** `formats=["mesh"]` (the
> decoder used for F-score / CD evaluation) imports
> `trellis/representations/mesh/flexicubes`. If you cloned without
> `--recurse-submodules`, fetch it once with `git submodule update --init`,
> otherwise mesh decoding will raise a `flexicubes` import error. The
> `gaussian` and `radiance_field` formats do not need it.

<details>
<summary><b>RTX 50-series (sm_120) note</b></summary>

The default flash-attn / sparse-conv kernels may not ship sm_120 builds. Select the
spconv backend and its native algorithm:

```bash
export SPARSE_CONV_BACKEND=spconv
export SPCONV_ALGO=native
```
</details>

---

## Results

Toys4K, **RTX 5090**, 25 steps/stage. Geometry is scored on the `formats=["mesh"]` decoder
output with area-weighted surface sampling, after a globally-optimal (Go-ICP) similarity
alignment to the ground-truth mesh — the same harness across every row, over the **32 objects
all variants align** (a few rotationally-symmetric objects Go-ICP cannot orient uniquely are
dropped equally from every row). Latency is end-to-end generation, one object at a time, weights
resident; "SLaT evals" is the median number of network evaluations on the structured-latent stage.

| | F1@0.05 mean ↑ | CD ↓ | latency ↓ | speedup | SLaT evals ↓ |
|---|:--:|:--:|:--:|:--:|:--:|
| TRELLIS (base) | 0.841 | 0.0556 | 3.49 s | 1.00× | ~25 |
| Fast-TRELLIS | 0.823 | 0.0594 | 1.65 s | 2.12× | 14 |
| **faster-trellis** | **0.825** | **0.0581** | **1.22 s** | **2.85×** | **6** |

<sub>faster-trellis runs **2.85× faster than base TRELLIS and 1.35× faster than Fast-TRELLIS**
(1.22 s vs 1.65 s) at **equal-or-better quality** — higher mean F-score (0.825 vs 0.823) and lower
Chamfer distance (0.0581 vs 0.0594) — using **6 structured-latent network evaluations versus
Fast-TRELLIS's 14**. The per-object F1 *median* sits within ~0.012 across the three rows; the mean
is the headline figure because it weights the hard objects acceleration is actually judged on.
Every row uses the identical Toys4K mesh F-score harness (same Go-ICP alignment, same seed).</sub>

---

## How it works

TRELLIS v1 samples a 3D asset in two flow-matching stages — **sparse structure** (the coarse
occupancy that fixes topology) then **structured latent** (the refined geometry). The velocity
field is smooth across diffusion steps and most tokens change slowly, so most model passes are
redundant. faster-trellis exploits both: a **HiCache** Hermite forecast thins the sparse-structure
stage, and a **token-carved** SLaT sampler skips whole steps and recomputes only the high-frequency
voxels on the steps it does run.

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

**Why Hermite over Taylor.** A Taylor (monomial) forecast uses `(−k)ⁱ`, whose terms grow without
bound as the horizon `k` increases — high-order corrections amplify finite-difference noise and
the forecast diverges. The Hermite polynomials are the eigenfunctions of the diffusion operator
governing the velocity field; the dual scaling (input `σx` **and** coefficient `σⁿ`, `σ∈(0,1)`)
contracts high-order terms into the numerically stable oscillatory regime. The forecast stays
bounded and accurate at longer skip horizons, which is what lets the compute interval widen.
Taylor is the monomial special case. *(arXiv:2508.16984)*
</details>

<details>
<summary><b>② Token-carved SLaT — recompute only the high-frequency voxels</b> (SLaT stage)</summary>

The structured-latent stage denoises a `SparseTensor` of voxel tokens, and most tokens change
slowly between steps. On each computed step we score every token by **spatial high-frequency
energy** (a 3D-FFT of the sparse-structure occupancy) together with its velocity magnitude and
motion, and recompute only the most active fraction — the **carving ratio**; the smoothest tokens
reuse their cached velocity under a staleness bound that forces a periodic full refresh. On top of
that a **learned-k delta cache** skips whole steps when the velocity field is locally linear
(`vₜ ≈ xₜ + Δ`). The pipeline hands the SS occupancy's frequency score to the SLaT sampler, so the
carving signal is the structure itself. *(Fast-TRELLIS token selection, paired with our Hermite SS
forecast; carving ratio = `GF_CARVE_RATIO`.)*

**The savings multiply:** the SLaT sampler skips whole steps *and* carves tokens on the steps it
runs, while HiCache independently thins the sparse-structure stage.
</details>

---

## Tuning

The shipped configuration is a single set of defaults; each is overridable by env var (e.g. to
trade a little speed for fidelity on a hard input):

| knob | env | default | meaning |
|---|---|:--:|---|
| carving ratio | `GF_CARVE_RATIO` | `0.25` | fraction of SLaT tokens *skipped* per step (cached, not recomputed) |
| SS interval | `GF_HICACHE_SS_INTERVAL` | `3` | sparse-structure: compute 1 step, forecast `interval − 1` |
| SS first-enhance | `GF_HICACHE_FIRST_ENHANCE` | `2` | always compute the first N sparse-structure steps |

Finer knobs live in `trellis/pipelines/samplers/faster_samplers.py`: `GF_CARVE_THRESH` (`5.0`,
SLaT delta-cache skip threshold) and the SS Hermite `sigma` (`0.5`, contraction σ∈(0,1)) /
`max_order` (`1`).

---

## What's added on top of TRELLIS

The Microsoft TRELLIS model and decoder math are unchanged. The acceleration lives under
`trellis/pipelines/samplers/`: `hicache.py` + `hicache_freq.py` (the Hermite cache and the 3D-FFT
token-frequency scoring that drives carving), `flow_euler_carved.py` (the token-carved SLaT
sampler), `adaptive_cfg.py`, and `faster_samplers.py` (the accelerated sampler classes), plus the
`enable_faster_mode()` (quality/speed) API in `trellis/pipelines/trellis_image_to_3d.py`. The
accelerators are independent re-implementations of the cited papers.

The one decoder-file touch is import-only: `representations/mesh/cube2mesh.py` imports the
optional FlexiCubes submodule **lazily** (at mesh-decoder construction time) rather than at
module top level, so the package — and the gaussian / radiance-field paths — stay importable
when the submodule has not been fetched. Mesh-decoder geometry is identical when FlexiCubes is
present.

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

"""Faster-TRELLIS image-to-3D example.

Identical to the stock TRELLIS ``example.py`` pipeline (gaussian / radiance
field / mesh / GLB / PLY outputs all preserved), with a SINGLE extra call --
``pipeline.enable_faster_mode(...)`` -- that swaps in the training-free
accelerated samplers (HiCache Hermite velocity forecast + Adaptive Guidance).

Modes
-----
* ``--mode faster``  : full stack (HiCache + Adaptive Guidance). Default. Fastest.
* ``--mode hicache`` : HiCache only (maximum-quality safety toggle).
* ``--mode none``    : vanilla TRELLIS (no acceleration).

Usage
-----
    SPCONV_ALGO=native ATTN_BACKEND=sdpa CUDA_VISIBLE_DEVICES=0 \
        python example_faster.py --image_path assets/example_image/T.png \
        --mode faster --weights microsoft/TRELLIS-image-large

Blackwell (sm_120) note: if spconv's native algo errors on your GPU, export
``SPARSE_CONV_BACKEND=spconv`` (and ``ATTN_BACKEND=sdpa``) before running.
"""
import os
import time
import argparse
from pathlib import Path

import numpy as np
import imageio
import trimesh
from PIL import Image

# Environment defaults (override externally as needed).
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def export_untextured_mesh(mesh, output_path):
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Faster-TRELLIS image-to-3D")
    parser.add_argument("--image_path", default="assets/example_image/T.png")
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large",
                        help="HF repo id or local path to TRELLIS-image-large")
    parser.add_argument("--mode", default="faster",
                        choices=["faster", "hicache", "none"],
                        help="acceleration mode (default: faster = full stack)")
    parser.add_argument("--output_dir", default="outputs_faster")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ss_steps", type=int, default=25)
    parser.add_argument("--ss_cfg", type=float, default=7.5)
    parser.add_argument("--slat_steps", type=int, default=25)
    parser.add_argument("--slat_cfg", type=float, default=3.0)
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--skip_glb", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load the full TRELLIS v1 pipeline ----
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.cuda()

    # ---- the ONE extra call: enable training-free acceleration ----
    pipeline.enable_faster_mode(args.mode)
    print(f"Faster-TRELLIS mode: {pipeline.faster_mode}")

    image = Image.open(args.image_path)

    t0 = time.time()
    outputs = pipeline.run(
        image,
        seed=args.seed,
        formats=["mesh", "gaussian", "radiance_field"],
        sparse_structure_sampler_params={"steps": args.ss_steps, "cfg_strength": args.ss_cfg},
        slat_sampler_params={"steps": args.slat_steps, "cfg_strength": args.slat_cfg},
    )
    print(f"[inference] {time.time() - t0:.3f}s  (mode={args.mode})")

    # ---- full set of TRELLIS outputs (unchanged) ----
    if not args.skip_video:
        imageio.mimsave(str(out / "sample_gs.mp4"),
                        render_utils.render_video(outputs["gaussian"][0])["color"], fps=30)
        imageio.mimsave(str(out / "sample_rf.mp4"),
                        render_utils.render_video(outputs["radiance_field"][0])["color"], fps=30)
        imageio.mimsave(str(out / "sample_mesh.mp4"),
                        render_utils.render_video(outputs["mesh"][0])["normal"], fps=30)

    if not args.skip_glb:
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95, texture_size=1024)
        glb.export(str(out / "sample.glb"))

    export_untextured_mesh(outputs["mesh"][0], out / "sample_mesh.obj")
    outputs["gaussian"][0].save_ply(str(out / "sample.ply"))
    print(f"saved outputs -> {out}")


if __name__ == "__main__":
    main()

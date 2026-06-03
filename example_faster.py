"""Faster-TRELLIS image-to-3D example.

Runs the standard TRELLIS image-to-3D pipeline (gaussian / radiance field /
mesh / GLB / PLY outputs are all produced), adding a single call --
``pipeline.enable_faster_mode()`` -- that swaps in the training-free
accelerated samplers (a HiCache Hermite sparse-structure forecast over the
token-carved SLaT sampler).

Modes
-----
* ``--mode faster`` : the accelerated config (default).
* ``--mode none``   : standard TRELLIS samplers, no acceleration.

Usage
-----
    SPCONV_ALGO=native ATTN_BACKEND=sdpa CUDA_VISIBLE_DEVICES=0 \
        python example_faster.py --image_path assets/example_image/T.png \
        --mode faster --weights microsoft/TRELLIS-image-large

``SPCONV_ALGO`` selects the spconv sparse-convolution implementation; ``native``
is a portable choice. If you prefer the spconv backend's own kernels, set
``SPARSE_CONV_BACKEND=spconv`` (with ``ATTN_BACKEND=sdpa``) before running.
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
                        choices=["faster", "none"],
                        help="faster = the accelerated config (default); none = stock sampler")
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

    # ---- enable training-free acceleration ----
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

    # ---- write the full set of TRELLIS outputs ----
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

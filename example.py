"""TRELLIS image-to-3D example (Faster-TRELLIS).

Full TRELLIS v1 pipeline with optional training-free acceleration selected via
``--mode {faster,none}``. See ``example_faster.py`` for the minimal annotated
version and ``README.md`` for the method.
"""
import os
import time
import argparse
from pathlib import Path

import numpy as np
import imageio
import trimesh
from PIL import Image

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def export_untextured_mesh(mesh, output_path):
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(str(output_path))


parser = argparse.ArgumentParser(description="TRELLIS / Faster-TRELLIS image-to-3D")
parser.add_argument("--image_path", default="assets/example_image/T.png")
parser.add_argument("--weights", default="microsoft/TRELLIS-image-large",
                    help="HF repo id or local path to TRELLIS-image-large")
parser.add_argument("--mode", default="faster", choices=["faster", "none"],
                    help="faster = the accelerated config (default); none = stock TRELLIS sampler")
parser.add_argument("--output_dir", default=".")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--ss_steps", type=int, default=25)
parser.add_argument("--ss_cfg", type=float, default=7.5)
parser.add_argument("--slat_steps", type=int, default=25)
parser.add_argument("--slat_cfg", type=float, default=3.0)
parser.add_argument("--skip_video", action="store_true")
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
pipeline.cuda()
pipeline.enable_faster_mode(args.mode)
print(f"Faster-TRELLIS mode: {pipeline.faster_mode}")

image = Image.open(args.image_path)

infer_start = time.time()
outputs = pipeline.run(
    image,
    seed=args.seed,
    formats=["mesh", "gaussian", "radiance_field"],
    sparse_structure_sampler_params={"steps": args.ss_steps, "cfg_strength": args.ss_cfg},
    slat_sampler_params={"steps": args.slat_steps, "cfg_strength": args.slat_cfg},
)
print(f"[Inference] {time.time() - infer_start:.3f}s")

if not args.skip_video:
    print("Processing preview videos")
    imageio.mimsave(str(output_dir / "sample_gs.mp4"),
                    render_utils.render_video(outputs["gaussian"][0])["color"], fps=30)
    imageio.mimsave(str(output_dir / "sample_rf.mp4"),
                    render_utils.render_video(outputs["radiance_field"][0])["color"], fps=30)
    imageio.mimsave(str(output_dir / "sample_mesh.mp4"),
                    render_utils.render_video(outputs["mesh"][0])["normal"], fps=30)

print("GLB post-processing")
glb = postprocessing_utils.to_glb(
    outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95, texture_size=1024)
glb.export(str(output_dir / "sample.glb"))

export_untextured_mesh(outputs["mesh"][0], output_dir / "sample_mesh.obj")
outputs["gaussian"][0].save_ply(str(output_dir / "sample.ply"))
print(f"Saved outputs -> {output_dir}")

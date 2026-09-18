"""CPU contract tests for the TRELLIS v1 acceleration seam.

These tests deliberately load the sampler modules without the optional GPU
runtime dependencies.  They exercise cache lifecycle and backend selection,
not model quality or end-to-end inference.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from hicache_pp import CacheBudget


ROOT = Path(__file__).resolve().parents[1]


class _EasyDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _load_sampler_modules():
    """Load the pure sampler modules while bypassing heavy package inits."""
    names = {
        "trellis": ROOT / "trellis",
        "trellis.pipelines": ROOT / "trellis" / "pipelines",
        "trellis.pipelines.samplers": ROOT / "trellis" / "pipelines" / "samplers",
    }
    for name, path in names.items():
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules.setdefault(name, pkg)
    easy = types.ModuleType("easydict")
    easy.EasyDict = _EasyDict
    sys.modules.setdefault("easydict", easy)
    modules_pkg = types.ModuleType("trellis.modules")
    modules_pkg.__path__ = []
    sys.modules.setdefault("trellis.modules", modules_pkg)
    sparse_mod = types.ModuleType("trellis.modules.sparse")
    sparse_mod.SparseTensor = object
    sys.modules.setdefault("trellis.modules.sparse", sparse_mod)

    def load(name):
        path = ROOT.joinpath(*name.split("."))
        path = path.with_suffix(".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    load("trellis.pipelines.samplers.base")
    load("trellis.pipelines.samplers.classifier_free_guidance_mixin")
    load("trellis.pipelines.samplers.guidance_interval_mixin")
    hicache = load("trellis.pipelines.samplers.hicache")
    load("trellis.pipelines.samplers.adaptive_cfg")
    faster = load("trellis.pipelines.samplers.faster_samplers")
    return hicache, faster


def test_backend_dispatch_and_validation():
    hicache, _ = _load_sampler_modules()
    assert hicache.normalize_backend(None) == "hermite"
    assert hicache.normalize_backend(" Hermite ") == "hermite"
    for bad in ("dmd", "carved"):
        try:
            hicache.normalize_backend(bad)
        except ValueError as exc:
            assert "hermite" in str(exc)
        else:
            raise AssertionError(f"unsupported backend was accepted: {bad}")

    state = hicache.hicache_init(8, interval=2, first_enhance=0,
                                 end_enhance=8, backend="hermite", stage="sparse_structure")
    assert state["backend"] == "hermite"
    assert state["stage"] == "sparse_structure"


def test_hicache_lifecycle_reset_and_actual_status():
    _, faster = _load_sampler_modules()
    sampler = faster.FlowEulerGuidanceIntervalSampler_hicache(
        sigma_min=0.001, is_sparse=False, steps=8)
    assert sampler._hicache_backend == "hermite"
    assert sampler._hicache_stage == "sparse_structure"
    sampler._hicache_begin_run(8)
    first_state = sampler._hicache
    assert sampler.hicache_status()["backend"] == "hermite"
    assert sampler.hicache_status()["stage"] == "sparse_structure"

    calls = {"n": 0}

    def model(x, t, cond, **kwargs):
        calls["n"] += 1
        return x + 0.1

    x = torch.zeros(2, 3)
    for step in range(4):
        # Guidance interval parked outside t: single-pass path, one model
        # call per full step (the stock CFG mixins are not under test here).
        sampler.sample_once(model, x, 1.0 - step * 0.1, 0.9 - step * 0.1,
                            cond=None, neg_cond=None, cfg_strength=0.0,
                            cfg_interval=(2.0, 3.0))
    assert calls["n"] < 4, "a forecast step must avoid a model call"
    assert sampler.hicache_status()["forecast_steps"] > 0

    sampler._hicache_end_run()
    assert sampler.hicache_status()["enabled"] is False
    sampler._hicache_begin_run(8)
    assert sampler._hicache is not first_state
    assert sampler.hicache_status()["full_steps"] == 0


def test_budget_manifest_records_guarded_fallback_and_is_portable(tmp_path):
    _, faster = _load_sampler_modules()
    sampler = faster.FlowEulerGuidanceIntervalSampler_hicache(
        sigma_min=0.001, is_sparse=False, steps=8,
        budget=CacheBudget(
            backend="hermite",
            allowed_stages=("sparse_structure",),
            max_horizon=1,
            quality_preset="test",
            fallback="full",
        ))
    sampler._hicache_begin_run(8)

    def model(x, t, cond, **kwargs):
        return x + 0.1

    x = torch.zeros(2, 3)
    for step in range(4):
        sampler.sample_once(model, x, 1.0 - step * 0.1, 0.9 - step * 0.1,
                            cond=None, neg_cond=None, cfg_strength=0.0,
                            cfg_interval=(2.0, 3.0))

    manifest = sampler.get_hicache_manifest()
    assert manifest["schema"] == "hicache-pp.run-manifest.v1"
    assert manifest["identity"]["stage"] == "sparse_structure"
    assert manifest["counts"]["fallback"] >= 1
    assert len(manifest["measurements"]) == 4
    output = tmp_path / "manifest.json"
    sampler.save_hicache_manifest(output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == manifest
    assert all("tensor" not in key.lower() for key in json.dumps(saved).split('"'))


def test_pipeline_preset_reports_faster_and_keeps_carved_slat_separate():
    """The user-facing preset accelerates SS and reports all stages."""
    # Importing the pipeline normally requires the full TRELLIS runtime.  A
    # small module shim is enough to exercise its configuration contract.
    import importlib.util

    class Sampler:
        def __init__(self, sigma_min=1e-5, **kwargs):
            self.sigma_min = sigma_min
            for key, value in kwargs.items():
                setattr(self, key, value)

    class HiCache(Sampler):
        pass

    class Carved(Sampler):
        carving_ratio = 0.25

    sampler_mod = types.ModuleType("trellis.pipelines.samplers")
    sampler_mod.Sampler = Sampler
    sampler_mod.FlowEulerGuidanceIntervalSampler = Sampler
    sampler_mod.FlowEulerGuidanceIntervalSampler_hicache = HiCache
    sampler_mod.FlowEulerGuidanceIntervalSampler_carved = Carved
    sys.modules["trellis.pipelines.samplers"] = sampler_mod

    base_mod = types.ModuleType("trellis.pipelines.base")

    class Pipeline:
        pass

    base_mod.Pipeline = Pipeline
    sys.modules["trellis.pipelines.base"] = base_mod
    tv_mod = types.ModuleType("torchvision")
    tv_mod.transforms = types.SimpleNamespace()
    sys.modules["torchvision"] = tv_mod
    pil_mod = types.ModuleType("PIL")
    pil_mod.Image = types.SimpleNamespace(Image=object)
    sys.modules["PIL"] = pil_mod
    sys.modules["rembg"] = types.ModuleType("rembg")
    modules_pkg = types.ModuleType("trellis.modules")
    modules_pkg.__path__ = []
    sys.modules["trellis.modules"] = modules_pkg
    sparse_mod = types.ModuleType("trellis.modules.sparse")
    sparse_mod.SparseTensor = object
    sys.modules["trellis.modules.sparse"] = sparse_mod

    name = "trellis.pipelines.trellis_image_to_3d_contract"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "trellis" / "pipelines" / "trellis_image_to_3d.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    pipe = module.TrellisImageTo3DPipeline.__new__(
        module.TrellisImageTo3DPipeline)
    pipe.sparse_structure_sampler = Sampler()
    pipe.slat_sampler = Sampler()
    pipe.sparse_structure_sampler_params = {"steps": 25}
    try:
        pipe.enable_faster_mode("dmd")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mode was accepted")
    pipe.enable_faster_mode("faster")
    status = pipe.acceleration_status()
    assert status["mode"] == "faster"
    assert status["backend"] == "hermite"
    assert status["stages"]["sparse_structure"] == "hicache:hermite"
    assert status["stages"]["slat"] == "carved_slat"
    assert pipe.slat_sampler.carving_ratio == 0.25
    pipe.enable_faster_mode("none")
    assert pipe.acceleration_status()["enabled"] is False


if __name__ == "__main__":
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            # pytest supplies tmp_path; direct runs get a TemporaryDirectory.
            wants_tmp = "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]
            if wants_tmp:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            print(f"[PASS] {name}")

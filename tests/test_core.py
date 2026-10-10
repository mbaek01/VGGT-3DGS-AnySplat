import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
from benchmark.datasets.mipnerf360 import load_split, prepare, resize_crop
from benchmark.evaluation.pose_alignment import estimate_similarity, transform_c2w, align_targets
from benchmark.evaluation.metrics import psnr_ssim
from benchmark.evaluation.evaluator import evaluate
from benchmark.utils.cameras import read_colmap
from benchmark.utils.io import ROOT, read_json, write_json


@pytest.mark.parametrize("scene,first,last,test", [
    ("kitchen", "DSCF0656", "DSCF0691", ["DSCF0664", "DSCF0673", "DSCF0682", "DSCF0691"]),
    ("room", "DSCF4667", "DSCF4702", ["DSCF4675", "DSCF4684", "DSCF4693", "DSCF4702"]),
])
def test_assignment(scene, first, last, test):
    split = load_split(ROOT / "splits" / f"{scene}.json")
    assert len(split["train"]) == 32
    assert split["test"] == test
    all_names = sorted(split["train"] + split["test"])
    assert all_names[0] == first and all_names[-1] == last
    assert not set(split["train"]) & set(test)


def test_duplicate_test_rejected(tmp_path):
    data = read_json(ROOT / "splits/kitchen.json")
    data["test"][0] = data["test"][1]
    write_json(tmp_path / "split.json", data)
    with pytest.raises(ValueError):
        load_split(tmp_path / "split.json")


def test_similarity_direction_and_rotation():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(32, 3))
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    target = 2.5 * (source @ R.T) + [3, 4, 5]
    s, rot, t, residual = estimate_similarity(source, target)
    assert s == pytest.approx(2.5)
    np.testing.assert_allclose(rot, R, atol=1e-12)
    np.testing.assert_allclose(t, [3, 4, 5], atol=1e-12)
    assert max(residual) < 1e-12
    pose = np.eye(4)
    pose[:3, 3] = source[0]
    transformed = transform_c2w(pose, s, rot, t)
    np.testing.assert_allclose(transformed[:3, 3], target[0])
    np.testing.assert_allclose(transformed[:3, :3], R, atol=1e-12)
    assert np.linalg.det(transformed[:3, :3]) == pytest.approx(1.)


def test_degenerate_alignment_fails():
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match="Collinear"):
        estimate_similarity(points, points)


def test_crop_projection_consistency():
    image = Image.new("RGB", (100, 60))
    K = np.array([[80., 0, 50], [0, 80, 30], [0, 0, 1]])
    resized, new_K, A = resize_crop(image, K, 48, 48)
    assert resized.size == (48, 48)
    X = np.array([.2, -.1, 2.])
    np.testing.assert_allclose(new_K @ X, A @ (K @ X))
    assert new_K[0, 2] == pytest.approx(24)
    assert new_K[1, 2] == pytest.approx(24)


@pytest.fixture
def prepared(tmp_path):
    split = load_split(ROOT / "splits/kitchen.json")
    source = tmp_path / "source"
    images = source / "images"
    sparse = source / "sparse/0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    with (sparse / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<QiiQQdddd", 1, 1, 1, 80, 60, 70., 70., 40., 30.))
    names = sorted(split["train"] + split["test"])
    with (sparse / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for index, name in enumerate(names):
            angle = index*.2
            tx, ty, tz = np.cos(angle), np.sin(angle), index*.01
            f.write(struct.pack("<i7di", index+1, 1., 0., 0., 0., tx, ty, tz, 1))
            f.write((name+".JPG").encode()+b"\0")
            f.write(struct.pack("<Q", 0))
            pixels = np.random.default_rng(index).integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(images / f"{name}.JPG")
    config = dict(width=48, height=48, policy="resize_short_edge_center_crop", interpolation="bicubic")
    out = tmp_path / "prepared"
    prepare(source, dict(image_dir="images", colmap_dir="sparse/0"), ROOT / "splits/kitchen.json", out, config)
    return out


def test_colmap_preparation_and_leakage_boundary(prepared):
    meta = read_json(prepared / "cameras.json")
    assert {p.stem for p in (prepared / "train/images").iterdir()} == set(meta["split"]["train"])
    assert {p.stem for p in (prepared / "gt").iterdir()} == set(meta["split"]["test"])
    assert len(meta["cameras"]) == 36
    assert "gt" not in meta and "source" not in meta
    assert np.asarray(meta["cameras"]["DSCF0656"]["c2w"])[0, 3] == pytest.approx(-1)


def test_test_cameras_never_used_for_fit(prepared):
    meta = read_json(prepared / "cameras.json")
    predicted = {n: np.array(meta["cameras"][n]["c2w"]) for n in meta["split"]["train"]}
    targets, before = align_targets(meta, predicted)
    name = meta["split"]["test"][0]
    meta["cameras"][name]["c2w"][0][3] += 1000
    changed, after = align_targets(meta, predicted)
    assert before == after
    assert changed[name][0, 3]-targets[name][0, 3] == pytest.approx(1000)
    predicted[name] = np.eye(4)
    with pytest.raises(ValueError, match="exactly"):
        align_targets(meta, predicted)


def test_evaluator_exact_views_and_resolution(prepared, tmp_path):
    meta = read_json(prepared / "cameras.json")
    out = tmp_path / "run"
    (out / "renders").mkdir(parents=True)
    write_json(out / "job.json", {"method": "synthetic_test", "metadata": meta, "alignment": {"mode": "none"}})
    write_json(out / "timing.json", {"reconstruction_compute_seconds": 0, "render_seconds": 0})
    for name in meta["split"]["test"]:
        (out / "renders" / f"{name}.png").write_bytes((prepared / "gt" / f"{name}.png").read_bytes())
    cfg = {"lpips_net": "alex", "device": "cpu"}
    result = evaluate(out, prepared, cfg, skip_lpips=True)
    assert result["mean"] == {"psnr": 120., "ssim": 1., "lpips": None}
    assert result["complete_metrics"] is False
    path = out / "renders" / f"{meta['split']['test'][0]}.png"
    Image.new("RGB", (32, 32)).save(path)
    with pytest.raises(ValueError, match="resolution mismatch"):
        evaluate(out, prepared, cfg, skip_lpips=True)
    path.unlink()
    with pytest.raises(ValueError, match="exactly"):
        evaluate(out, prepared, cfg, skip_lpips=True)


def test_psnr_known_error():
    a = np.zeros((32, 32, 3), dtype=np.float32)
    b = np.full_like(a, .1)
    assert psnr_ssim(a, b)["psnr"] == pytest.approx(20., abs=1e-5)


@pytest.mark.parametrize("module", ["benchmark.run", "benchmark.reproduce", "benchmark.workers.native", "benchmark.workers.vggt_export"])
def test_help_without_site_packages(module):
    result = subprocess.run([sys.executable, "-S", "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_dry_run_no_data(tmp_path):
    result = subprocess.run([sys.executable, "-m", "benchmark.run", "run", "--scene", "kitchen", "--method", "vggt_3dgs", "--dry-run", "--data-root", str(tmp_path / "absent")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["split"]["train"]) == 32


def test_removed_method_adapters_fail_explicitly():
    from benchmark.methods import get_method
    for name in ("mpr_gs", "ours"):
        with pytest.raises(ValueError, match="Unknown method"):
            get_method(name)

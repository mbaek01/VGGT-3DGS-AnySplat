from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from benchmark.reproduce import validate_refinement, refine_cameras
from benchmark.intrinsic_refinement.optimizer import scaled_intrinsics
from benchmark.pose_refinement.se3 import apply_correction
from benchmark.utils.io import ROOT, read_json, read_yaml, write_json, sha256
from benchmark.utils.native import geometry_fingerprint, compare_native_config
from test_intrinsic_refinement import calibrated_scene


def save_refinement(path, names, poses, K, scale):
    delta = np.zeros((len(names), 6)); delta[2] = [.01, -.01, 0, .01, 0, 0]
    a = np.log(1.03)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, names=names, initial_c2w=poses, initial_K=K,
        refined_c2w=apply_correction(poses, delta, 'camera'), refined_K=scaled_intrinsics(K, a),
        deltas=delta, log_focal_scale=a, scene_scale=scale)


def test_saved_refinement_detects_changed_principal_point(tmp_path):
    poses, K, _ = calibrated_scene(); names = ['a', 'b', 'c', 'd']
    path = tmp_path / 'result.npz'; save_refinement(path, names, poses, K, 1.5)
    validate_refinement(path, names, poses, K, 1.5)
    with np.load(path, allow_pickle=False) as saved:
        values = {k: saved[k] for k in saved.files}
    values['refined_K'][1, 0, 2] += .5
    np.savez_compressed(path, **values)
    with pytest.raises(AssertionError):
        validate_refinement(path, names, poses, K, 1.5)


def test_reuse_refinement_never_runs_matching_or_optimizer(tmp_path, monkeypatch):
    from benchmark import reproduce
    from benchmark.intrinsic_refinement import optimizer
    poses, K, _ = calibrated_scene(); names = ['a', 'b', 'c', 'd']
    path = tmp_path / 'saved/result.npz'; save_refinement(path, names, poses, K, 1.5)
    intrinsic = read_yaml(ROOT / 'configs/methods/rgb_pose_focal_refine.yaml')
    pose = read_yaml(ROOT / intrinsic['pose_config'])
    method = dict(rgb_matching=pose['rgb_matching'], pose_refinement=pose['pose_refinement'],
                  intrinsic_refinement=intrinsic['intrinsic_refinement'])
    write_json(path.parent / 'settings.json', dict(pose=method['pose_refinement'],
        intrinsic=method['intrinsic_refinement'], mode='pose_focal'))
    monkeypatch.setattr(reproduce, 'native_inputs', lambda source: (names, poses, K, 1.5))
    monkeypatch.setattr(optimizer, 'refine', lambda *a, **k: pytest.fail('No optimization on saved result reuse'))
    assert refine_cameras(tmp_path / 'output', tmp_path / 'source', {}, {}, method, saved_refinement=path) == path
    assert not (tmp_path / 'output').exists()


def test_frozen_camera_install_preserves_points_splats_optimizer_and_rng(tmp_path, monkeypatch):
    import torch
    from benchmark.workers import pose_focal as worker
    poses, K, _ = calibrated_scene()
    poses, K = np.tile(poses, (8, 1, 1)), np.tile(K, (8, 1, 1))
    names = [f'camera{i:02d}' for i in range(32)]
    path = tmp_path / 'refinement/result.npz'; save_refinement(path, names, poses, K, 1.5)

    @dataclass
    class Config:
        result_dir: str = 'new'
        max_steps: int = 30000

    parameter = torch.nn.Parameter(torch.tensor([[1., 2., 3.]]))
    parser = SimpleNamespace(image_names=[n + '.png' for n in names], camtoworlds=poses.copy(),
        Ks_dict={i: k.copy() for i, k in enumerate(K)}, camera_ids=list(range(32)), scene_scale=1.5,
        points=np.array([[0., 0., 3.]]), points_rgb=np.array([[20, 30, 40]], np.uint8), transform=np.eye(4))
    runner = SimpleNamespace(parser=parser, cfg=Config(), scene_scale=1.65,
        splats=torch.nn.ParameterDict({'means': parameter}), optimizers=dict(means=torch.optim.Adam([parameter])))
    source, folder = tmp_path / 'source', tmp_path / 'output'; folder.mkdir()
    write_json(source / 'predicted_train_cameras.json', {n: p.tolist() for n, p in zip(names, poses)})
    write_json(source / 'native_environment.json', {n: 'test' for n in ('torch', 'torchvision', 'numpy', 'gsplat')})
    (source / 'train/sparse').mkdir(parents=True); (source / 'train/sparse/points3D.bin').write_bytes(b'unchanged')
    monkeypatch.setattr(worker.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(worker, 'read_native_config', lambda p: dict(result_dir='old', max_steps=30000))
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    before = geometry_fingerprint(parser, runner)
    job = dict(source_run=str(source), run_dir=str(folder), method_config=dict(gsplat=dict(iterations=30000)),
        validation=dict(variant='pose_focal', refinement=str(path), refinement_sha256=sha256(path)),
        metadata=dict(split=dict(train=names, test=['heldout0', 'heldout1', 'heldout2', 'heldout3'])))
    worker.install_cameras(runner, job)
    after = geometry_fingerprint(parser, runner)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(parser.camtoworlds, saved['refined_c2w'])
        np.testing.assert_array_equal(np.stack(list(parser.Ks_dict.values())), saved['refined_K'])
    assert {k: v for k, v in before.items() if k != 'intrinsics'} == {k: v for k, v in after.items() if k != 'intrinsics'}
    assert read_json(folder / 'invariants.json')['rng_unchanged']


def test_config_comparison_rejects_training_setting_changes():
    previous = dict(max_steps=30000, result_dir='old', ssim_lambda=.2)
    assert compare_native_config(dict(previous, result_dir='new'), previous)['equal']
    with pytest.raises(ValueError):
        compare_native_config(dict(previous, ssim_lambda=.1), previous)

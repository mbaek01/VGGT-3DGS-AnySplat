import copy
from pathlib import Path
import numpy as np
import pytest
from benchmark.rgb_pose_refinement.geometry import (normalized_pixels, essential_matrix,
    fundamental_matrix, sampson_squared, homogeneous, fundamental_torch, sampson_residual_torch)
from benchmark.rgb_pose_refinement.matching import train_paths, neighbor_pairs, verify_pair, cached_matches, load_matches
from benchmark.rgb_pose_refinement.optimizer import refine
from benchmark.pose_refinement.se3 import apply_correction, se3_exp
from benchmark.pose_refinement.pose_metrics import pose_metrics
from benchmark.utils.io import ROOT, read_yaml


def synthetic():
    rng = np.random.default_rng(7)
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, :3, 3] = [[0, 0, 0], [.7, .1, 0], [1.2, -.3, .1], [.2, .9, .1]]
    poses[2, :3, :3] = se3_exp([.02, -.03, .01, 0, 0, 0])[:3, :3]
    K = np.array([[430., 0, 224], [0, 410., 224], [0, 0, 1.]])
    Ks = np.repeat(K[None], 4, axis=0)
    points = rng.uniform([-1., -.9, 3.], [1.5, 1.2, 7.], (200, 3))
    pixels = []
    for p in poses:
        q = (points-p[:3, 3]) @ p[:3, :3]; uv = q @ K.T
        pixels.append(uv[:, :2]/uv[:, 2:])
    matches = [dict(indices=[i, j], names=[str(i), str(j)], xy_i=pixels[i], xy_j=pixels[j],
        confidence=np.ones(len(points)), weight=1.) for i in range(4) for j in range(i+1, 4)]
    return poses, Ks, matches, points


def settings():
    return read_yaml(ROOT/'configs/methods/rgb_pose_refine.yaml')['pose_refinement']


def test_normalized_pixels_full_intrinsic_inverse():
    K = np.array([[400., 12., 210.], [0, 500., 220.], [0, 0, 1.]])
    rays = np.array([[.2, -.3, 1.], [-.4, .1, 1.]])
    xy = (rays @ K.T)[:, :2]
    np.testing.assert_allclose(normalized_pixels(xy, K), rays, atol=1e-14)


def test_c2w_relative_essential_fundamental_and_zero_sampson():
    poses, Ks, matches, points = synthetic()
    for pair in matches:
        i, j = pair['indices']; relative = np.linalg.inv(poses[j]) @ poses[i]
        X_i = (points-poses[i, :3, 3]) @ poses[i, :3, :3]
        X_j = (points-poses[j, :3, 3]) @ poses[j, :3, :3]
        np.testing.assert_allclose(X_i@relative[:3, :3].T+relative[:3, 3], X_j, atol=1e-14)
        E = essential_matrix(poses[i], poses[j]); F = fundamental_matrix(poses[i], poses[j], Ks[i], Ks[j])
        np.testing.assert_allclose(Ks[j].T@F@Ks[i], E, atol=1e-14)
        np.testing.assert_allclose(np.sum(normalized_pixels(pair['xy_j'], Ks[j])*
            (normalized_pixels(pair['xy_i'], Ks[i])@E.T), axis=-1), 0., atol=1e-14)
        assert np.max(sampson_squared(pair['xy_i'], pair['xy_j'], F)) < 1e-23


def test_known_right_perturbation_increases_sampson():
    poses, Ks, pairs, _ = synthetic(); p = pairs[0]
    noisy = poses.copy(); noisy[1] = poses[1]@se3_exp([.01, -.02, .01, .02, -.01, .01])
    error = sampson_squared(p['xy_i'], p['xy_j'], fundamental_matrix(noisy[0], noisy[1], Ks[0], Ks[1]))
    assert error.mean() > 1.


def test_numpy_torch_geometry_and_gradients_agree():
    import torch
    poses, Ks, matches, _ = synthetic(); pairs = torch.tensor([p['indices'] for p in matches])
    variable = torch.tensor(poses, requires_grad=True)
    F, _ = fundamental_torch(variable, torch.tensor(np.linalg.inv(Ks)), pairs)
    for k, p in enumerate(matches):
        i, j = p['indices']; expected = fundamental_matrix(poses[i], poses[j], Ks[i], Ks[j])
        np.testing.assert_allclose(F[k].detach().numpy(), expected, atol=1e-17)
    p = matches[0]
    residual = sampson_residual_torch(torch.tensor(homogeneous(p['xy_i'])),
        torch.tensor(homogeneous(p['xy_j'])), F[0].expand(len(p['xy_i']), -1, -1))
    residual.square().mean().backward()
    assert torch.isfinite(variable.grad).all()


def test_synthetic_optimizer_improves_actual_rotation_and_centers():
    clean, Ks, pairs, _ = synthetic()
    noise = np.array([[0., 0, 0, 0, 0, 0], [.01, -.008, .005, .01, -.015, .01],
        [-.008, .012, -.009, -.01, .01, -.012], [.009, .004, -.01, .012, -.01, .008]])
    initial = apply_correction(clean, noise, 'camera'); cfg = settings()
    cfg.update(iterations=400, lambda_rotation=10., lambda_translation=10.)
    final, delta, log = refine(initial, Ks, pairs, 1.5, cfg)
    before, after = pose_metrics(initial, clean), pose_metrics(final, clean)
    assert after['rotation_mean_deg'] < before['rotation_mean_deg']*.3
    assert after['center_rmse_normalized'] < before['center_rmse_normalized']*.6
    assert log['after']['mean_sampson_px'] < log['before']['mean_sampson_px']*.05
    np.testing.assert_array_equal(final[0], initial[0]); np.testing.assert_array_equal(delta[0], 0.)


def test_zero_right_correction_and_noncommutative_composition():
    pose = se3_exp([.2, -.1, .3, 1., 2., 3.])[None]
    zero = np.zeros((1, 6)); np.testing.assert_array_equal(apply_correction(pose, zero, 'camera'), pose)
    delta = np.array([[.03, .01, -.02, .01, .02, -.03]])
    np.testing.assert_allclose(apply_correction(pose, delta, 'camera'), pose@se3_exp(delta))
    assert not np.allclose(apply_correction(pose, delta, 'camera'), se3_exp(delta)@pose)


def test_total_bounds_and_fixed_gauge_with_large_optimizer_steps():
    poses, Ks, pairs, _ = synthetic(); poses[1] = poses[1]@se3_exp([.02, -.02, 0, .02, 0, .01])
    cfg = settings(); cfg.update(iterations=20, rotation_lr=.5, translation_lr_ratio=.5,
        max_rotation_deg=.2, max_translation_ratio=.001)
    final, delta, log = refine(poses, Ks, pairs, 2., cfg)
    assert np.rad2deg(np.linalg.norm(delta[:, :3], axis=-1)).max() <= .2
    assert (np.linalg.norm(delta[:, 3:], axis=-1)/2.).max() <= .001
    assert sum(log['bounded_steps'].values()) > 0
    np.testing.assert_array_equal(final[0], poses[0])


def test_pair_graph_sparse_connected_and_degree_bounded():
    pairs = neighbor_pairs(32, 3, 6); assert len(pairs) == 90
    assert all(0 < j-i <= 3 for i, j in pairs)
    assert np.bincount(np.array(pairs).ravel()).max() == 6


def test_degenerate_centers_and_unanchored_disconnected_graph_are_rejected():
    with pytest.raises(ValueError): essential_matrix(np.eye(4), np.eye(4))
    poses, Ks, pairs, _ = synthetic()
    cfg = settings(); cfg['lambda_prior'] = 0.
    with pytest.raises(ValueError): refine(poses, Ks, pairs[:1], 1., cfg)


def test_disconnected_components_retain_vggt_prior_and_isolated_pose():
    poses, Ks, pairs, _ = synthetic(); cfg = settings(); cfg['iterations'] = 5
    poses[1] = poses[1]@se3_exp([.01, .01, 0, 0, 0, 0])
    final, delta, log = refine(poses, Ks, pairs[:1], 1., cfg)
    assert log['matching_graph_components'] == [[0, 1], [2], [3]]
    np.testing.assert_array_equal(final[2:], poses[2:])
    np.testing.assert_array_equal(delta[2:], 0.)


def create_train(tmp_path):
    from PIL import Image
    image_dir = tmp_path/'train/images'; image_dir.mkdir(parents=True)
    names = [f'{i:02d}' for i in range(32)]
    for name in names: Image.new('RGB', (448, 448)).save(image_dir/f'{name}.png')
    # Held-out images are outside the matcher input directory, with poison bytes.
    (tmp_path/'gt').mkdir(); (tmp_path/'gt/test.png').write_bytes(b'must never be opened')
    return image_dir.parent, dict(train=names, test=['test'])


def test_train_only_paths_reject_test_overlap_extra_files_and_duplicates(tmp_path):
    train, split = create_train(tmp_path); assert len(train_paths(train, split)) == 32
    invalid = copy.deepcopy(split); invalid['test'] = [split['train'][0]]
    with pytest.raises(ValueError): train_paths(train, invalid)
    invalid = copy.deepcopy(split); invalid['train'][-1] = invalid['train'][0]
    with pytest.raises(ValueError): train_paths(train, invalid)
    (train/'images/test.png').write_bytes(b'forbidden')
    with pytest.raises(ValueError): train_paths(train, split)


def test_matching_cache_reuses_exact_arrays_without_network_or_test_rgb(tmp_path):
    train, split = create_train(tmp_path); poses, Ks, pairs, _ = synthetic()
    cfg = read_yaml(ROOT/'configs/methods/rgb_pose_refine.yaml')['rgb_matching']; cfg['min_coverage'] = .001
    calls = []
    def factory(paths):
        assert {p.stem for p in paths} == set(split['train']); calls.append(paths)
        p = pairs[0]
        valid = ((p['xy_i'] >= 0)&(p['xy_i'] < 448)&(p['xy_j'] >= 0)&(p['xy_j'] < 448)).all(-1)
        return lambda i, j:(p['xy_i'][valid], p['xy_j'][valid], np.full(valid.sum(), .9))
    first, _, _ = cached_matches(train, split, tmp_path/'cache', cfg, {'weights':'fake'}, factory)
    def forbidden(paths): raise AssertionError('Network called on cache hit')
    second, _, _ = cached_matches(train, split, tmp_path/'cache', cfg, {'weights':'fake'}, forbidden)
    assert len(calls) == 1 and len(first) == 90
    for a, b in zip(first, second):
        for key in ('xy_i', 'xy_j', 'confidence'): np.testing.assert_array_equal(a[key], b[key])
    changed = dict(cfg, ransac_threshold_px=2.)
    with pytest.raises(ValueError): cached_matches(train, split, tmp_path/'cache', changed, {'weights':'fake'}, forbidden)
    (tmp_path/'cache/pair_00_01.npz').write_bytes(b'changed')
    with pytest.raises(ValueError): load_matches(tmp_path/'cache')


def test_geometric_filter_rejects_outliers_and_weak_pairs():
    _, _, pairs, _ = synthetic(); p = pairs[0]
    cfg = read_yaml(ROOT/'configs/methods/rgb_pose_refine.yaml')['rgb_matching']
    xy_j = p['xy_j'].copy(); xy_j[:50] = np.random.default_rng(1).uniform(0, 448, (50, 2))
    mask, F, row = verify_pair(p['xy_i'], xy_j, np.ones(200)*.8, [(448, 448)]*2, cfg, 0)
    assert mask[:50].sum() < 10 and mask[50:].sum() >= 140
    assert F.shape == (3, 3) and row['accepted'] and row['pair_weight'] > 0
    weak = dict(cfg, min_inliers=300)
    assert not verify_pair(p['xy_i'], xy_j, np.ones(200)*.8, [(448, 448)]*2, weak, 0)[2]['accepted']


def test_scene_scale_normalized_optimizer_is_equivariant():
    poses, Ks, pairs, _ = synthetic(); poses[1] = poses[1]@se3_exp([.01, .01, 0, .01, 0, -.01])
    cfg = settings(); cfg['iterations'] = 15
    _, delta, _ = refine(poses, Ks, pairs, 1.5, cfg)
    scaled = poses.copy(); scaled[:, :3, 3] *= 7.
    _, scaled_delta, _ = refine(scaled, Ks, pairs, 10.5, cfg)
    np.testing.assert_allclose(scaled_delta[:, :3], delta[:, :3], atol=1e-12)
    np.testing.assert_allclose(scaled_delta[:, 3:]/7., delta[:, 3:], atol=1e-12)








def test_sampson_autograd_agrees_with_finite_differences():
    import torch
    from benchmark.pose_refinement.se3 import se3_exp_torch
    poses, Ks, matches, _ = synthetic(); p = matches[0]
    def residual_at(delta):
        corrected = torch.tensor(poses[1])@se3_exp_torch(delta)
        current = torch.stack((torch.tensor(poses[0]), corrected))
        F, _ = fundamental_torch(current, torch.tensor(np.linalg.inv(Ks[:2])), torch.tensor([[0, 1]]))
        residual = sampson_residual_torch(torch.tensor(homogeneous(p['xy_i'])),
            torch.tensor(homogeneous(p['xy_j'])), F[0].expand(len(p['xy_i']), -1, -1))
        return residual.square().mean()
    delta = torch.tensor([.01, -.02, .01, .01, -.01, .005], dtype=torch.float64, requires_grad=True)
    residual_at(delta).backward(); finite = []
    for i in range(6):
        epsilon = torch.zeros(6, dtype=torch.float64); epsilon[i] = 1e-6
        finite.append(float((residual_at(delta.detach()+epsilon)-residual_at(delta.detach()-epsilon))/2e-6))
    np.testing.assert_allclose(delta.grad.numpy(), finite, rtol=1e-7, atol=1e-5)

import copy
import numpy as np
import pytest
from benchmark.utils.io import ROOT, read_yaml
from benchmark.pose_refinement.se3 import se3_exp, apply_correction
from benchmark.pose_refinement.pose_metrics import pose_metrics
from benchmark.intrinsic_refinement.optimizer import scaled_intrinsics, refine, validate_intrinsic_settings
from test_rgb_pose_refinement import synthetic, settings as pose_settings


def intrinsic_settings():
    return read_yaml(ROOT/'configs/methods/rgb_pose_focal_refine.yaml')['intrinsic_refinement']


def calibrated_scene():
    poses, Ks, _, points = synthetic()
    rotations = [[0, .10, 0], [.03, -.12, .04], [-.05, .18, -.03], [.12, .04, -.06]]
    poses[:, :3, :3] = se3_exp(np.column_stack((rotations, np.zeros((4, 3)))))[:, :3, :3]
    pixels = []
    for pose, K in zip(poses, Ks):
        q = (points-pose[:3, 3])@pose[:3, :3]; uv = q@K.T; pixels.append(uv[:, :2]/uv[:, 2:])
    matches = [dict(indices=[i, j], names=[str(i), str(j)], xy_i=pixels[i], xy_j=pixels[j],
        confidence=np.ones(len(points)), weight=1.) for i in range(4) for j in range(i+1, 4)]
    return poses, Ks, matches


def test_shared_scale_preserves_principal_point_aspect_ratio_and_skew():
    _, Ks, _ = calibrated_scene(); Ks[0, 0, 1] = 3.
    scaled = scaled_intrinsics(Ks, np.log(1.07))
    np.testing.assert_allclose(scaled[:, 0, 0], 1.07*Ks[:, 0, 0])
    np.testing.assert_allclose(scaled[:, 1, 1], 1.07*Ks[:, 1, 1])
    np.testing.assert_array_equal(scaled[:, :2, 2], Ks[:, :2, 2])
    np.testing.assert_array_equal(scaled[:, 0, 1], Ks[:, 0, 1])
    np.testing.assert_array_equal(scaled[:, 2], Ks[:, 2])
    np.testing.assert_array_equal(scaled_intrinsics(Ks, 0.), Ks)


def test_focal_gradient_matches_finite_difference():
    import torch
    from benchmark.rgb_pose_refinement.geometry import fundamental_torch, sampson_residual_torch, homogeneous
    poses, Ks, pairs = calibrated_scene(); p = pairs[0]
    def loss(a):
        K = scaled_intrinsics(torch.tensor(Ks), a)
        F, _ = fundamental_torch(torch.tensor(poses), torch.linalg.inv(K), torch.tensor([[0, 1]]))
        r = sampson_residual_torch(torch.tensor(homogeneous(p['xy_i'])), torch.tensor(homogeneous(p['xy_j'])),
            F[0].expand(len(p['xy_i']), -1, -1))
        return r.square().mean()
    a = torch.tensor(.02, dtype=torch.float64, requires_grad=True); loss(a).backward()
    finite = (loss(a.detach()+1e-6)-loss(a.detach()-1e-6))/2e-6
    assert float(a.grad) == pytest.approx(float(finite), rel=1e-7)


@pytest.mark.parametrize('factor', [.95, 1.05])
def test_focal_only_recovers_synthetic_scale_and_freezes_every_pose(factor):
    clean, Ks, matches = calibrated_scene(); noisy_K = scaled_intrinsics(Ks, np.log(factor))
    final, K, delta, log = refine(clean, noisy_K, matches, 1.5, pose_settings(), intrinsic_settings(), 'focal_only')
    np.testing.assert_array_equal(final, clean); np.testing.assert_array_equal(delta, 0.)
    assert log['learned_focal_scale'] == pytest.approx(1/factor, abs=.003)
    assert np.abs(K[:, 0, 0]/Ks[:, 0, 0]-1).mean() < .003
    assert log['focal_parameter_count'] == 1


@pytest.mark.parametrize('schedule', ['joint', 'staged'])
def test_joint_pose_focal_synthetic_recovery_and_gauge(schedule):
    clean, Ks, pairs = calibrated_scene()
    noise = np.array([[0., 0, 0, 0, 0, 0], [.005, -.004, .003, .008, -.006, .005],
        [-.005, .004, -.002, -.004, .007, -.006], [.004, .003, -.005, .005, -.008, .004]])
    noisy = apply_correction(clean, noise, 'camera'); noisy_K = scaled_intrinsics(Ks, np.log(.95))
    intrinsic = intrinsic_settings(); intrinsic['schedule'] = schedule
    cfg = pose_settings(); cfg.update(lambda_rotation=10., lambda_translation=10.)
    final, K, delta, log = refine(noisy, noisy_K, pairs, 1.5, cfg, intrinsic)
    assert pose_metrics(final, clean)['rotation_mean_deg'] < pose_metrics(noisy, clean)['rotation_mean_deg']*.7
    assert np.abs(K[:, 0, 0]/Ks[:, 0, 0]-1).mean() < .02
    np.testing.assert_array_equal(final[0], noisy[0]); np.testing.assert_array_equal(delta[0], 0.)
    assert sum(s['iterations'] for s in log['stages']) == intrinsic['total_iterations']


def test_staged_bounds_apply_to_original_intrinsics_and_total_pose_twists():
    clean, Ks, pairs = calibrated_scene(); noisy_K = scaled_intrinsics(Ks, np.log(.9))
    cfg = pose_settings(); cfg.update(max_rotation_deg=.1, max_translation_ratio=.001,
        rotation_lr=.1, translation_lr_ratio=.1)
    intrinsic = intrinsic_settings(); intrinsic.update(max_focal_scale_change=.001,
        focal_lr=.1, total_iterations=30, focal_stage_iterations=10)
    final, K, delta, log = refine(clean, noisy_K, pairs, 1.5, cfg, intrinsic)
    assert .999-1e-12 <= log['learned_focal_scale'] <= 1.001+1e-12
    assert np.rad2deg(np.linalg.norm(delta[:, :3], axis=-1)).max() <= .1+1e-12
    assert np.linalg.norm(delta[:, 3:], axis=-1).max()/1.5 <= .001+1e-12
    assert sum(s['bound_hits']['focal'] for s in log['stages']) > 0
    np.testing.assert_array_equal(K[:, :2, 2], noisy_K[:, :2, 2])


def test_disabled_intrinsics_exactly_reuses_existing_pose_optimizer():
    from benchmark.rgb_pose_refinement.optimizer import refine as previous
    poses, Ks, pairs = calibrated_scene(); poses[1] = poses[1]@se3_exp([.01, .01, 0, .01, 0, 0])
    cfg = pose_settings(); cfg['iterations'] = 10
    expected, delta_expected, _ = previous(poses, Ks, pairs, 1.5, cfg)
    intrinsic = intrinsic_settings(); intrinsic['enabled'] = False
    final, K, delta, _ = refine(poses, Ks, pairs, 1.5, cfg, intrinsic)
    np.testing.assert_array_equal(final, expected); np.testing.assert_array_equal(delta, delta_expected)
    np.testing.assert_array_equal(K, Ks)


@pytest.mark.parametrize('key,value', [('optimize_principal_point', True), ('model', 'per_camera'),
    ('max_focal_scale_change', 1.), ('initial_log_scale', .5), ('schedule', 'unknown')])
def test_unsupported_intrinsic_models_and_bounds_rejected(key, value):
    cfg = intrinsic_settings(); cfg[key] = value
    with pytest.raises(ValueError): validate_intrinsic_settings(cfg)

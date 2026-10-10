"""RGB-only bounded pose + one scene-shared focal scale, without reference access."""
import numpy as np
from benchmark.pose_refinement.se3 import se3_exp_torch, apply_correction, project_corrections
from benchmark.rgb_pose_refinement.geometry import homogeneous, fundamental_torch, sampson_residual_torch, pair_residuals
from benchmark.rgb_pose_refinement.optimizer import graph_components


def validate_intrinsic_settings(settings):
    if settings['model'] != 'shared_focal_scale' or settings['optimize_principal_point']:
        raise ValueError('Only one shared focal scale with fixed principal point is supported')
    change = settings['max_focal_scale_change']
    if not 0 < change < 1: raise ValueError('Focal change bound must be in (0,1)')
    low, high = np.log1p(-change), np.log1p(change)
    if not low <= settings['initial_log_scale'] <= high: raise ValueError('Initial focal scale exceeds bounds')
    if settings['schedule'] not in ('joint', 'staged'): raise ValueError('Unknown optimization schedule')
    if settings['lambda_focal'] < 0 or settings['focal_lr'] <= 0: raise ValueError('Invalid focal prior or learning rate')
    if not 0 < settings['focal_stage_iterations'] < settings['total_iterations']:
        raise ValueError('Focal stage must fit inside total iteration budget')
    return low, high


def scaled_intrinsics(Ks, log_scale):
    """Change only fx/fy by exp(a). Preserve skew, principal point and last row."""
    import torch
    if torch.is_tensor(Ks):
        mask = torch.diag(Ks.new_tensor([1., 1., 0.]))
        return Ks*(1+(log_scale.exp()-1)*mask)
    result = np.array(Ks, dtype=float, copy=True)
    result[..., 0, 0] *= np.exp(log_scale); result[..., 1, 1] *= np.exp(log_scale)
    return result


def refine(initial, Ks, matches, scene_scale, pose_settings, settings, mode='pose_focal', device='cpu'):
    """staged: 200 focal-only + 400 joint; joint: 600 simultaneous Adam updates.

    All focal corrections/prior/bounds remain relative to INPUT K across stages.
    Pose corrections/prior/bounds remain relative to INPUT poses. Adam moments
    reset between stages; stage 2 starts from the stage-1 scalar, never reference K.
    """
    import torch
    lower, upper = validate_intrinsic_settings(settings)
    if mode not in ('pose_only', 'focal_only', 'pose_focal'): raise ValueError('Invalid refinement mode')
    if mode == 'pose_only' or not settings['enabled']:
        from benchmark.rgb_pose_refinement.optimizer import refine as pose_refine
        final, delta, log = pose_refine(initial, Ks, matches, scene_scale, pose_settings, device)
        log.update(mode='pose_only', learned_log_focal_scale=0., learned_focal_scale=1.,
            focal_parameter_count=0, schedule='previous_pose_only',
            focal_bounds=[1-settings['max_focal_scale_change'], 1+settings['max_focal_scale_change']])
        return final, np.array(Ks, copy=True), delta, log
    if not matches or not np.isfinite(scene_scale) or scene_scale <= 0: raise ValueError('Invalid matching/scene inputs')
    components = graph_components(len(initial), [p['indices'] for p in matches])
    if len(components) > 1 and any(pose_settings[k] <= 0 for k in ('lambda_prior', 'lambda_rotation', 'lambda_translation')):
        raise ValueError('Disconnected graph requires positive pose priors')
    if any(p['weight'] <= 0 or len(p['xy_i']) == 0 for p in matches): raise ValueError('Invalid accepted pair')
    reference = pose_settings['reference_camera_index']
    if not 0 <= reference < len(initial): raise ValueError('Invalid gauge camera')
    tensor = lambda a:torch.as_tensor(a, dtype=torch.float64, device=device).detach()
    poses0, K0 = tensor(initial), tensor(Ks)
    pairs = torch.tensor([p['indices'] for p in matches], dtype=torch.long, device=device)
    counts = torch.tensor([len(p['xy_i']) for p in matches], dtype=torch.long, device=device)
    pair_ids = torch.repeat_interleave(torch.arange(len(matches), device=device), counts)
    xi = tensor(homogeneous(np.concatenate([p['xy_i'] for p in matches])))
    xj = tensor(homogeneous(np.concatenate([p['xy_j'] for p in matches])))
    pair_weights = np.array([p['weight'] for p in matches]); pair_weights /= pair_weights.sum()
    weights = tensor(np.concatenate([w*np.maximum(p['confidence'], 1e-8)/np.maximum(p['confidence'], 1e-8).sum()
        for p,w in zip(matches, pair_weights)]))
    active = tensor(np.ones((len(initial), 1))); active[reference] = 0.
    delta_best = np.zeros((len(initial), 6)); a_best = settings['initial_log_scale']
    if mode == 'focal_only': stages = [('focal', settings['total_iterations'])]
    elif settings['schedule'] == 'joint': stages = [('joint', settings['total_iterations'])]
    else: stages = [('focal', settings['focal_stage_iterations']),
        ('joint', settings['total_iterations']-settings['focal_stage_iterations'])]
    logs = []
    for stage_name, iterations in stages:
        pose_enabled = stage_name == 'joint'
        rotation = torch.nn.Parameter(tensor(delta_best[:, :3]).clone(), requires_grad=pose_enabled)
        translation = torch.nn.Parameter(tensor(delta_best[:, 3:]/scene_scale).clone(), requires_grad=pose_enabled)
        a = torch.nn.Parameter(tensor(a_best).clone())  # exactly ONE scalar per scene
        groups = [dict(params=[a], lr=settings['focal_lr'])]
        if pose_enabled:
            groups += [dict(params=[rotation], lr=pose_settings['rotation_lr']),
                dict(params=[translation], lr=pose_settings['translation_lr_ratio'])]
        adam = torch.optim.Adam(groups)
        stage_best, best_step = float('inf'), 0
        history, bound_hits = [], dict(rotation=0, translation=0, focal=0)
        rejected = 0
        for step in range(iterations+1):
            adam.zero_grad(set_to_none=True)
            delta = torch.cat((rotation*active, translation*active*scene_scale), -1)
            poses = poses0@se3_exp_torch(delta)
            K = scaled_intrinsics(K0, a)
            F, lengths = fundamental_torch(poses, torch.linalg.inv(K), pairs)
            residual = sampson_residual_torch(xi, xj, F[pair_ids])
            absolute, huber = residual.abs(), pose_settings['huber_px']
            robust = torch.where(absolute <= huber, .5*residual.square(), huber*(absolute-.5*huber))
            match_loss = (weights*robust).sum()
            pose_prior = ((pose_settings['lambda_rotation']*rotation.square().sum(-1)+
                pose_settings['lambda_translation']*translation.square().sum(-1))*active[:, 0]).sum()/(len(initial)-1)
            focal_prior = a.square()
            loss = match_loss+pose_settings['lambda_prior']*pose_prior+settings['lambda_focal']*focal_prior
            value = float(loss.detach())
            if not np.isfinite(value): raise RuntimeError('Nonfinite joint RGB objective')
            feasible = bool((lengths > pose_settings['minimum_baseline_ratio']*scene_scale).all())
            rejected += int(not feasible)
            if feasible and value < stage_best:
                stage_best, best_step = value, step
                delta_best, a_best = delta.detach().cpu().numpy().copy(), float(a.detach())
            if step % 25 == 0 or step == iterations:
                history.append(dict(step=step, total_loss=value, match_loss=float(match_loss.detach()),
                    pose_prior=float(pose_prior.detach()), focal_prior=float(focal_prior.detach()),
                    focal_scale=float(a.detach().exp()), feasible=feasible))
            if step == iterations: break
            loss.backward()
            parameters = [a, rotation, translation] if pose_enabled else [a]
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError('Nonfinite joint pose/focal gradient')
            adam.step()
            with torch.no_grad():
                bound_hits['focal'] += int(bool((a < lower)|(a > upper))); a.clamp_(lower, upper)
                if pose_enabled:
                    for name, parameter, maximum in [('rotation', rotation, np.deg2rad(pose_settings['max_rotation_deg'])),
                            ('translation', translation, pose_settings['max_translation_ratio'])]:
                        norm = parameter.norm(dim=-1, keepdim=True)
                        bound_hits[name] += int((norm > maximum).sum())
                        parameter.mul_((maximum/norm.clamp_min(1e-20)).clamp_max(1.)); parameter[reference] = 0.
        if not np.isfinite(stage_best): raise ValueError('No feasible joint pose/focal iterate')
        logs.append(dict(stage=stage_name, iterations=iterations, best_step=best_step, best_total_loss=stage_best,
            final_log_focal_scale=a_best, final_focal_scale=float(np.exp(a_best)), history=history,
            bound_hits=bound_hits, degenerate_iterates=rejected))
    delta_best, _ = project_corrections(delta_best, scene_scale, pose_settings); delta_best[reference] = 0.
    final = apply_correction(initial, delta_best, 'camera'); K = scaled_intrinsics(Ks, a_best)
    np.testing.assert_array_equal(final[reference], initial[reference])
    np.testing.assert_array_equal(K[:, :2, 2], np.asarray(Ks)[:, :2, 2])
    log = dict(mode=mode, schedule=settings['schedule'], stages=logs, focal_parameter_count=1,
        learned_log_focal_scale=a_best, learned_focal_scale=float(np.exp(a_best)),
        focal_bounds=[1-settings['max_focal_scale_change'], 1+settings['max_focal_scale_change']],
        matching_graph_components=components, before=pair_residuals(initial, Ks, matches),
        after=pair_residuals(final, K, matches), scene_scale=float(scene_scale),
        correction_rotation_deg=np.rad2deg(np.linalg.norm(delta_best[:, :3], axis=-1)).tolist(),
        correction_translation_twist_ratio=(np.linalg.norm(delta_best[:, 3:], axis=-1)/scene_scale).tolist(),
        correction_center_displacement_ratio=(np.linalg.norm(final[:, :3, 3]-initial[:, :3, 3], axis=-1)/scene_scale).tolist())
    return final, K, delta_best, log

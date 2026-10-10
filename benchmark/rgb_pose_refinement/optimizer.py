"""Joint deterministic pose-only optimization; no evaluation/reference imports."""
import numpy as np
from benchmark.pose_refinement.se3 import se3_exp_torch, apply_correction, project_corrections
from .geometry import fundamental_torch, sampson_residual_torch, homogeneous, pair_residuals


def graph_components(count, pairs):
    remaining, components = set(range(count)), []
    while remaining:
        reached = {min(remaining)}
        while True:
            previous = len(reached)
            for i, j in pairs:
                if i in reached or j in reached: reached.update((i, j))
            if len(reached) == previous: break
        remaining.difference_update(reached); components.append(sorted(reached))
    return components


def refine(initial, Ks, matches, scene_scale, settings, device='cpu'):
    """T=T0 exp([omega,v]); prior averages lambda_r|omega|²+lambda_t|v/s|².

    Epipolar loss cannot determine global scale (even with camera 0 fixed).
    The VGGT prior retains the input scale. The best feasible TOTAL objective
    selects the iterate; reference accuracy never selects a pose or parameter.
    """
    import torch
    if not np.isfinite(scene_scale) or scene_scale <= 0: raise ValueError('Invalid scene scale')
    if not matches: raise ValueError('No accepted training correspondences')
    components = graph_components(len(initial), [p['indices'] for p in matches])
    # The pose prior fixes disconnected components to the VGGT frame. Their
    # relative placement has no RGB constraint and must be reported explicitly.
    if len(components) > 1 and (settings['lambda_prior'] <= 0 or
        settings['lambda_rotation'] <= 0 or settings['lambda_translation'] <= 0):
        raise ValueError('Disconnected matching graph requires positive pose priors')
    if any(len(p['xy_i']) == 0 or p['weight'] <= 0 for p in matches):
        raise ValueError('Accepted pairs need positive weights and observations')
    reference = settings['reference_camera_index']
    if not 0 <= reference < len(initial): raise ValueError('Invalid reference camera')
    dtype = torch.float64
    tensor = lambda a: torch.as_tensor(a, dtype=dtype, device=device).detach()
    poses0, inverse_K = tensor(initial), tensor(np.linalg.inv(Ks))
    pairs = torch.tensor([p['indices'] for p in matches], dtype=torch.long, device=device)
    counts = np.array([len(p['xy_i']) for p in matches])
    pair_ids = torch.repeat_interleave(torch.arange(len(matches), device=device),
        torch.as_tensor(counts, device=device))
    xi = tensor(homogeneous(np.concatenate([p['xy_i'] for p in matches])))
    xj = tensor(homogeneous(np.concatenate([p['xy_j'] for p in matches])))
    # Each pair contributes its mean confidence-weighted loss times reliability.
    pair_weights = np.array([p['weight'] for p in matches]); pair_weights /= pair_weights.sum()
    weights = tensor(np.concatenate([w*np.maximum(p['confidence'], 1e-8)/
        np.maximum(p['confidence'], 1e-8).sum() for p, w in zip(matches, pair_weights)]))
    rotation = torch.nn.Parameter(torch.zeros((len(initial), 3), dtype=dtype, device=device))
    translation = torch.nn.Parameter(torch.zeros_like(rotation))  # dimensionless v/scene_scale
    active = torch.ones((len(initial), 1), dtype=dtype, device=device); active[reference] = 0.
    adam = torch.optim.Adam([dict(params=[rotation], lr=settings['rotation_lr']),
        dict(params=[translation], lr=settings['translation_lr_ratio'])])
    best, best_loss, best_step, history = np.zeros((len(initial), 6)), float('inf'), 0, []
    clamp_counts = dict(rotation=0, translation=0); rejected = 0
    for step in range(settings['iterations']+1):
        adam.zero_grad(set_to_none=True)
        delta = torch.cat((rotation*active, translation*active*scene_scale), -1)
        poses = poses0 @ se3_exp_torch(delta)
        F, lengths = fundamental_torch(poses, inverse_K, pairs)
        residual = sampson_residual_torch(xi, xj, F[pair_ids])
        abs_r, huber = residual.abs(), settings['huber_px']
        robust = torch.where(abs_r <= huber, .5*residual.square(), huber*(abs_r-.5*huber))
        match_loss = (weights*robust).sum()
        prior = ((settings['lambda_rotation']*rotation.square().sum(-1)+
            settings['lambda_translation']*translation.square().sum(-1))*active[:, 0]).sum()/(len(initial)-1)
        loss = match_loss + settings['lambda_prior']*prior
        value = float(loss.detach())
        if not np.isfinite(value): raise RuntimeError('Nonfinite RGB pose objective')
        feasible = bool((lengths > settings['minimum_baseline_ratio']*scene_scale).all())
        rejected += int(not feasible)
        if feasible and value < best_loss:
            best_loss, best, best_step = value, delta.detach().cpu().numpy().copy(), step
        if step % 25 == 0 or step == settings['iterations']:
            history.append(dict(step=step, loss=value, match_loss=float(match_loss.detach()),
                prior_loss=float(prior.detach()), feasible=feasible))
        if step == settings['iterations']: break
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in (rotation, translation)):
            raise RuntimeError('Missing/nonfinite RGB pose gradients')
        adam.step()
        with torch.no_grad():
            for name, parameter, maximum in [('rotation', rotation, np.deg2rad(settings['max_rotation_deg'])),
                    ('translation', translation, settings['max_translation_ratio'])]:
                norm = parameter.norm(dim=-1, keepdim=True)
                clamp_counts[name] += int((norm > maximum).sum())
                parameter.mul_((maximum/norm.clamp_min(1e-20)).clamp_max(1.))
                parameter[reference] = 0.
    if not np.isfinite(best_loss): raise ValueError('No feasible epipolar iterate')
    best, _ = project_corrections(best, scene_scale, settings); best[reference] = 0.
    final = apply_correction(initial, best, 'camera')
    if not np.array_equal(final[reference], initial[reference]): raise RuntimeError('Gauge camera moved')
    log = dict(best_step=best_step, best_total_loss=best_loss, history=history,
        matching_graph_components=components,
        bounded_steps=clamp_counts, degenerate_iterates=rejected, reference_camera_index=reference,
        scene_scale=float(scene_scale), composition='T_i0 @ exp(delta_i)',
        before=pair_residuals(initial, Ks, matches), after=pair_residuals(final, Ks, matches),
        correction_rotation_deg=np.rad2deg(np.linalg.norm(best[:, :3], axis=-1)).tolist(),
        correction_translation_twist_ratio=(np.linalg.norm(best[:, 3:], axis=-1)/scene_scale).tolist(),
        correction_center_displacement_ratio=(np.linalg.norm(final[:, :3, 3]-initial[:, :3, 3], axis=-1)/scene_scale).tolist())
    return final, best, log

"""OpenCV c2w epipolar geometry, x_j^T F_ji x_i = 0.

Distances use prepared-image pixels. Intrinsics and correspondences are fixed.
No reference poses, VGGT tracks, XYZ or depths enter this module.
"""
import numpy as np


def homogeneous(xy):
    xy = np.asarray(xy, dtype=np.float64)
    return np.concatenate((xy, np.ones(xy.shape[:-1]+(1,))), axis=-1)


def normalized_pixels(xy, K):
    rays = homogeneous(xy) @ np.linalg.inv(K).T
    return rays / rays[..., 2:]


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def essential_matrix(c2w_i, c2w_j):
    # X_j = R_j^T R_i X_i + R_j^T (C_i-C_j).
    R = c2w_j[:3, :3].T @ c2w_i[:3, :3]
    t = c2w_j[:3, :3].T @ (c2w_i[:3, 3]-c2w_j[:3, 3])
    if np.linalg.norm(t) <= 1e-12:
        raise ValueError('Epipolar geometry is undefined for coincident centers')
    return skew(t/np.linalg.norm(t)) @ R


def fundamental_matrix(c2w_i, c2w_j, K_i, K_j):
    return np.linalg.inv(K_j).T @ essential_matrix(c2w_i, c2w_j) @ np.linalg.inv(K_i)


def sampson_squared(xy_i, xy_j, F):
    xi, xj = homogeneous(xy_i), homogeneous(xy_j)
    a, b = xi @ F.T, xj @ F
    denominator = (a[..., :2]**2).sum(-1)+(b[..., :2]**2).sum(-1)
    if np.any(denominator <= 1e-24):
        raise ValueError('Undefined Sampson distance at degenerate epipolar lines')
    return np.sum(xj*a, axis=-1)**2/denominator


def fundamental_torch(poses, inverse_K, pairs):
    import torch
    i, j = pairs.unbind(-1)
    Ri, Rj = poses[i, :3, :3], poses[j, :3, :3]
    Rt = Rj.transpose(-1, -2)
    rotation = Rt @ Ri
    translation = (Rt @ (poses[i, :3, 3]-poses[j, :3, 3])[..., None])[..., 0]
    lengths = translation.norm(dim=-1)
    t = translation / lengths.clamp_min(1e-12)[..., None]
    x, y, z = t.unbind(-1); zero = torch.zeros_like(x)
    cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    return inverse_K[j].transpose(-1, -2) @ cross @ rotation @ inverse_K[i], lengths


def sampson_residual_torch(xi, xj, F):
    a = (F @ xi[..., None])[..., 0]
    b = (F.transpose(-1, -2) @ xj[..., None])[..., 0]
    denominator = (a[..., :2]**2).sum(-1)+(b[..., :2]**2).sum(-1)
    return (xj*a).sum(-1)/denominator.clamp_min(1e-24).sqrt()


def pair_residuals(poses, Ks, matches):
    rows = []
    for pair in matches:
        i, j = pair['indices']
        error = np.sqrt(sampson_squared(pair['xy_i'], pair['xy_j'],
            fundamental_matrix(poses[i], poses[j], Ks[i], Ks[j])))
        rows.append(dict(indices=[i, j], names=pair['names'], count=len(error),
            mean_sampson_px=float(error.mean()), median_sampson_px=float(np.median(error)),
            rms_sampson_px=float(np.sqrt(np.mean(error**2))), pair_weight=pair['weight']))
    errors = np.concatenate([np.sqrt(sampson_squared(p['xy_i'], p['xy_j'],
        fundamental_matrix(poses[p['indices'][0]], poses[p['indices'][1]],
            Ks[p['indices'][0]], Ks[p['indices'][1]]))) for p in matches])
    return dict(mean_sampson_px=float(errors.mean()), median_sampson_px=float(np.median(errors)),
        rms_sampson_px=float(np.sqrt(np.mean(errors**2))), pairs=rows)

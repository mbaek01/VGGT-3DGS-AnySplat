"""OpenCV c2w cameras; configurable world/left or camera/right corrections.

VGGT/COLMAP extrinsics are w2c. gsplat and this package store c2w.
Twists are [rotation radians, translation in normalized reconstruction units].
world: exp(delta) @ c2w; camera: c2w @ exp(delta).
The translation in exp(delta) is J(rotation) @ v. Bounds apply to the total
twist [omega, v] relative to the ORIGINAL pose, not incremental Adam updates.
"""
import numpy as np


def invert_pose(pose):
    pose = np.asarray(pose)
    if pose.shape[-2:] == (3, 4):
        bottom = np.broadcast_to([0., 0., 0., 1.], pose.shape[:-2] + (1, 4))
        pose = np.concatenate((pose, bottom), axis=-2)
    if pose.shape[-2:] != (4, 4):
        raise ValueError("Expected (..., 3, 4) or (..., 4, 4) camera matrix")
    return np.linalg.inv(pose)


def w2c_to_c2w(pose):
    return invert_pose(pose)


def c2w_to_w2c(pose):
    return invert_pose(pose)


def se3_exp(delta):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape[-1] != 6 or not np.isfinite(delta).all():
        raise ValueError("Expected finite (..., 6) SE(3) corrections")
    w, v = delta[..., :3], delta[..., 3:]
    skew = np.zeros(delta.shape[:-1] + (3, 3))
    skew[..., 0, 1], skew[..., 0, 2] = -w[..., 2], w[..., 1]
    skew[..., 1, 0], skew[..., 1, 2] = w[..., 2], -w[..., 0]
    skew[..., 2, 0], skew[..., 2, 1] = -w[..., 1], w[..., 0]
    theta2 = np.sum(w*w, axis=-1)
    safe2 = np.maximum(theta2, 1e-12)
    theta = np.sqrt(safe2)
    small = theta2 < 1e-4
    a = np.where(small, 1-theta2/6+theta2**2/120, np.sin(theta)/theta)
    b = np.where(small, .5-theta2/24+theta2**2/720, (1-np.cos(theta))/safe2)
    c = np.where(small, 1/6-theta2/120+theta2**2/5040, (theta-np.sin(theta))/(safe2*theta))
    eye = np.broadcast_to(np.eye(3), skew.shape)
    square = skew @ skew
    rotation = eye + a[..., None, None]*skew + b[..., None, None]*square
    jacobian = eye + b[..., None, None]*skew + c[..., None, None]*square
    result = np.zeros(delta.shape[:-1] + (4, 4))
    result[..., :3, :3] = rotation
    result[..., :3, 3] = (jacobian @ v[..., None])[..., 0]
    result[..., 3, 3] = 1.
    return result


def apply_correction(c2w, delta, perturbation_frame="world"):
    return compose_correction(np.asarray(c2w), se3_exp(delta), perturbation_frame)


def compose_correction(c2w, correction, perturbation_frame="world"):
    """Shared multiplication direction, also usable with differentiable tensors."""
    if perturbation_frame == "world":
        return correction @ c2w
    if perturbation_frame == "camera":
        return c2w @ correction
    raise ValueError("perturbation_frame must be world or camera")


def project_corrections(deltas, scene_scale, limits):
    """Project total original-pose twists onto separate rotation/translation balls."""
    result = np.array(deltas, dtype=np.float64, copy=True)
    if not np.isfinite(result).all() or not np.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("Corrections and scene scale must be finite; scale must be positive")
    hits = {}
    for name, part, maximum in [("rotation", slice(0, 3), np.deg2rad(limits["max_rotation_deg"])),
                                ("translation", slice(3, 6), scene_scale*limits["max_translation_ratio"])]:
        norm = np.linalg.norm(result[..., part], axis=-1)
        hits[name] = norm > maximum
        factor = np.divide(np.nextafter(maximum, 0.), norm, out=np.ones_like(norm), where=hits[name])
        result[..., part] *= factor[..., None]
    return result, hits


def se3_exp_torch(delta):
    """Differentiable exp map with finite gradients at exactly zero rotation."""
    import torch
    w, v = delta[..., :3], delta[..., 3:]
    x, y, z = w.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(delta.shape[:-1] + (3, 3))
    theta2 = (w*w).sum(-1)
    safe2 = theta2.clamp_min(1e-12)
    theta = safe2.sqrt()
    # A 0.01-rad cutoff avoids float32 cancellation in 1-cos(theta)
    # and theta-sin(theta) throughout the small gradient-update regime.
    small = theta2 < 1e-4
    a = torch.where(small, 1-theta2/6+theta2.square()/120, theta.sin()/theta)
    b = torch.where(small, .5-theta2/24+theta2.square()/720, (1-theta.cos())/safe2)
    c = torch.where(small, 1/6-theta2/120+theta2.square()/5040, (theta-theta.sin())/(safe2*theta))
    eye = torch.eye(3, dtype=delta.dtype, device=delta.device).expand(skew.shape)
    square = skew @ skew
    rotation = eye + a[..., None, None]*skew + b[..., None, None]*square
    jacobian = eye + b[..., None, None]*skew + c[..., None, None]*square
    top = torch.cat((rotation, (jacobian @ v[..., None])), -1)
    bottom = delta.new_tensor([0., 0., 0., 1.]).expand(delta.shape[:-1] + (1, 4))
    return torch.cat((top, bottom), -2)


def transform_points(points, similarity):
    """VGGT world -> gsplat normalized world, including its scene scale."""
    return np.asarray(points) @ similarity[:3, :3].T + similarity[:3, 3]

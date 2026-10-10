"""Evaluation-only camera metrics. Never imported by scoring or optimization."""
import numpy as np
from benchmark.evaluation.pose_alignment import estimate_similarity, transform_c2w


def rotation_errors_deg(predicted, reference):
    relative = np.swapaxes(reference[..., :3, :3], -1, -2) @ predicted[..., :3, :3]
    # Polar projection removes float32 export roundoff without biasing tiny angles.
    U, _, Vh = np.linalg.svd(relative)
    signs = np.ones(relative.shape[:-2]+(3,))
    signs[..., -1] = np.linalg.det(U @ Vh)
    relative = (U*signs[..., None, :]) @ Vh
    skew = np.stack((relative[..., 2, 1]-relative[..., 1, 2], relative[..., 0, 2]-relative[..., 2, 0],
                     relative[..., 1, 0]-relative[..., 0, 1]), -1)/2
    sine = np.linalg.norm(skew, axis=-1)
    cosine = (np.trace(relative, axis1=-2, axis2=-1)-1)/2
    return np.rad2deg(np.arctan2(sine, np.clip(cosine, -1, 1)))


def pose_metrics(predicted, reference, align=True):
    predicted, reference = np.asarray(predicted, float), np.asarray(reference, float)
    if predicted.shape != reference.shape or predicted.ndim != 3 or predicted.shape[1:] != (4, 4):
        raise ValueError("Pose evaluation needs matched Nx4x4 c2w arrays")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("Pose evaluation requires finite poses")
    s, R, t = 1., np.eye(3), np.zeros(3)
    if align:
        s, R, t, _ = estimate_similarity(predicted[:, :3, 3], reference[:, :3, 3], with_scale=True)
    aligned = transform_c2w(predicted, s, R, t)
    angles = rotation_errors_deg(aligned, reference)
    centers = np.linalg.norm(aligned[:, :3, 3]-reference[:, :3, 3], axis=-1)
    scene_scale = float(np.linalg.norm(reference[:, :3, 3]-reference[:, :3, 3].mean(0), axis=-1).max())
    if scene_scale <= 0:
        raise ValueError("Reference scene has no camera-center extent")
    rmse = float(np.sqrt(np.mean(centers**2)))
    return dict(rotation_mean_deg=float(angles.mean()), rotation_median_deg=float(np.median(angles)),
        center_rmse=rmse, center_median=float(np.median(centers)), center_rmse_normalized=rmse/scene_scale,
        center_median_normalized=float(np.median(centers))/scene_scale, reference_scene_scale=scene_scale,
        rotation_errors_deg=angles.tolist(), center_errors=centers.tolist(),
        alignment=dict(direction="predicted_to_reference", scale=s, rotation=R.tolist(), translation=t.tolist()))


def evaluate_train_reference(names, predicted, metadata):
    """Use original COLMAP TRAIN poses only, after saving optimization results."""
    split = metadata["split"]
    if set(names) != set(split["train"]) or set(names) & set(split["test"]):
        raise ValueError("Reference pose evaluation is train-only")
    cameras = metadata.get("cameras", {})
    if any(name not in cameras or "c2w" not in cameras[name] for name in names):
        return dict(available=False, reason="Original COLMAP train poses unavailable")
    reference = np.asarray([cameras[name]["c2w"] for name in names])
    result = pose_metrics(predicted, reference)
    result.update(available=True, fit_names=list(names), source="original COLMAP train cameras from prepared metadata")
    return result

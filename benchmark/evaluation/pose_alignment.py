"""Fit reference -> reconstruction Sim(3) using TRAIN camera centers only."""
import numpy as np


def estimate_similarity(source, target, with_scale=True):
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Need at least three matched 3D camera centers")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Camera centers must be finite")
    x, y = source-source.mean(0), target-target.mean(0)
    if min(np.linalg.matrix_rank(x), np.linalg.matrix_rank(y)) < 2:
        raise ValueError("Collinear/coincident centers cannot determine a unique rotation")
    U, singular, Vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(U @ Vt)
    rotation = U @ np.diag(sign) @ Vt
    scale = float((singular * sign).sum() / (x*x).sum() * len(x)) if with_scale else 1.
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Invalid similarity scale")
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    residual = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
    return scale, rotation, translation, residual


def transform_c2w(c2w, scale, rotation, translation):
    result = np.array(c2w, dtype=float, copy=True)
    result[..., :3, :3] = rotation @ result[..., :3, :3]
    result[..., :3, 3] = scale * (result[..., :3, 3] @ rotation.T) + translation
    return result


def align_targets(metadata, predicted_train, mode="sim3_train_centers"):
    train, test = metadata["split"]["train"], metadata["split"]["test"]
    if set(predicted_train) != set(train) or set(train) & set(test):
        raise ValueError("Alignment must contain exactly the 32 training cameras")
    ref = np.stack([metadata["cameras"][n]["c2w"] for n in train])
    pred = np.stack([predicted_train[n] for n in train])
    if mode in ("sim3_train_centers", "se3_train_centers"):
        s, R, t, residual = estimate_similarity(ref[:, :3, 3], pred[:, :3, 3], mode.startswith("sim3"))
    elif mode == "none":
        s, R, t = 1., np.eye(3), np.zeros(3)
        residual = np.linalg.norm(ref[:, :3, 3]-pred[:, :3, 3], axis=1)
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")
    targets = {n: transform_c2w(metadata["cameras"][n]["c2w"], s, R, t) for n in test}
    report = {"mode": mode, "direction": "reference_to_reconstruction", "fit_names": train,
              "scale": s, "rotation": R.tolist(), "translation": t.tolist(),
              "train_center_rmse": float(np.sqrt(np.mean(residual**2))),
              "train_center_residuals": dict(zip(train, residual.tolist()))}
    return targets, report

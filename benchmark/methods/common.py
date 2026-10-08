from pathlib import Path
import numpy as np
from PIL import Image
from benchmark.evaluation.pose_alignment import align_targets
from benchmark.utils.io import write_json


def target_cameras(job, predicted_train):
    targets, report = align_targets(job["metadata"], predicted_train, job["alignment"]["mode"])
    out = Path(job["run_dir"])
    write_json(out / "alignment.json", report)
    write_json(out / "predicted_train_cameras.json", {n: np.asarray(p).tolist() for n, p in predicted_train.items()})
    write_json(out / "target_cameras.json", {n: p.tolist() for n, p in targets.items()})
    return targets


def save_rgb(run_dir, name, image):
    folder = Path(run_dir) / "renders"
    folder.mkdir(exist_ok=True)
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite render: {name}")
    Image.fromarray(np.rint(np.clip(image, 0, 1)*255).astype(np.uint8), "RGB").save(folder / f"{name}.png")

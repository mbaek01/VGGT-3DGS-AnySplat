from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
from PIL import Image
from benchmark.utils.io import read_json, write_json, sha256
from benchmark.utils.cameras import read_colmap


def load_split(path):
    data = read_json(path)
    first, last = data["range"]
    a, b = re.fullmatch(r"DSCF(\d{4})", first), re.fullmatch(r"DSCF(\d{4})", last)
    if not a or not b:
        raise ValueError("Expected DSCF#### endpoints")
    names = [f"DSCF{i:04d}" for i in range(int(a[1]), int(b[1])+1)]
    test = data["test"]
    train = [x for x in names if x not in test]
    if len(names) != 36 or len(train) != 32 or len(set(test)) != 4 or not set(test) <= set(names):
        raise ValueError("Assignment requires exactly 32 train and 4 test images")
    return {"scene": data["scene"], "train": train, "test": sorted(test)}


def resize_crop(image, K, width, height):
    """Resize to cover and integer center crop, with exactly the same K transform."""
    w, h = image.size
    scale = max(width / w, height / h)
    nw, nh = max(width, int(w*scale)), max(height, int(h*scale))
    left, top = (nw-width)//2, (nh-height)//2
    image = image.resize((nw, nh), Image.Resampling.BICUBIC).crop((left, top, left+width, top+height))
    A = np.array([[nw/w, 0, -left], [0, nh/h, -top], [0, 0, 1.]])
    return image, A @ K, A


def prepare(scene_dir, scene_cfg, split_file, destination, preprocessing):
    split = load_split(split_file)
    scene_dir, destination = Path(scene_dir), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Prepared directory exists: {destination}; select a new prepared_root or remove it explicitly")
    if preprocessing["policy"] != "resize_short_edge_center_crop" or preprocessing["interpolation"] != "bicubic":
        raise ValueError("Only resize_short_edge_center_crop with bicubic is implemented")
    width, height = preprocessing["width"], preprocessing["height"]
    if min(width, height) < 32:
        raise ValueError("Evaluation dimensions must be >= 32")
    available = {}
    for p in (scene_dir / scene_cfg["image_dir"]).iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if p.stem in available:
                raise ValueError(f"Ambiguous image stem: {p.stem}")
            available[p.stem] = p
    cameras = read_colmap(scene_dir / scene_cfg["colmap_dir"])
    missing = set(split["train"] + split["test"]) - (available.keys() & cameras.keys())
    if missing:
        raise ValueError(f"Missing images or reference cameras: {sorted(missing)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
    metadata = {"scene": split["scene"], "split": split, "preprocessing": preprocessing, "cameras": {}}
    provenance = {"source": str(scene_dir.resolve()), "images": {}, "colmap": {}}
    try:
        for part in ("train", "test"):
            folder = temp / ("train/images" if part == "train" else "gt")
            folder.mkdir(parents=True)
            for stem in split[part]:
                cam, source = cameras[stem], available[stem]
                with Image.open(source) as raw:
                    # COLMAP uses stored pixel coordinates, never auto-rotate by EXIF.
                    image = raw.convert("RGB")
                    iw, ih = image.size
                    K = cam["K"].copy()
                    K[0] *= iw / cam["width"]
                    K[1] *= ih / cam["height"]
                    image, K, A = resize_crop(image, K, width, height)
                    image.save(folder / f"{stem}.png")
                metadata["cameras"][stem] = {"c2w": cam["c2w"].tolist(), "K": K.tolist(), "width": width, "height": height}
                provenance["images"][stem] = {"source_sha256": sha256(source), "prepared_sha256": sha256(folder / f"{stem}.png"), "pixel_transform": A.tolist()}
        for file in ("cameras.bin", "images.bin"):
            provenance["colmap"][file] = sha256(scene_dir / scene_cfg["colmap_dir"] / file)
        write_json(temp / "cameras.json", metadata)
        write_json(temp / "provenance.json", provenance)
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return destination

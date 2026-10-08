"""Minimal COLMAP binary reader. Camera-to-world matrices use OpenCV axes.

No point tracks or sparse points from the reference dataset are used.
Distorted cameras must be undistorted offline before preparation.
"""
import struct
from pathlib import Path
import numpy as np


def unpack(f, fmt):
    fmt = "<" + fmt
    n = struct.calcsize(fmt)
    data = f.read(n)
    if len(data) != n:
        raise ValueError("Truncated COLMAP binary file")
    return struct.unpack(fmt, data)


def quaternion_rotation(q):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid COLMAP quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def read_colmap(directory):
    directory = Path(directory)
    cameras = {}
    with (directory / "cameras.bin").open("rb") as f:
        for _ in range(unpack(f, "Q")[0]):
            cid, model, width, height = unpack(f, "iiQQ")
            if model not in (0, 1):
                raise ValueError(f"Camera model {model}: use undistorted PINHOLE/SIMPLE_PINHOLE data")
            params = unpack(f, "d" * (3 if model == 0 else 4))
            fx, fy, cx, cy = (params[0], params[0], *params[1:]) if model == 0 else params
            cameras[cid] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), width, height)
    images = {}
    with (directory / "images.bin").open("rb") as f:
        for _ in range(unpack(f, "Q")[0]):
            values = unpack(f, "i" + "d"*7 + "i")
            name = bytearray()
            while True:
                c = f.read(1)
                if c == b"\0":
                    break
                if not c:
                    raise ValueError("Truncated COLMAP image name")
                name.extend(c)
            name = name.decode("utf-8")
            n = unpack(f, "Q")[0]
            f.seek(n * 24, 1)  # x, y, point3D_id; ignored by design
            w2c = np.eye(4)
            w2c[:3, :3] = quaternion_rotation(values[1:5])
            w2c[:3, 3] = values[5:8]
            K, width, height = cameras[values[8]]
            stem = Path(name).stem
            if stem in images:
                raise ValueError(f"Duplicate COLMAP stem: {stem}")
            images[stem] = dict(name=name, c2w=np.linalg.inv(w2c), K=K.copy(), width=width, height=height)
    return images

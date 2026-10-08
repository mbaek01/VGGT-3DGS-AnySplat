#!/usr/bin/env bash
# Download the official Mip-NeRF 360 archive and extract benchmark inputs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/downloads
archive=data/downloads/360_v2.zip
curl --fail --location --retry 3 --continue-at - \
  --output "$archive" https://storage.googleapis.com/gresearch/refraw360/360_v2.zip
python3 - "$archive" <<'PY'
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

root = Path('data/mipnerf360')
with zipfile.ZipFile(sys.argv[1]) as archive:
    selected = []
    for entry in archive.infolist():
        parts = PurePosixPath(entry.filename).parts
        if entry.is_dir() or '..' in parts:
            continue
        for scene in ('kitchen', 'room'):
            if scene not in parts:
                continue
            relative = parts[parts.index(scene):]
            if (len(relative) == 3 and relative[1] == 'images') or (
                relative[1:] in (('sparse', '0', 'cameras.bin'), ('sparse', '0', 'images.bin'))
            ):
                selected.append((entry, root.joinpath(*relative)))
    destinations = {destination for _, destination in selected}
    for scene in ('kitchen', 'room'):
        for name in ('cameras.bin', 'images.bin'):
            if root / scene / 'sparse' / '0' / name not in destinations:
                raise RuntimeError(f'Archive is missing {scene}/sparse/0/{name}')
        if not any(path.parent == root / scene / 'images' for path in destinations):
            raise RuntimeError(f'Archive is missing {scene}/images')
    for entry, destination in selected:
        if destination.exists():
            print(f'Already exists, keeping: {destination}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.part')
        with archive.open(entry) as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target)
        temporary.replace(destination)
print(f'Dataset ready: {root.resolve()}')
PY

"""Train-only SuperPoint/LightGlue with pose-independent MAGSAC verification."""
import argparse
import importlib.metadata
from pathlib import Path
import shutil
import time
import numpy as np
from benchmark.utils.io import ROOT, read_json, write_json, sha256


def train_paths(train_dir, split):
    names, test = split['train'], split['test']
    if len(names) != 32 or len(set(names)) != 32 or set(names)&set(test):
        raise ValueError('Matcher requires exactly 32 disjoint training images')
    paths = sorted((Path(train_dir)/'images').iterdir())
    if len(paths) != 32 or {p.stem for p in paths} != set(names) or any(p.suffix != '.png' for p in paths):
        raise ValueError('Matcher directory must contain exactly the training PNGs')
    return paths


def neighbor_pairs(count, window, maximum):
    if window < 1 or maximum < 1: raise ValueError('Invalid pair graph settings')
    degree, pairs = np.zeros(count, int), []
    # Prioritize short edges to retain a connected chain even under a degree cap.
    for distance in range(1, window+1):
        for i in range(count-distance):
            j = i+distance
            if degree[i] < maximum and degree[j] < maximum:
                pairs.append((i, j)); degree[i] += 1; degree[j] += 1
    return sorted(pairs)


def coverage(xy, size):
    import cv2
    return float(cv2.contourArea(cv2.convexHull(np.asarray(xy, np.float32)))/(size[0]*size[1])) if len(xy) >= 3 else 0.


def verify_pair(xy_i, xy_j, confidence, sizes, cfg, pair_index):
    import cv2
    xy_i, xy_j, confidence = np.asarray(xy_i, float), np.asarray(xy_j, float), np.asarray(confidence, float)
    if xy_i.shape != xy_j.shape or xy_i.shape != (len(confidence), 2): raise ValueError('Invalid match arrays')
    if not all(np.isfinite(a).all() for a in (xy_i, xy_j, confidence)): raise ValueError('Nonfinite matcher output')
    if np.any((confidence < 0)|(confidence > 1)): raise ValueError('Confidence outside [0,1]')
    for xy, size in zip((xy_i, xy_j), sizes):
        if np.any((xy < 0)|(xy >= np.asarray(size))): raise ValueError('Matcher coordinates outside training pixels')
    F, mask = None, np.zeros(len(confidence), bool)
    cv2.setNumThreads(1); cv2.setRNGSeed(cfg['ransac_seed']+pair_index)
    if len(confidence) >= 8:
        F, inliers = cv2.findFundamentalMat(xy_i, xy_j, cv2.USAC_MAGSAC,
            cfg['ransac_threshold_px'], cfg['ransac_confidence'], cfg['ransac_max_iterations'])
        if F is not None and F.shape == (3, 3) and np.isfinite(F).all() and inliers is not None:
            mask = inliers[:, 0].astype(bool)
        else: F = None
    count = int(mask.sum()); ratio = count/max(len(confidence), 1)
    cov_i, cov_j = coverage(xy_i[mask], sizes[0]), coverage(xy_j[mask], sizes[1])
    cov = min(cov_i, cov_j); mean_conf = float(confidence[mask].mean()) if count else 0.
    count_factor = min(count/cfg['reliability_saturation_inliers'], 1.)
    weight = count_factor*ratio*np.sqrt(cov)*mean_conf
    failures = []
    for label, value, threshold in [('inliers', count, cfg['min_inliers']),
        ('inlier_ratio', ratio, cfg['min_inlier_ratio']), ('coverage', cov, cfg['min_coverage']),
        ('confidence', mean_conf, cfg['min_mean_confidence'])]:
        if value < threshold: failures.append(label)
    if F is None: failures.append('ransac_failed')
    accepted = bool(not failures and weight > 0)
    row = dict(raw_count=len(confidence), inlier_count=count, inlier_ratio=ratio,
        coverage_i=cov_i, coverage_j=cov_j, coverage=cov, mean_inlier_confidence=mean_conf,
        mean_raw_confidence=float(confidence.mean()) if len(confidence) else 0.,
        count_factor=count_factor, coverage_factor=float(np.sqrt(cov)), reliability=float(weight),
        pair_weight=float(weight) if accepted else 0., accepted=accepted, rejected_reasons=failures)
    return mask, F, row


def load_matches(cache, expected_manifest=None):
    cache = Path(cache); manifest = read_json(cache/'manifest.json')
    if expected_manifest is not None and manifest['inputs'] != expected_manifest:
        raise ValueError('Cached matches have different training RGB/settings/model/code provenance')
    if sha256(cache/'pairs.json') != manifest['pairs_sha256']: raise ValueError('Pair diagnostics cache changed')
    diagnostics = read_json(cache/'pairs.json'); matches = []
    for row in diagnostics:
        path = cache/row['file']
        if sha256(path) != manifest['files'][row['file']]: raise ValueError('Correspondence cache changed')
        with np.load(path, allow_pickle=False) as data:
            if row['accepted']:
                mask = data['inlier_mask']
                matches.append(dict(indices=row['indices'], names=row['names'], weight=row['pair_weight'],
                    xy_i=data['xy_i'][mask].copy(), xy_j=data['xy_j'][mask].copy(),
                    confidence=data['confidence'][mask].copy()))
    return matches, diagnostics, manifest


def cached_matches(train_dir, split, cache, cfg, model_provenance, factory):
    paths = train_paths(train_dir, split)
    inputs = dict(names=[p.stem for p in paths], rgb_sha256={p.stem:sha256(p) for p in paths},
        settings=cfg, models=model_provenance, matching_code_sha256=sha256(Path(__file__)))
    cache = Path(cache)
    if (cache/'manifest.json').exists(): return load_matches(cache, inputs)
    if cache.exists() and any(cache.iterdir()): raise ValueError('Incomplete matching cache; choose a new directory')
    cache.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    sizes = []
    for path in paths:
        with Image.open(path) as image: sizes.append(image.size)
    network = factory(paths)
    rows, files = [], {}
    start = time.perf_counter()
    for k, (i, j) in enumerate(neighbor_pairs(32, cfg['neighbor_window'], cfg['max_pairs_per_image'])):
        xy_i, xy_j, confidence = network(i, j)
        mask, F, row = verify_pair(xy_i, xy_j, confidence, (sizes[i], sizes[j]), cfg, k)
        filename = f'pair_{i:02d}_{j:02d}.npz'
        np.savez_compressed(cache/filename, xy_i=xy_i, xy_j=xy_j, confidence=confidence,
            inlier_mask=mask, ransac_F=np.empty((0, 0)) if F is None else F)
        row.update(indices=[i, j], names=[paths[i].stem, paths[j].stem], file=filename)
        rows.append(row); files[filename] = sha256(cache/filename)
        print(f'match {k+1}/90 {row["names"]}: {row["raw_count"]} -> {row["inlier_count"]}, '
            f'weight={row["pair_weight"]:.4f}', flush=True)
    write_json(cache/'pairs.json', rows)
    write_json(cache/'manifest.json', dict(inputs=inputs, files=files,
        pairs_sha256=sha256(cache/'pairs.json'), matching_seconds=time.perf_counter()-start,
        geometry_verification='OpenCV USAC_MAGSAC fundamental matrix, RGB only',
        train_only=True, reference_poses_used=False, vggt_tracks_or_xyz_used=False))
    visualize_pairs(cache, paths, rows)
    return load_matches(cache, inputs)


def visualize_pairs(cache, paths, rows, count=3):
    from PIL import Image, ImageDraw
    ranked = sorted(rows, key=lambda r:r['reliability'])
    selected = {r['file']:r for r in ranked[:count]+ranked[-count:]}
    for row in selected.values():
        i, j = row['indices']
        with Image.open(paths[i]) as a, Image.open(paths[j]) as b:
            canvas = Image.new('RGB', (a.width+b.width, max(a.height, b.height)+48), '#111111')
            canvas.paste(a, (0, 48)); canvas.paste(b, (a.width, 48)); offset = a.width
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), f'{row["names"]} weight={row["pair_weight"]:.4f} raw={row["raw_count"]} '
            f'inlier={row["inlier_count"]} accepted={row["accepted"]}', fill='white')
        with np.load(Path(cache)/row['file'], allow_pickle=False) as data:
            ids = np.linspace(0, len(data['confidence'])-1, min(80, len(data['confidence'])), dtype=int)
            for k in ids:
                x, y = data['xy_i'][k]; u, v = data['xy_j'][k]
                color = '#66ee88' if data['inlier_mask'][k] else '#ff5555'
                draw.line((x, y+48, u+offset, v+48), fill=color, width=1)
        destination = Path(cache)/'visualizations'; destination.mkdir(exist_ok=True)
        canvas.save(destination/(Path(row['file']).stem+'.png'))


def prepare_models(cfg):
    import torch
    import lightglue
    from lightglue import LightGlue
    from urllib.parse import urlparse
    distribution = importlib.metadata.distribution('lightglue')
    direct = __import__('json').loads(distribution.read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != cfg['lightglue_commit']:
        raise ValueError('LightGlue revision differs from the declared dependency')
    if LightGlue.version != 'v0.1_arxiv': raise ValueError('Unexpected pretrained LightGlue release')
    hub = ROOT/cfg['model_cache']; torch.hub.set_dir(str(hub))
    checkpoints = hub/'checkpoints'; checkpoints.mkdir(parents=True, exist_ok=True)
    urls = [f'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/{name}' for name in
        ('superpoint_v1.pth', 'superpoint_lightglue.pth')]
    files = {}
    for url in urls:
        filename = Path(urlparse(url).path).name
        if filename == 'superpoint_lightglue.pth': filename = 'superpoint_lightglue_v0-1_arxiv.pth'
        destination = checkpoints/filename
        existing = Path.home()/'.cache/torch/hub/checkpoints'/filename
        if not destination.exists():
            if existing.exists(): shutil.copy2(existing, destination)
            else:
                temporary = destination.with_suffix('.download')
                torch.hub.download_url_to_file(url, str(temporary), progress=True)
                temporary.replace(destination)
        files[filename] = dict(url=url, sha256=sha256(destination))
    package = Path(lightglue.__file__).parent
    return dict(distribution=direct, weights=files,
        code_sha256={p.name:sha256(p) for p in sorted(package.glob('*.py'))},
        torch=torch.__version__, opencv=__import__('cv2').__version__)


def network_factory(cfg, device):
    def factory(paths):
        import torch
        from lightglue import SuperPoint, LightGlue
        from lightglue.utils import load_image
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        extractor = SuperPoint(max_num_keypoints=cfg['max_keypoints']).eval().to(device)
        matcher = LightGlue(features='superpoint', filter_threshold=cfg['filter_threshold'],
            depth_confidence=cfg['depth_confidence'], width_confidence=cfg['width_confidence'],
            flash=cfg['flash']).eval().to(device)
        with torch.inference_mode():
            features = [extractor.extract(load_image(p).to(device), resize=None) for p in paths]
        def match(i, j):
            with torch.inference_mode():
                result = matcher(dict(image0=features[i], image1=features[j]))
                ids, confidence = result['matches'][0], result['scores'][0]
                xy_i = features[i]['keypoints'][0, ids[:, 0]].cpu().numpy().copy()
                xy_j = features[j]['keypoints'][0, ids[:, 1]].cpu().numpy().copy()
                return xy_i, xy_j, confidence.cpu().numpy().copy()
        return match
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args(); job = read_json(args.job)
    # Validate all image inputs BEFORE loading any pretrained network.
    train_paths(job['train_dir'], job['split'])
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('RGB matching requires NVIDIA CUDA driver access')
    cfg = job['rgb_matching']; provenance = prepare_models(cfg)
    matches, rows, manifest = cached_matches(job['train_dir'], job['split'], job['cache_dir'], cfg,
        provenance, network_factory(cfg, 'cuda'))
    print(f'RGB pairs: {len(matches)}/{len(rows)} accepted; train-only cache complete', flush=True)


if __name__ == '__main__': main()

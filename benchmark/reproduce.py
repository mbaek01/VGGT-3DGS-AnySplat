"""Reproduce final RGB Pose+Focal from a VGGT baseline export.

Uses the original matcher, optimizer, native GS runner and reference-K evaluator.
No MPR, Oracle, Shared-K or historical experiment directory is required.
"""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import time
from benchmark.utils.io import ROOT, read_json, read_yaml, write_json, root_path, sha256


def validate_source(source, prepared, cfg):
    from benchmark.datasets.mipnerf360 import load_split
    from benchmark.rgb_pose_refinement.matching import train_paths
    job = read_json(source / 'job.json')
    if job['method'] != 'vggt_3dgs' or read_json(source / 'status.json')['status'] != 'complete':
        raise ValueError('Use a completed VGGT+3DGS baseline as the source')
    if job['method_config']['seed'] != 42 or job['method_config']['vggt']['use_ba']:
        raise ValueError('Final Ours requires seed42 and the VGGT export without BA')
    meta = read_json(prepared / 'cameras.json')
    expected = load_split(ROOT / 'splits' / f"{meta['scene']}.json")
    if meta != job['metadata'] or meta['split'] != expected or meta['preprocessing'] != cfg['preprocessing']:
        raise ValueError('Source/prepared assignment split or preprocessing differs')
    if job['alignment'] != {'mode': 'sim3_train_centers'}:
        raise ValueError('Final primary requires the original train-center Sim(3)')
    provenance = read_json(prepared / 'provenance.json')
    if read_json(source / 'dataset_provenance.json') != provenance:
        raise ValueError('Source image provenance differs')
    paths = train_paths(source / 'train', expected)
    for path in paths:
        if sha256(path) != provenance['images'][path.stem]['prepared_sha256']:
            raise ValueError(f'Training RGB changed: {path.stem}')
    return job


def native_inputs(source):
    """Same Parser construction as the unchanged baseline Runner; no training."""
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT / 'third_party/gsplat/examples'))
    from datasets.colmap import Parser
    parser = Parser(data_dir=str(source / 'train'), factor=1, normalize=True, test_every=8)
    names = [Path(n).stem for n in parser.image_names]
    K = np.stack([parser.Ks_dict[cid] for cid in parser.camera_ids])
    poses = parser.camtoworlds.copy()
    stored = read_json(source / 'predicted_train_cameras.json')
    np.testing.assert_array_equal(poses, np.array([stored[n] for n in names]))
    return names, poses, K, float(parser.scene_scale)


def validate_cache(cache, names, train_dir, matching_settings):
    from benchmark.rgb_pose_refinement.matching import load_matches
    matches, _, manifest = load_matches(cache)
    inputs = manifest['inputs']
    if inputs['names'] != names or inputs['settings'] != matching_settings:
        raise ValueError('Match cache camera ordering or settings differ')
    from benchmark.rgb_pose_refinement import matching
    if inputs['matching_code_sha256'] != sha256(matching.__file__):
        raise ValueError('Match cache was produced by different matching code')
    for name in names:
        if inputs['rgb_sha256'][name] != sha256(train_dir / 'images' / f'{name}.png'):
            raise ValueError(f'Match-cache training image differs: {name}')
    if not manifest.get('train_only') or manifest.get('reference_poses_used') or manifest.get('vggt_tracks_or_xyz_used'):
        raise ValueError('Only independent train-RGB correspondence caches are allowed')
    return matches


def validate_refinement(path, names, initial, K, scene_scale):
    import numpy as np
    from benchmark.pose_refinement.se3 import apply_correction
    from benchmark.intrinsic_refinement.optimizer import scaled_intrinsics
    with np.load(path, allow_pickle=False) as saved:
        if saved['names'].tolist() != names or float(saved['scene_scale']) != scene_scale:
            raise ValueError('Refinement camera ordering or normalization differs')
        np.testing.assert_array_equal(saved['initial_c2w'], initial)
        np.testing.assert_array_equal(saved['initial_K'], K)
        np.testing.assert_allclose(saved['refined_c2w'], apply_correction(initial, saved['deltas'], 'camera'), atol=1e-12, rtol=0)
        np.testing.assert_array_equal(saved['refined_K'], scaled_intrinsics(K, float(saved['log_focal_scale'])))
        np.testing.assert_array_equal(saved['refined_c2w'][0], initial[0])
        np.testing.assert_array_equal(saved['refined_K'][:, :2, 2], K[:, :2, 2])


def refine_cameras(folder, source, source_job, cfg, method, cache=None, saved_refinement=None):
    import numpy as np
    from benchmark.run import execute_stage
    names, initial, K, scale = native_inputs(source)
    if saved_refinement:
        saved_refinement = saved_refinement.resolve()
        expected_settings = dict(pose=method['pose_refinement'], intrinsic=method['intrinsic_refinement'], mode='pose_focal')
        if read_json(saved_refinement.parent / 'settings.json') != expected_settings:
            raise ValueError('Saved refinement is not the final staged configuration')
        validate_refinement(saved_refinement, names, initial, K, scale)
        return saved_refinement
    if cache is None:
        cache = folder / 'rgb_matches'
        match_job = dict(train_dir=str(source / 'train'), split=source_job['metadata']['split'],
                         cache_dir=str(cache), rgb_matching=method['rgb_matching'])
        write_json(folder / 'matching_job.json', match_job)
        execute_stage(cfg['environments']['vggt'] + ['-m', 'benchmark.rgb_pose_refinement.matching',
                      '--job', str(folder / 'matching_job.json')], folder / 'logs/matching.log', ROOT)
    matches = validate_cache(cache, names, source / 'train', method['rgb_matching'])
    from benchmark.intrinsic_refinement.optimizer import refine
    start = time.perf_counter()
    poses, refined_K, deltas, log = refine(initial, K, matches, scale,
        method['pose_refinement'], method['intrinsic_refinement'], mode='pose_focal', device='cpu')
    log['optimization_seconds'] = time.perf_counter() - start
    path = folder / 'refinement/result.npz'
    path.parent.mkdir()
    np.savez_compressed(path, names=names, initial_c2w=initial, initial_K=K,
        refined_c2w=poses, refined_K=refined_K, deltas=deltas,
        log_focal_scale=log['learned_log_focal_scale'], scene_scale=scale)
    write_json(path.parent / 'optimization.json', log)
    write_json(path.parent / 'settings.json', dict(pose=method['pose_refinement'],
        intrinsic=method['intrinsic_refinement'], mode='pose_focal'))
    validate_refinement(path, names, initial, K, scale)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True, help='Completed VGGT+3DGS run')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory, never overwrite an existing run')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/benchmark.yaml')
    parser.add_argument('--match-cache', type=Path, help='Existing independent RGB cache; otherwise compute matches')
    parser.add_argument('--refinement', type=Path, help='Reuse saved staged result.npz and settings.json')
    parser.add_argument('--render-checkpoint', type=Path, help='Saved final Ours Gaussian checkpoint; render without GS training')
    parser.add_argument('--stage', choices=['all', 'refine', 'train', 'evaluate'], default='all')
    parser.add_argument('--dry-run', action='store_true', help='Validate/print configuration without writes or numerical/native imports')
    args = parser.parse_args(argv)
    cfg = read_yaml(args.config)
    intrinsic = read_yaml(ROOT / 'configs/methods/rgb_pose_focal_refine.yaml')
    pose = read_yaml(root_path(intrinsic['pose_config']))
    method = dict(rgb_matching=pose['rgb_matching'], pose_refinement=pose['pose_refinement'],
                  intrinsic_refinement=intrinsic['intrinsic_refinement'])
    source, folder = args.source_run.resolve(), args.output_dir.resolve()
    if folder.exists() and args.stage in ('all', 'refine'):
        parser.error('Output directory exists; use a new directory or --stage train/evaluate')
    if args.match_cache and args.refinement:
        parser.error('Choose --match-cache or --refinement')
    if args.render_checkpoint and not args.refinement:
        parser.error('--render-checkpoint requires its saved --refinement')
    if args.dry_run:
        import json
        print(json.dumps(dict(source_run=str(source), output_dir=str(folder), method=method,
            gs_seed=42, gs_iterations='same as source baseline', test_K='prepared reference K, never focal-scaled',
            test_poses='original reference targets with original VGGT train-center Sim3'), indent=2))
        return
    from benchmark.run import check_pins, execute_stage
    check_pins()
    source_job = read_json(source / 'job.json')
    prepared = root_path(cfg['prepared_root']) / source_job['metadata']['scene']
    source_job = validate_source(source, prepared, cfg)
    if args.stage in ('all', 'refine'):
        folder.mkdir(parents=True, exist_ok=False)
        (folder / 'logs').mkdir()
        write_json(folder / 'resolved_config.json', cfg)
        write_json(folder / 'method_config.json', method)
        write_json(folder / 'status.json', dict(status='refining', started_utc=datetime.now(timezone.utc).isoformat()))
        path = refine_cameras(folder, source, source_job, cfg, method,
                              args.match_cache.resolve() if args.match_cache else None, args.refinement)
        job = copy.deepcopy(source_job)
        job.update(method='rgb_pose_focal', source_run=str(source), run_dir=str(folder),
            train_dir=str(source / 'train'), prepared_dir=str(prepared),
            validation=dict(variant='pose_focal', refinement=str(path), refinement_sha256=sha256(path)))
        if args.render_checkpoint:
            job['validation']['render_checkpoint'] = str(args.render_checkpoint.resolve())
            job['validation']['render_checkpoint_sha256'] = sha256(args.render_checkpoint)
        write_json(folder / 'job.json', job)
        write_json(folder / 'dataset_provenance.json', read_json(source / 'dataset_provenance.json'))
        write_json(folder / 'status.json', dict(status='refined'))
    job = read_json(folder / 'job.json')
    if job['source_run'] != str(source) or job['metadata'] != source_job['metadata']:
        raise ValueError('Resume source differs from the saved job')
    if args.stage in ('all', 'train'):
        if (folder / 'native/final.pt').exists():
            raise FileExistsError('Gaussian checkpoint exists; use --stage evaluate or a new directory')
        wall = execute_stage(cfg['environments']['gsplat'] + ['-m', 'benchmark.workers.pose_focal',
            '--job', str(folder / 'job.json')], folder / 'logs/gsplat.log', ROOT / 'third_party/gsplat')
        native = read_json(folder / 'native_timing.json')
        if job['validation'].get('render_checkpoint'):
            native['checkpoint_load_seconds'] = native.pop('optimization_seconds')
            native['optimization_seconds'] = 0.
        write_json(folder / 'timing.json', dict(native=native,
            stage_wall_seconds=dict(gsplat=wall), reused_vggt_inference=True,
            reconstruction_compute_seconds=native['optimization_seconds'], render_seconds=native['render_seconds'],
            definitions=dict(reconstruction_compute_seconds='Native GS loop only; reused VGGT export, matching/camera stages and startup excluded. See refinement/optimization.json for camera compute; not standalone end-to-end reconstruction time.'),
            gaussian_training_iterations=0 if job['validation'].get('render_checkpoint') else job['method_config']['gsplat']['iterations']))
        write_json(folder / 'status.json', dict(status='rendered'))
    if args.stage in ('all', 'evaluate'):
        execute_stage(['conda', 'run', '--no-capture-output', '-n', 'dtb-eval', 'python',
            '-m', 'benchmark.run', '--config', str(args.config.resolve()), 'evaluate',
            '--run-dir', str(folder), '--prepared-dir', str(prepared)], folder / 'logs/evaluation.log', ROOT)
        write_json(folder / 'status.json', dict(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()))
    print(folder)


if __name__ == '__main__':
    main()

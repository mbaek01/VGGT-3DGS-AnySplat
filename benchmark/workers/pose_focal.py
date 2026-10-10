"""Final Pose+Focal: unchanged GS runner, saved train cameras, frozen test protocol."""
import argparse
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from benchmark.utils.io import ROOT, read_json, write_json, sha256
from benchmark.methods.vggt_3dgs import VGGT3DGS
from benchmark.utils.native import (read_native_config, compare_native_config,
    geometry_fingerprint, rng_fingerprint)
from benchmark.pose_refinement.se3 import apply_correction
from benchmark.intrinsic_refinement.optimizer import scaled_intrinsics


def install_cameras(runner, job):
    """Called after unchanged initialization, before the native training loop."""
    parser, folder = runner.parser, Path(job['run_dir'])
    source = Path(job['source_run']); variant = job['validation']['variant']
    if variant != 'pose_focal': raise ValueError('This worker reproduces final Pose+Focal only')
    names = [Path(n).stem for n in parser.image_names]
    if len(names) != 32 or set(names) != set(job['metadata']['split']['train']) or set(names)&set(job['metadata']['split']['test']):
        raise ValueError('Expected exactly the 32 training cameras')
    historical = read_native_config(source/'native/cfg.yml')
    comparison = compare_native_config(asdict(runner.cfg), historical)
    write_json(folder/'effective_native_config.json',asdict(runner.cfg))
    environment = read_json(source/'native_environment.json')
    for name in ('torch', 'torchvision', 'numpy', 'gsplat'):
        if importlib.metadata.version(name) != environment[name]: raise ValueError(f'Native dependency changed: {name}')
    original = parser.camtoworlds.copy()
    initial_K = np.stack([parser.Ks_dict[cid] for cid in parser.camera_ids])
    stored = read_json(source/'predicted_train_cameras.json')
    np.testing.assert_array_equal(original, np.array([stored[n] for n in names]))
    before, rng_before = geometry_fingerprint(parser, runner), rng_fingerprint()
    path = Path(job['validation']['refinement'])
    if sha256(path) != job['validation']['refinement_sha256']: raise ValueError('Refinement changed')
    with np.load(path, allow_pickle=False) as saved:
        assert names == saved['names'].tolist()
        np.testing.assert_array_equal(saved['initial_c2w'], original)
        np.testing.assert_array_equal(saved['initial_K'], initial_K)
        assert float(saved['scene_scale']) == parser.scene_scale
        poses, Ks = saved['refined_c2w'].copy(), saved['refined_K'].copy()
        delta, a = saved['deltas'], float(saved['log_focal_scale'])
        np.testing.assert_allclose(poses, apply_correction(original, delta, 'camera'), atol=1e-12, rtol=0)
        np.testing.assert_array_equal(Ks, scaled_intrinsics(initial_K, a))
        np.testing.assert_array_equal(poses[0], original[0])
        np.testing.assert_array_equal(Ks[:, :2, 2], initial_K[:, :2, 2])
        assert .9 <= np.exp(a) <= 1.1
    parser.camtoworlds = poses.astype(original.dtype)
    for i, cid in enumerate(parser.camera_ids): parser.Ks_dict[cid] = Ks[i].copy()
    after, rng_after = geometry_fingerprint(parser, runner), rng_fingerprint()
    allowed = {'intrinsics'}
    if {k:v for k,v in before.items() if k not in allowed} != {k:v for k,v in after.items() if k not in allowed}:
        raise ValueError('Camera installation changed non-camera initialization')
    if rng_before != rng_after: raise ValueError('Camera installation consumed random numbers')
    runner.original_prediction = dict(zip(names, original))
    np.savez_compressed(folder/'training_cameras.npz', names=names, initial_c2w=original,
        training_c2w=poses, initial_K=initial_K, training_K=Ks, parser_transform=parser.transform,
        scene_scale=parser.scene_scale)
    write_json(folder/'invariants.json', dict(passed=True, variant=variant, before=before, after=after,
        rng_unchanged=True, same_native_initialization_policy=True, native_config=comparison,
        unchanged_points_normalization_optimizers=True, training_reference_K_used=False,
        training_reference_pose_used=False, source_points_sha256=sha256(source/'train/sparse/points3D.bin')))


class FrozenCameraGS(VGGT3DGS):
    def reconstruct(self, job):
        sys.path.insert(0, str(ROOT/'third_party/gsplat/examples'))
        import simple_trainer
        NativeRunner = simple_trainer.Runner
        class Instrumented(NativeRunner):
            def train(self):
                install_cameras(self, job)
                checkpoint = job['validation'].get('render_checkpoint')
                if checkpoint:
                    import torch
                    if sha256(checkpoint) != job['validation']['render_checkpoint_sha256']:
                        raise ValueError('Render checkpoint changed')
                    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                    if state['completed_steps'] != self.cfg.max_steps: raise ValueError('Incomplete checkpoint')
                    self.splats = torch.nn.ParameterDict({k:torch.nn.Parameter(v.to(self.device)) for k,v in state['splats'].items()})
                else: super().train()
        with patch.object(simple_trainer, 'Runner', Instrumented): return super().reconstruct(job)

    def render_test_views(self, job, state):
        actual, original = state['predicted'], state['runner'].original_prediction
        timing = super().render_test_views(job, dict(state, predicted=original))
        folder, source = Path(job['run_dir']), Path(job['source_run'])
        write_json(folder/'predicted_train_cameras.json', {n:p.tolist() for n,p in actual.items()})
        write_json(folder/'alignment_fit_train_cameras.json', {n:p.tolist() for n,p in original.items()})
        if read_json(folder/'target_cameras.json') != read_json(source/'target_cameras.json'):
            raise ValueError('Evaluation camera poses changed')
        alignment = read_json(folder/'alignment.json')
        alignment['fit_pose_set'] = 'original VGGT train centers, frozen for all conditions and budgets'
        write_json(folder/'alignment.json', alignment)
        write_json(folder/'evaluation_cameras.json', dict(poses=read_json(folder/'target_cameras.json'),
            intrinsics={n:job['metadata']['cameras'][n]['K'] for n in job['metadata']['split']['test']},
            sizes={n:[job['metadata']['cameras'][n]['width'],job['metadata']['cameras'][n]['height']]
                for n in job['metadata']['split']['test']},
            calibration='prepared reference COLMAP test K, fixed and never focal-scaled',
            alignment='reference test poses mapped by original VGGT train-center Sim3',
            image_policy=job['metadata']['preprocessing']))
        return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Native GS rendering/training needs NVIDIA driver access')
    from benchmark.run import check_pins
    check_pins()
    job = read_json(args.job); method = FrozenCameraGS()
    state = method.reconstruct(job); timing = method.render_test_views(job, state)
    write_json(Path(job['run_dir'])/'native_timing.json', timing)
    write_json(Path(job['run_dir'])/'native_environment.json',
        {d.metadata['Name']:d.version for d in importlib.metadata.distributions()})


if __name__ == '__main__': main()

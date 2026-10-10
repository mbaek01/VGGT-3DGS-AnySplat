"""Native GS configuration and initialization checks used by final Ours.

These data-only helpers were extracted unchanged from the former diagnostics.
They do not read reference cameras or run any optimization.
"""
import hashlib
from pathlib import Path
import pickle
import random
import numpy as np


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_native_config(path):
    import yaml
    class ConfigLoader(yaml.SafeLoader): pass
    ConfigLoader.add_constructor('tag:yaml.org,2002:python/object:gsplat.strategy.default.DefaultStrategy',
        lambda loader, node: loader.construct_mapping(node, deep=True))
    return yaml.load(Path(path).read_text(), Loader=ConfigLoader)


def compare_native_config(current, historical, short_budget=False):
    ignored = {'result_dir'} | ({'max_steps', 'save_steps'} if short_budget else set())
    differences = {key:dict(current=current.get(key), historical=historical.get(key))
        for key in current.keys()|historical.keys() if key not in ignored and current.get(key) != historical.get(key)}
    if differences: raise ValueError(f'Native settings differ from source baseline: {differences}')
    return dict(equal=True, allowed_differences=sorted(ignored))


def geometry_fingerprint(parser, runner):
    return dict(points=array_hash(parser.points), points_rgb=array_hash(parser.points_rgb),
        transform=array_hash(parser.transform), intrinsics={str(k):array_hash(v) for k,v in parser.Ks_dict.items()},
        parser_scene_scale=float(parser.scene_scale), runner_scene_scale=float(runner.scene_scale),
        initial_splats={name:array_hash(value.detach().cpu().numpy()) for name,value in runner.splats.items()},
        optimizer_groups={name:[{k:v for k,v in group.items() if k != 'params'} for group in optimizer.param_groups]
                          for name,optimizer in runner.optimizers.items()})


def rng_fingerprint():
    import torch
    values = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().cpu().numpy(),
        cuda=[s.cpu().numpy() for s in torch.cuda.get_rng_state_all()])
    return hashlib.sha256(pickle.dumps(values)).hexdigest()

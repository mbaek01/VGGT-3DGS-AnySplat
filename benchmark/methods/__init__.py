def get_method(name):
    if name == "vggt_3dgs":
        from .vggt_3dgs import VGGT3DGS
        return VGGT3DGS()
    if name == "anysplat":
        from .anysplat import AnySplatBaseline
        return AnySplatBaseline()
    if name == "ours":
        from .ours import OursBaseline
        return OursBaseline()
    raise ValueError(f"Unknown method: {name}")

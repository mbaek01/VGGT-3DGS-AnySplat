from .base import Baseline


class OursBaseline(Baseline):
    def reconstruct(self, job):
        raise NotImplementedError(
            "'ours' is a reserved adapter: initialize with AnySplat, add VGGT train-only "
            "depth/pose refinement, then implement native rendering. No baseline is substituted."
        )

    def render_test_views(self, job, reconstruction):
        raise NotImplementedError("Implement native rendering for the refined representation")

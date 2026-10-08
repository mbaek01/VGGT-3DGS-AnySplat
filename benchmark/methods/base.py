from abc import ABC, abstractmethod


class Baseline(ABC):
    """Workers receive train-only RGB plus camera metadata, never test RGB.

    reconstruct returns native state and timings; render_test_views consumes it
    and writes four RGB PNGs. Heavy imports belong inside implementations.
    """
    @abstractmethod
    def reconstruct(self, job):
        raise NotImplementedError

    @abstractmethod
    def render_test_views(self, job, reconstruction):
        raise NotImplementedError

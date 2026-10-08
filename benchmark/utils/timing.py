from contextlib import contextmanager
from time import perf_counter


@contextmanager
def cuda_timer(record, key):
    import torch
    torch.cuda.synchronize()
    start = perf_counter()
    yield
    torch.cuda.synchronize()
    record[key] = perf_counter() - start


def hardware():
    import torch
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2}

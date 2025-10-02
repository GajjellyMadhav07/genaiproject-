import time
from contextlib import contextmanager
from typing import Dict, Any, Callable, Tuple


@contextmanager
def timer() -> Tuple[None, Callable[[], float]]:
    start = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - start) * 1000.0

    try:
        yield None, elapsed
    finally:
        pass


def with_timing(fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000.0
        return result, ms

    return wrapper



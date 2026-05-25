import time
from typing import Callable, TypeVar, Any, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


def timeit(func: Callable[P, T]) -> Callable[P, tuple[T, float]]:
    """Times the execution of a function.

    Args:
        func: The function to time.

    Returns:
        A wrapped function that, when called, returns a tuple containing:
        1. The result of the function call.
        2. The elapsed time in seconds as a float.
    """
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper

import unittest
import time
from yft_utils.timeit import timeit


class TimeitTest(unittest.TestCase):

    def test_timeit_returns_correct_result_and_positive_time(self) -> None:
        def sample_func(x: int, y: int) -> int:
            time.sleep(0.01)
            return x + y

        timed_func = timeit(sample_func)
        result, elapsed = timed_func(3, 5)

        self.assertEqual(result, 8)
        self.assertGreater(elapsed, 0.0)
        self.assertLess(elapsed, 1.0)

    def test_timeit_preserves_kwargs(self) -> None:
        def sample_func(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        timed_func = timeit(sample_func)
        result, elapsed = timed_func("Alice", greeting="Hi")

        self.assertEqual(result, "Hi, Alice!")
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()

import functools
import logging
from contextlib import contextmanager
from time import perf_counter_ns
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    # filename="panda.log",
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class TextColor:
    """
    Text color class
    """

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"

    @staticmethod
    def colorize(text: str, color: str | None = None) -> str:
        """
        Colorize text
        :param text: text to colorize
        :param color: color
        :return: colorized text
        """
        color = color or TextColor.RESET
        return f"{color}{text}{TextColor.RESET}"


def log_message(txt: str, color: str = TextColor.RESET):
    r"""
    Log message
    :param txt: text to log
    :param color: color

    """
    _color = TextColor.RESET
    if _color in dir(TextColor):
        _color = getattr(TextColor, color.upper())
    logging.debug(TextColor.colorize(txt, _color))


@contextmanager
def runtime(txt: str):
    start = perf_counter_ns()
    yield
    end = perf_counter_ns()
    logging.debug(
        TextColor.colorize(
            f"<{txt}> Runtime: {(end-start) * 1e-6:.1f} ms", TextColor.MAGENTA
        )
    )


def runtime_(func: Callable):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        with runtime(func.__name__):
            return func(*args, **kwargs)

    return _wrapper

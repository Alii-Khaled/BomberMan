import contextlib
import os


class QuietFallback:
    def __getattr__(self, item):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __iter__(self):
        return iter([])


try:
    with contextlib.redirect_stdout(None):
        import pygame
        LOADED_PYGAME = True
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    pygame.init()
except ModuleNotFoundError:
    pygame = QuietFallback()
    LOADED_PYGAME = True

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable, *args, **kwargs: iterable

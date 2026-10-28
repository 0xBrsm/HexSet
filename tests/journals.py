"""A fresh journal for every game-playing call in a test: every game is kept
(`hexset.arena.compete` and the rest name a journal), and a test's journals go
to one temporary directory removed when the session ends."""
import atexit
import itertools
import os
import shutil
import tempfile

_DIR = tempfile.mkdtemp(prefix="hexset-test-journals-")
atexit.register(shutil.rmtree, _DIR, True)
_COUNT = itertools.count()


def fresh(name: str = "games") -> str:
    """A journal path no other call has used."""
    return os.path.join(_DIR, f"{name}-{os.getpid()}-{next(_COUNT)}.jsonl")

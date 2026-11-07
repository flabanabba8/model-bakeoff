"""Cross-platform process locks for benchmark state and one in-flight run."""
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None
try:
    import msvcrt
except ImportError:
    msvcrt = None


@contextmanager
def exclusive_lock(path, busy, *, blocking=False, error=RuntimeError):
    with path.open('a+b') as stream:
        if fcntl is not None:
            try: fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except OSError: raise error(busy) from None
            yield
        elif msvcrt is not None:
            stream.seek(0, 2)
            if stream.tell() == 0: stream.write(b'\0'); stream.flush()
            stream.seek(0)
            try: msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            except OSError: raise error(busy) from None
            try: yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            raise ValueError('Benchmark operations require supported file locking')

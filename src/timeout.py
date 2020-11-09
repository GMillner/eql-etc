import signal
import threading
from contextlib import contextmanager

from six.moves import _thread


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    if hasattr(signal, "SIGALRM"):
        # for Linux
        def signal_handler(signum, frame):
            raise TimeoutException("Timeout after {} seconds.".format(seconds))

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # for Windows
        timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            raise TimeoutException("Timeout after {} seconds.".format(seconds))
        finally:
            timer.cancel()
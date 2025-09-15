import logging
import multiprocessing as mp
import traceback

logger = logging.getLogger(__name__)


class FunctionTimeoutError(TimeoutError):
    """Raised when a function call exceeds its allowed timeout."""


def run_with_timeout(func, args=(), kwargs=None, timeout=None):
    """
    Run `func(*args, **kwargs)` in a separate process.

    If the function takes longer than `timeout` seconds, kill it and raise FunctionTimeoutError.
    """
    logger.debug("In run_with_timeout")
    if kwargs is None:
        kwargs = {}

    def wrapper(q, *a, **kw):
        logger.debug("In timer wrapper")
        try:
            val = func(*a, **kw)
            logger.debug(f"Function returned: {val}")
            q.put((True, val))
        except Exception as e:
            logger.debug("Function exception!")
            q.put((False, (e, traceback.format_exc())))

    q = mp.Queue()
    p = mp.Process(target=wrapper, args=(q, *args), kwargs=kwargs)
    p.start()
    logger.debug("started worker thread")
    p.join(timeout)
    logger.debug("joined")

    if p.is_alive():
        logger.debug("p was still alive")
        p.terminate()
        p.join()
        raise FunctionTimeoutError(f"Function exceeded timeout of {timeout} seconds")

    if not q.empty():
        logger.debug("p was dead")
        ok, payload = q.get_nowait()
        logger.debug("got payload")
        if ok:
            return payload
        else:
            err, tb = payload
            raise err.__class__(f"{err}\nOriginal traceback:\n{tb}")
    else:
        raise RuntimeError("Worker exited without returning anything")


# ---------------- example ----------------
if __name__ == "__main__":
    import time

    def slow(x):
        time.sleep(x)
        return x

    try:
        print(run_with_timeout(slow, args=(10,), timeout=1))
    except FunctionTimeoutError as e:
        print("Timeout:", e)

    print(run_with_timeout(slow, args=(1,), timeout=3))  # returns 1

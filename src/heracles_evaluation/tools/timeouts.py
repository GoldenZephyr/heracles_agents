import multiprocessing as mp
import traceback

class FunctionTimeoutError(TimeoutError):
    """Raised when a function call exceeds its allowed timeout."""


def run_with_timeout(func, args=(), kwargs=None, timeout=None):
    """
    Run `func(*args, **kwargs)` in a separate process.
    
    If the function takes longer than `timeout` seconds, kill it and raise FunctionTimeoutError.
    """
    if kwargs is None:
        kwargs = {}

    def wrapper(q, *a, **kw):
        try:
            q.put((True, func(*a, **kw)))
        except Exception as e:
            q.put((False, (e, traceback.format_exc())))

    q = mp.Queue()
    p = mp.Process(target=wrapper, args=(q, *args), kwargs=kwargs)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise FunctionTimeoutError(f"Function exceeded timeout of {timeout} seconds")

    ok, payload = q.get()
    if ok:
        return payload
    else:
        err, tb = payload
        raise err.__class__(f"{err}\nOriginal traceback:\n{tb}")


# ---------------- example ----------------
if __name__ == "__main__":
    import time

    def slow(x):
        time.sleep(x)
        return x

    try:
        print(run_with_timeout(slow, args=(2,), timeout=1))
    except FunctionTimeoutError as e:
        print("Timeout:", e)

    print(run_with_timeout(slow, args=(1,), timeout=3))  # returns 1


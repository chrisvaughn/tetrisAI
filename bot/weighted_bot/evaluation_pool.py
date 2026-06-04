import multiprocessing
import signal

pool = None


def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_pool(parallel=4):
    global pool
    if pool is None:
        print("Initializing Pool")
        multiprocessing.set_start_method("spawn")
        pool = multiprocessing.Pool(parallel, initializer=_worker_init)
    return pool


def shutdown_pool():
    global pool
    if pool is not None:
        pool.terminate()
        pool.join()
        pool = None

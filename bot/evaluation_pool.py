import multiprocessing

pool = None


def get_pool(parallel=8):
    global pool
    if pool is None:
        pool = multiprocessing.Pool(parallel)
    return pool

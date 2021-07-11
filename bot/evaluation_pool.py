import multiprocessing

pool = None


def get_pool(parallel=6):
    global pool
    if pool is None:
        multiprocessing.set_start_method("spawn")
        pool = multiprocessing.Pool(parallel)
    return pool

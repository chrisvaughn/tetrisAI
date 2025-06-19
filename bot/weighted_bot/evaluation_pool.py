import multiprocessing

pool = None


def get_pool(parallel=4):
    global pool
    if pool is None:
        print("Initializing Pool")
        multiprocessing.set_start_method("spawn")
        pool = multiprocessing.Pool(parallel)
    return pool

import multiprocessing
import signal

pool = None
_genome_fitness_fn = None


def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _genome_worker_init(fitness_fn):
    global _genome_fitness_fn
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _genome_fitness_fn = fitness_fn


def _evaluate_genome(weights):
    return _genome_fitness_fn(weights)


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

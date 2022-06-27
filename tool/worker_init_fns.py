import numpy as np



def worker_init_fn(worker_id):
    np.random.seed(1 + worker_id)
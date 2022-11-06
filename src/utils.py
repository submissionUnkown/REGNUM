from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def mult_proc_apply(func, elements, initializer=None, disable_tqdm=False, nb_workers=12):
    nb_workers = min(cpu_count(), nb_workers)
    p = Pool(nb_workers, initializer=initializer)
    try:
        res = list(tqdm(p.imap(func, elements), total=len(elements), disable=disable_tqdm))
    finally:
        p.close()

    return res


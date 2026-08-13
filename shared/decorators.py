import time


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


def init_decorator_without_torch(func):
    """
    Similar to init_decorator, but does not import torch or set torch settings.
    Useful for scripts that do not require PyTorch.
    """

    def init(h: dict):
        import numpy as np
        np.random.seed(h['seed'])

    def wrapper(h: dict, *args, **kwargs):
        assert type(h) == dict, \
            ("See `wandb_decorator` for more information.")

        init(h)
        result = func(h, *args, **kwargs)

        return result

    return wrapper


class Key:  # used for wandb logging
    @staticmethod
    def name(h: dict):
        return (f"targ={h['stop_early_threshold']}_alph={h['alpha']}_{h['data_type']}_N={h['N']}"
                f"_bsize={h['batch_size']}")

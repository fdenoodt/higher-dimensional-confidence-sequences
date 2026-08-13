from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from adaptive_sample.shared_code import DataDistributionChecker
from tqdm import tqdm


class BaseConfidenceSequence(ABC):
    def __init__(self, h: dict, data):
        self.h: dict = h
        self.alpha = h['alpha']
        self.batch_size = h['batch_size']
        self.data_type = h['data_type']

        self.N = len(data)
        self.D = DataDistributionChecker.dim(h)  # Number of dimensions
        self.data = data
        self.ts, self.lowers, self.uppers, self.means, self.volumes, self.volume_stds = self._initialize_arrays()

        self.sum_x = 0.0  # used for empirical mean
        self.total_n_seen = 0  # total number of observations seen so far

    def _initialize_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize lowers, uppers, and means based on dimensionality."""
        T = self.N // self.batch_size  # Number of batches
        shape = (T, self.D) if self.D > 1 else (T,)
        lowers = np.zeros(shape)  # (T, D)
        uppers = np.ones(shape)
        means = np.zeros(shape)
        volumes = np.zeros(T)
        volume_stds = np.zeros(T)

        # 1, 2, ... T
        ts = np.arange(1, T + 1)

        return ts, lowers, uppers, means, volumes, volume_stds

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.N // self.batch_size  # Number of batches
        D = self.D
        with tqdm(total=self.N, desc=f"{self.__class__.__name__} (T={T}, D={D})",
                  disable=not self.h['verbose_progress']) as pbar:
            for k, batch in enumerate(self.batch_generator(), 1):  # k starts at 1
                for x in batch:
                    self.total_n_seen += 1
                    self.sum_x += x
                    pbar.update(1)

                self.update(k, batch)
                mu_t = self.sum_x / self.total_n_seen
                self.means[k - 1] = mu_t  # k is batch_idx, starting from 1

        return self.ts, self.means, self.lowers, self.uppers, self.volumes, self.volume_stds

    def __call__(self):
        return self.run()

    @abstractmethod
    def update(self, k: int, batch: np.ndarray):
        """Subclasses must implement this method to update their results."""
        pass

    def batch_generator(self):
        for i in range(0, self.N, self.batch_size):
            yield self.data[i:i + self.batch_size]

    # some static function used in various hedged capital classes (e.g. 1d but also nd convex)
    @staticmethod
    def _extract_means(h, data, N: int, batch_size: int) -> np.ndarray:
        """Stack means from each 1-D calculator into a (T, D) array."""

        def _batch_generator(data, N, batch_size):
            for i in range(0, N, batch_size):
                yield data[i:i + batch_size]

        data = np.asarray(data)
        data_shape = data.shape  # (N, D), (N,), or occasionally (N, 1) for 1d data
        if len(data_shape) == 1:
            n, D = N, 1
            data_1d = data
        else:
            n, D = data.shape  # Purposely not use DataDistributionChecker.dim(h) because this function might be called from hedged_cap_1d which was instantiated by convex_sum and thus D > 1 but the 1d calculator only sees 1d
            assert n == N, f"Expected data length {N}, got {n}."
            data_1d = data[:, 0] if D == 1 else None

        T = N // batch_size
        means = np.zeros((T, D)) if D > 1 else np.zeros(T)
        sum_x = np.zeros(D) if D > 1 else 0.0
        total_n_seen = 0
        batch_source = data if D > 1 else data_1d
        for k, batch in enumerate(_batch_generator(batch_source, N, batch_size), 1):  # k starts at 1
            for x in batch:
                total_n_seen += 1
                sum_x += x

            mu_t = sum_x / total_n_seen
            means[k - 1] = mu_t  # k is batch_idx, starting from 1

        return means


class MultiDimensionalConfidenceSequence():

    @abstractmethod
    def compute_volume(self, t, Kd_calcs, low: np.ndarray, up: np.ndarray, extra: dict) -> Tuple[
        float, float]:  # val, std
        # part of `run`
        raise NotImplementedError

    @abstractmethod
    def compute_volume_at(self, t: int) -> Tuple[float, float]:
        """Compute volume at time t. Returns (volume, std)."""
        # independent of run()
        raise NotImplementedError

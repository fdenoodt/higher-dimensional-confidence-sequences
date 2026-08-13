from typing import List, Tuple

import numpy as np

from adaptive_sample.conf_sequences import BaseConfidenceSequence, MultiDimensionalConfidenceSequence
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_1d import HedgedCapitalCI
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_convex_sum import _portfolio_weights
from adaptive_sample.conf_sequences.hedged_capital.hedged_factory import HedgedFactory
from adaptive_sample.helper.hypercube_tools import HypercubeTools
from adaptive_sample.helper.volume import VolumeCalculator
from adaptive_sample.plotting import Plotter


class HedgedCapitalCIMultiDimensionalBonferroni(BaseConfidenceSequence, MultiDimensionalConfidenceSequence):
    """
    Hedged capital confidence intervals for multi-dimensional data using Bonferroni correction.
    Reports individual confidence intervals for each dimension.
    """

    def __init__(self, h: dict, data):
        super().__init__(h, data)
        self.volumes = None  # Volumes of the confidence sets
        self._kd_calcs_cache = None

    def update(self, k: int, batch: np.ndarray):
        """
        This method is not used in this class.
        It is here to satisfy the BaseConfidenceSequence interface.
        """
        raise NotImplementedError("This class does not support incremental updates. Use `run` instead.")

    def run(self):
        data = self.data

        n_samples, D = data.shape
        B = self.batch_size
        T = n_samples // B  # Number of intervals
        weights = _portfolio_weights(self.h, D)

        # IMPORTANT: build Kd_calculators with the SAME alpha as the multidim routine.
        # Original implementation was construct Kd_calcs and then set calc.alpha = self.h['alpha'] / D.
        # but this mutatation results in different behaviour because otherwise calc._lambda_tilde bets
        # are different, and proof doesn't hold
        Kd_calculators: List[HedgedCapitalCI] = HedgedFactory.build_and_cache_Khedgeds(self.h, data)

        # allocate
        lowers = np.zeros((T, D))
        uppers = np.zeros((T, D))

        for d, calc in enumerate(Kd_calculators):
            target_d = 1.0 / (self.alpha * weights[d])
            lowers[:, d] = [calc.find_confidence_interval_1d(t_idx, target_d)[0] for t_idx in range(T)]
            uppers[:, d] = [calc.find_confidence_interval_1d(t_idx, target_d)[1] for t_idx in range(T)]

        if self.h['cut_to_simplex']:
            # Adjust the lower and upper bounds to respect the simplex constraint
            for t in range(T):
                lowers_t, uppers_t = HypercubeTools.hypercube_and_simplex_intersection(lowers[t], uppers[t], D)
                lowers[t, :], uppers[t, :] = lowers_t, uppers_t

        volumes = VolumeCalculator.hypercube_volumes(
            self.h, lowers, uppers
        )

        self.lowers = lowers
        self.uppers = uppers
        self.means = self._extract_means(self.h, data, n_samples, B)
        self.volumes = volumes

        return self.ts, self.means, self.lowers, self.uppers, self.volumes, self.volume_stds

    def compute_volume(self, t, Kd_calcs, low: np.ndarray, up: np.ndarray, extra: dict) -> Tuple[float, float]:
        raise NotImplementedError("This class does not support incremental volume computation. Use `run` instead.")

    def compute_volume_at(self, t: int) -> Tuple[float, float]:
        """Compute volume for the Bonferroni multi-dim CS at batch-index t (0-based)."""
        data = self.data
        n_samples, D = data.shape
        B = self.batch_size
        T = n_samples // B
        if not (0 <= t < T):
            raise IndexError(f"t={t} out of range [0, {T - 1}]")

        # rebuild Kd calculators (important: HedgedFactory should match run() usage)
        Kd_calculators: List[HedgedCapitalCI] = HedgedFactory.build_and_cache_Khedgeds(self.h, data)
        weights = _portfolio_weights(self.h, D)

        # compute per-d interval at time t
        lowers = np.zeros(D)
        uppers = np.zeros(D)
        for d, calc in enumerate(Kd_calculators):
            target_d = 1.0 / (self.alpha * weights[d])
            lower_d, upper_d = calc.find_confidence_interval_1d(t, target_d)
            lowers[d] = lower_d
            uppers[d] = upper_d

        # optional simplex cut
        if self.h.get('cut_to_simplex', False) and D > 1:
            lowers, uppers = HypercubeTools.hypercube_and_simplex_intersection(lowers, uppers, D)

        vol = VolumeCalculator.hypercube_volume(self.h, lowers, uppers)
        return float(vol), 0.0

    def is_member_at(self, x, t: int) -> bool:
        """Check if x is in the Bonferroni confidence set at 0-based batch index t."""
        data = self.data
        n_samples, D = data.shape
        B = self.batch_size
        T = n_samples // B
        if not (0 <= t < T):
            raise IndexError(f"t={t} out of range [0, {T - 1}]")

        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = np.array([float(x)])
        if x.shape[-1] != D:
            raise ValueError(f"Expected x with last dimension {D}, got {x.shape}.")

        if self._kd_calcs_cache is None:
            self._kd_calcs_cache = HedgedFactory.build_and_cache_Khedgeds(self.h, data)

        weights = _portfolio_weights(self.h, D)

        lowers = np.zeros(D)
        uppers = np.zeros(D)
        for d, calc in enumerate(self._kd_calcs_cache):
            target_d = 1.0 / (self.alpha * weights[d])
            lower_d, upper_d = calc.find_confidence_interval_1d(t, target_d)
            lowers[d] = lower_d
            uppers[d] = upper_d

        if self.h.get('cut_to_simplex', False) and D > 1:
            lowers, uppers = HypercubeTools.hypercube_and_simplex_intersection(lowers, uppers, D)

        return np.all(x >= lowers) and np.all(x <= uppers)

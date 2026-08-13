import itertools
import math
from typing import List, Tuple
import numpy as np

from adaptive_sample.conf_sequences import BaseConfidenceSequence
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_1d import HedgedCapitalCI
from adaptive_sample.conf_sequences.hedged_capital.hedged_factory import HedgedFactory
from adaptive_sample.plotting import Plotter


class HedgedCapitalCIMultiDimensionalMax(BaseConfidenceSequence):
    """
    Hedged capital as max of weighted martingales:
    M_t(m) = max_{d=1}^D w_d * K^d_t(m_d)

    The confidence region is {m | max_d(w_d * K^d(m_d)) < 1/α}
    which is equivalent to {m | w_d * K^d(m_d) < 1/α for all d}
    """

    def __init__(self, h: dict, data):
        raise NotImplementedError("This class is not fully implemented yet. Use HedgedCapitalCIMultiDimensionalBonferroni instead.")
        if h['hedged_computation'] == 'grid_search':
            self.run = self.run_grid_search
        elif h['hedged_computation'] == 'fast_approximation':
            self.run = self.run_fast_approximation
        else:
            raise ValueError(f"Unknown hedged computation method: {h['hedged_computation']}")
        super().__init__(h, data)

    def update(self, k: int, batch: np.ndarray):
        raise NotImplementedError("This method should not be called directly. Use run instead.")

    @staticmethod
    def K_max_combination(m_vec: Tuple[float], Kd_calculators: List[HedgedCapitalCI],
                          m_grids: List[np.ndarray], w, D, t):
        """
        Compute K(m) = max_d(w_d * K^d(m_d))

        Parameters:
        - m_vec: tuple of (D), e.g. (0.1, 0.2, 0.7) representing the point in D-dimensional space
        - Kd_calculators: list of D one-dimensional hedged capital calculators
        - m_grids: list of grids for each dimension
        - w: weights array of shape (D,)
        - D: number of dimensions
        - t: time index

        Returns:
        - scalar: max over dimensions of weighted hedged capital
        """
        K_values = [w[d] * Kd_calculators[d].get_Khedged_t(t, m_vec[d]) for d in range(D)]
        return max(K_values)

    @staticmethod
    def build_and_cache_Khedgeds(h: dict, data: np.ndarray, D: int) -> List[HedgedCapitalCI]:
        """Build one-dimensional hedged capital calculators for each dimension."""
        Kd_calculators: List[HedgedCapitalCI] = [
            HedgedCapitalCI(h, data[:, d]) for d in range(D)
        ]
        for d, calc in enumerate(Kd_calculators):
            calc.run()
        return Kd_calculators

    def run_fast_approximation(self, data: np.ndarray):
        """
        Fast approximation for max-based confidence intervals.

        For the max approach, the confidence region is:
        {m | max_d(w_d * K^d(m_d)) < 1/α}

        This is equivalent to:
        {m | w_d * K^d(m_d) < 1/α for all d}

        So for each dimension d, we need:
        K^d(m_d) < 1/(α * w_d)
        """
        n, D = data.shape
        B = self.batch_size
        T = n // B
        C = 1.0 / self.alpha

        # Build 1-D calculators
        Kd_calcs = self.build_and_cache_Khedgeds(self.h, data, D)

        lowers = np.zeros((T, D))
        uppers = np.zeros((T, D))

        for t in range(T):
            # Compute weights (equal weights for simplicity)
            w = np.ones(D) / D

            # For each dimension, find confidence interval with threshold 1/(α * w_d)
            for d, calc in enumerate(Kd_calcs):
                threshold_d = C / w[d]  # = D/α for equal weights
                lower_d, upper_d = calc.find_confidence_interval_1d_at(t, threshold_d)
                lowers[t, d] = lower_d
                uppers[t, d] = upper_d

            # Intersect with simplex if requested
            if self.h.get('cut_to_simplex', False) and D > 1:
                for d in range(D):
                    other_l = lowers[t, np.arange(D) != d].sum()
                    other_u = uppers[t, np.arange(D) != d].sum()
                    lowers[t, d] = max(lowers[t, d], 1 - other_u)
                    uppers[t, d] = min(uppers[t, d], 1 - other_l)

        Plotter.plot_2d_boundary(self.h, lowers, uppers)

        self.lowers = lowers
        self.uppers = uppers
        self.means = self._extract_means(self.h, data, n, B)

        return self.ts, self.means, self.lowers, self.uppers

    def run_grid_search(self, data: np.ndarray):
        """
        Grid search implementation for max-based confidence intervals.
        """
        n_samples, D = data.shape
        B = self.batch_size
        T = n_samples // B

        # Create D calculators for each dimension
        Kd_calculators = self.build_and_cache_Khedgeds(self.h, data, D)

        # Build Cartesian grid
        m_grids = [calc.m_grid for calc in Kd_calculators]
        grid_pts: List[tuple] = list(itertools.product(*m_grids))

        # Optionally filter for simplex
        if self.h['cut_to_simplex']:
            grid_pts = [pt for pt in grid_pts if math.isclose(sum(pt), 1.0, rel_tol=1e-9)]

        M = len(grid_pts)

        # Compute max-based wealth process
        K_max = np.zeros((T, M))

        for t in range(T):
            # Equal weights
            w = np.ones(D) / D

            for idx, m_vec in enumerate(grid_pts):
                # Compute K(m) = max_d(w_d * K^d(m_d))
                K_max[t, idx] = self.K_max_combination(
                    m_vec, Kd_calculators, m_grids, w, D, t
                )

        if self.h['plot_grid_2d']:
            self.plot_khedged_2d(K_max, T, D, grid_pts, n_m=Kd_calculators[0].n_m)

        # Find confidence regions
        lowers = np.zeros((T, D))
        uppers = np.zeros((T, D))

        for t in range(T):
            # Points in confidence region
            ok = K_max[t] < 1.0 / self.alpha

            if not ok.any():
                # Empty region - fallback to full space
                for d in range(D):
                    lowers[t, d] = 0.0
                    uppers[t, d] = 1.0
                continue

            # Project to marginal intervals
            kept = np.array(grid_pts)[ok]
            for d in range(D):
                lowers[t, d] = kept[:, d].min()
                uppers[t, d] = kept[:, d].max()

            print(f"T {t + 1}/{T}: Max-based confidence region computed")

        # Store results
        self.wealth_process_over_time = K_max
        self.lowers = lowers
        self.uppers = uppers

        # Compute empirical means
        self.means = self._extract_means(self.h, data, n_samples, B)

        return self.ts, self.means, self.lowers, self.uppers

    def plot_khedged_2d(self, wealth, T, D, grid_pts, n_m=99):
        """Plot 2D visualization of the max-based wealth process."""
        if D == 2:
            if self.h['cut_to_simplex']:
                print("2D visualization only works for the full grid, not simplex.")
                return

            wealth_in_org_shape = wealth.reshape((T, n_m, n_m))
            for t in range(T):
                Plotter.plot_2d_function(self.h, wealth_in_org_shape[t, :, :],
                                         grid_pts, target=1.0 / self.alpha)

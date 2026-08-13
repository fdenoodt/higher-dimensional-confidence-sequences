from copy import deepcopy
from typing import Tuple

import numpy as np
from scipy.special import gammaln

from adaptive_sample.conf_sequences import BaseConfidenceSequence, MultiDimensionalConfidenceSequence
from adaptive_sample.helper.grid_cs_helper import GridCSHelper
from adaptive_sample.helper.volume import VolumeCalculator
from adaptive_sample.shared_code import DataDistributionChecker
from shared.decorators import timer_decorator

EPS = 1e-12


class UniversalPortfolioCS(BaseConfidenceSequence, MultiDimensionalConfidenceSequence):
    def __init__(self, h: dict, data):
        super().__init__(h, data)
        assert h['cut_to_simplex'] == True
        assert h['batch_size'] == 1, \
            ("Universal Portfolio CS requires batch_size=1, "
             "otherwise might be some abiguity in total_n_seen considered as time t.")

        self.alpha_dirchl = np.array(
            np.ones(self.D) * (1 / (2 * self.D)))  # Dirichlet prior parameters (dirichl prior for UP)
        self.log_B_alpha = self.log_multivar_beta(self.alpha_dirchl)

        # DP table: keys are count vectors (tuples), values are log(y^t[k])
        self.dp = {tuple([0] * self.D): 0.0}  # Start with zero counts (log(1) = 0)
        # so dp[k] \in \R with k \in N^D. More specially, k's components sum to t
        # ^^ y is updated on the fly I think
        # we thus have, e.g. for 2d observations: dp[(8, 2)] \in R where k=(8, 2) sums to 10

        # cache for rebuilt dp tables: num_obs -> dp dict (log-values)
        self._dp_cache = {}  # simple memoization to avoid repeated rebuilds

    def log_multivar_beta(self, alpha_vec):
        """Compute log of multivariate beta function"""
        # \Beta(\alpha_dirichl) := (\prod_i^K gamma(\alpha_dirch)) / gamma(\sum_i alpha_dirchl_i)
        return np.sum(gammaln(alpha_vec)) - gammaln(np.sum(alpha_vec))

    def log_wealth(self, m, dp: dict = None):
        """
        Compute log wealth for candidate mean m
        If dp is None, uses self.dp (stateful). If dp is provided, uses that dict (stateless).
        usually None, but for `volume_at(t)` is useful to give our own dp to specify `t`
        """

        if dp is None:  # usual case
            dp_iter = self.dp.items()
            log_B_alpha_use = self.log_B_alpha
        else:
            dp_iter = dp.items()
            # we still reuse the instance alpha vector
            log_B_alpha_use = self.log_multivar_beta(self.alpha_dirchl)

        total = -np.inf  # Start with log(0) = -inf
        for k_vec, log_path_val in dp_iter:  # iterate over all vectors that sum to t
            # Compute term: log(y^t[k]) + sum(k_j * log(1/m_j)) + log(B(k+alpha)/B(alpha))
            log_term = log_path_val  # log(y^t[k])

            # Add log((1/m_j)^{k_j})
            for j in range(self.D):
                log_term += k_vec[j] * np.log(1 / (m[j] + 1e-20))  # Avoid division by zero

            # Add log(B(k+alpha)/B(alpha))
            k_alpha = np.array(k_vec) + self.alpha_dirchl
            log_B_k_alpha = self.log_multivar_beta(k_alpha)
            log_term += (log_B_k_alpha - log_B_alpha_use)

            # Accumulate in log space
            total = np.logaddexp(total, log_term)  # calculates log(exp(x1) + exp(x2))

        return total

    def wealth(self, m, dp: dict = None):
        """Compute wealth for candidate mean m"""
        return np.exp(self.log_wealth(m, dp))

    def is_in_cs(self, m, dp: dict = None):
        """Check if m is in confidence set"""
        # return np.exp(self.log_wealth(m)) < 1 / self.alpha
        # return Wealth(m) < 1 / self.alpha
        # W(m) < (1 / self.alpha)
        # <=>
        # log(W(m)) < log(1/delta)
        # <=>
        # log(W(m)) < log(1) - log(delta) = -log(delta)
        return self.log_wealth(m, dp) < -np.log(self.alpha)

    def _dp_from_data(self, num_obs: int):
        """Reconstruct DP table from the start up to num_obs observations (num_obs >= 0).
        Returns a dict mapping count-tuple -> log-value.
        Uses a simple cache (self._dp_cache[num_obs]) to avoid repeated rebuilds."""
        if num_obs in self._dp_cache:
            return self._dp_cache[num_obs]

        dp_local = {tuple([0] * self.D): 0.0}
        data = self.data
        # guard: if num_obs is 0, return the base dp containing zero-count
        for obs_idx in range(num_obs):
            x = np.asarray(data[obs_idx], dtype=float)
            new_dp = {}
            for k_vec, log_val in dp_local.items():
                for j in range(self.D):
                    new_k = list(k_vec)
                    new_k[j] += 1
                    new_k = tuple(new_k)
                    new_log_val = log_val + np.log(x[j] + 1e-20)
                    if new_k in new_dp:
                        new_dp[new_k] = np.logaddexp(new_dp[new_k], new_log_val)
                    else:
                        new_dp[new_k] = new_log_val
            dp_local = new_dp

        # store in cache (shallow copy is fine; keys/values are immutable floats/tuples)
        self._dp_cache[num_obs] = dp_local
        return dp_local

    def update(self, t: int, batch: np.ndarray):
        # t starts from 1
        assert len(batch) == 1, "Batch should be a single observation (1D array)"
        x = batch[0]  # Single observation at time t, in paper is y_t

        """Update DP table with new observation y_t"""
        new_dp = {}

        # dynamic programming: y^t[k] := \sum_j^K y_{tj} y^{t-1}[k-e_j] II{k_j >= 1}
        for k_vec, log_val in self.dp.items():
            for j in range(self.D):
                # Create new count vector by adding one to j-th dimension
                new_k = list(k_vec)
                new_k[j] += 1
                new_k = tuple(new_k)

                # Update DP: log(y^{t}[k]) = log(y^{t-1}[k']) + log(y_tj)
                new_log_val = log_val + np.log(x[j])
                if new_k in new_dp:
                    new_dp[new_k] = np.logaddexp(new_dp[new_k], new_log_val)
                else:
                    new_dp[new_k] = new_log_val

        self.dp = new_dp

        # also update cached version
        self._dp_cache[t - 1] = self.dp.copy()

        # Compute confidence set bounds
        grid, in_cs = GridCSHelper.confidence_set(
            self.h, self.is_in_cs, self.D, grid_resolution=self.h[
                'grid_resolution'])  # in_cs is a boolean array indicating which points are in the confidence set

        vol, vol_std = self.compute_volume(None, None, None, None, {})

        # Store bounds for each axis
        lower_t, upper_t = np.zeros(self.D), np.ones(self.D)  # Full simplex bounds
        for d_idx in range(self.D):
            if in_cs:
                lower_t[d_idx] = np.min(grid[in_cs][:, d_idx])
                upper_t[d_idx] = np.max(grid[in_cs][:, d_idx])
            else:
                lower_t[d_idx] = 0.0
                upper_t[d_idx] = 1.0

            # Plotter.GridPlotter.plot_confidence_region(
            #     self.h, grid, in_cs, f"Confidence Set at Time {self.total_n_seen}",
            #     self.D)

        self.lowers[t - 1] = lower_t  # notice that we use observation idx although should be batch idx
        self.uppers[t - 1] = upper_t
        self.volumes[t - 1] = vol
        self.volume_stds[t - 1] = vol_std

    def compute_volume(self, _t, _Kd_calcs, _low: np.ndarray, _up: np.ndarray, _extra: dict) -> Tuple[
        float, float]:
        # No args are used

        # volumes[r, t] = self.compute_cs_volume(grid, in_cs)
        # volume = GridCSHelper.compute_cs_volume(self.h, grid, in_cs, self.D)
        is_in_cs_func = lambda m: self.is_in_cs(m)
        vol, vol_std = VolumeCalculator.monte_carlo_cs_volume(self.h, is_in_cs_func, D=self.D)
        return vol, vol_std

    def compute_volume_at(self, t: int) -> Tuple[float, float]:
        """Compute volume for batch-index t (0-based). Rebuilds DP up to obs (t+1)*B and uses
                the stateless log_wealth/is_in_cs to avoid mutating instance state."""
        assert self.h['calculate_volume'] == True

        n_samples = self.data.shape[0]
        B = self.batch_size
        T = n_samples // B
        if not (0 <= t < T):
            raise IndexError(f"t={t} out of range [0, {T - 1}]")

        num_obs = (t + 1) * B  # for batch_size==1 this equals t+1

        # rebuild dp using helper (cached)
        dp_local = self._dp_from_data(num_obs)

        # membership function: uses stateless is_in_cs with dp_local
        is_in_cs_func = lambda m: self.is_in_cs(m, dp=dp_local)

        vol, vol_std = VolumeCalculator.monte_carlo_cs_volume(self.h, is_in_cs_func, D=self.D)
        return float(vol), float(vol_std)


class BoundedVectorUPCS(BaseConfidenceSequence, MultiDimensionalConfidenceSequence):
    """
    Wrapper that maps [0,1]^{K-1} observations to a K-dimensional probability vector
    via eq.(16) in the paper, then delegates to UniversalPortfolioCS that expects simplex data.
    This is a thin wrapper; it **creates** a UniversalPortfolioCS with transformed data and
    then forwards update/log_wealth/is_in_cs/compute_volume_at.
    """

    def __init__(self, h: dict, data: np.ndarray):
        """
        data: shape (N, K-1) with values in [0,1]
        After transformation the new data shape is (N, K) and sums to 1 by eq.(16).
        """
        super().__init__(h, data)
        assert h['cut_to_simplex'] == False, "Wrapper will set cut_to_simplex=True for sub-UP."

        # compute K from input
        assert data.ndim == 2  # (N, K-1)

        N, K_minus_1 = data.shape
        K = K_minus_1 + 1
        Y_tilde = self._transform__dataset_to_simplex(data)  # (N, K)

        # now construct a UniversalPortfolioCS with simplex data
        h_up = deepcopy(h)
        h_up['cut_to_simplex'] = True
        # make e.g. `fixed_2d` to 3d
        data_type_name = h['data_type'].split('_')[0]  # e.g. 'fixed'
        h_up['data_type'] = f"{data_type_name}_{K}d"  # e.g. 'fixed_3d'

        # pass the mapped data into UniversalPortfolioCS
        self.up = UniversalPortfolioCS(h_up, Y_tilde)

    # forwarders:
    def log_wealth(self, m: np.ndarray, dp: dict = None) -> float:
        # m: a vector in [0,1]^{K-1} (user's parameterization) — we must map candidate m to m_tilde in simplex
        m_tilde = self._transform_to_simplex(m)
        return self.up.log_wealth(m_tilde, dp=dp)

    def _transform_to_simplex(self, m: np.ndarray) -> np.ndarray:
        """
        Transform a vector m in [0,1]^{K-1} to a K-dimensional probability vector m_tilde
        """
        m = np.asarray(m, dtype=float)  # (K-1,)
        assert m.ndim == 1

        K_minus_1 = m.shape[0]
        K = K_minus_1 + 1

        scale = float(1.0 / (K - 1))
        m_tilde = np.zeros(K, dtype=float)
        m_tilde[:K_minus_1] = m * scale
        m_tilde[K_minus_1] = 1.0 - scale * np.sum(m)

        m_tilde = np.clip(m_tilde, EPS, 1.0 - EPS)  # avoid exact zeros
        m_tilde = m_tilde / np.sum(m_tilde)  # renormalize to sum to 1 (due to clipping)

        return m_tilde

    def _transform__dataset_to_simplex(self, data: np.ndarray) -> np.ndarray:
        N, K_minus_1 = data.shape
        K = K_minus_1 + 1
        # apply eq.(16) mapping: for each t
        # Y_tilde_j = Y_tj / (K-1) for j in [K-1]
        # Y_tilde_K = 1 - (1/(K-1)) * sum_{j=1}^{K-1} Y_tj
        scale = float(1.0 / (K - 1))
        Y_tilde = np.zeros((N, K), dtype=float)
        Y_tilde[:, :K_minus_1] = data * scale
        sums = np.sum(data, axis=1) * scale
        Y_tilde[:, K_minus_1] = 1.0 - sums  # last coordinate

        # numeric safety: clip and renormalize row-wise
        Y_tilde = np.clip(Y_tilde, EPS, 1.0 - EPS)
        row_sums = Y_tilde.sum(axis=1, keepdims=True)
        Y_tilde = Y_tilde / row_sums

        return Y_tilde

    def _inverse_bounds_from_up_box(self, up_lowers, up_uppers):
        """
        Analytic inversion for axis-aligned box:
          up_lowers/up_uppers: length-K arrays (simplex coords)
        Returns (orig_lower, orig_upper) for the first K-1 original coords.
        """
        K = len(up_lowers)
        s = K - 1
        orig_lower = s * up_lowers[:s]  # (K-1,)
        orig_upper = s * up_uppers[:s]

        orig_lower = np.clip(orig_lower, 0.0, 1.0)
        orig_upper = np.clip(orig_upper, 0.0, 1.0)

        return orig_lower, orig_upper  # (K-1,)

    def is_in_cs(self, m: np.ndarray, dp: dict = None) -> bool:
        # (K-1)-dimensional array, will be mapped to K in `self.log_wealth`
        return self.log_wealth(m, dp=dp) < -np.log(self.up.alpha) and np.all(
            (0.0 <= m) & (m <= 1.0))

    def update(self, t: int, batch: np.ndarray):
        """
        batch: shape (1, K-1) (a single observation in original [0,1]^{K-1} space)
        transform it to simplex and forward update
        """
        assert batch.shape[0] == 1
        x = batch[0]

        x_tilde = self._transform_to_simplex(x)

        self.up.update(t, x_tilde.reshape(1, -1))

        # copy statistics from sub-UP to self
        up_lowers, up_uppers = self.up.lowers[t - 1], self.up.uppers[t - 1]
        self.lowers[t - 1], self.uppers[t - 1] = self._inverse_bounds_from_up_box(
            up_lowers, up_uppers)

        if self.h['calculate_volume']:  # use wrapper's compute_volume_at since it works on orginal space
            vol, vol_std = self.up.compute_volume_at(t - 1)  # t-1 since sub-up is 0-based
            self.volumes[t - 1] = vol
            self.volume_stds[t - 1] = vol_std

    def compute_volume_at(self, t: int) -> Tuple[float, float]:
        # return self.up.compute_volume_at(t)
        assert self.h[
                   'calculate_volume'] == True, \
            ("Volume computation must be enabled in hyperparams, "
             "else monte carlo sampling is skipped.")

        dp_local = self.up._dp_from_data((t + 1) * self.batch_size)
        is_in_cs_func = lambda m: self.is_in_cs(  # use wrapper's `is_in_cs` so volume is still in [0,1]^{K-1}
            m, dp=dp_local)

        vol, vol_std = VolumeCalculator.monte_carlo_cs_volume(self.h, is_in_cs_func, D=self.D,
                                                              bbx_lower=np.zeros(self.D),
                                                              bbx_upper=np.ones(self.D))
        return float(vol), float(vol_std)

    def is_member_at(self, x, t: int) -> bool:
        """Check if x is in the confidence set at 0-based batch index t."""
        n_samples = self.data.shape[0]
        B = self.batch_size
        T = n_samples // B
        if not (0 <= t < T):
            raise IndexError(f"t={t} out of range [0, {T - 1}]")

        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = np.array([float(x)])
        if x.shape[-1] != self.D:
            raise ValueError(f"Expected x with last dimension {self.D}, got {x.shape}.")

        dp_local = self.up._dp_from_data((t + 1) * B)
        return self.is_in_cs(x, dp=dp_local)


@timer_decorator
def main():
    # set fixed seed etc
    np.random.seed(42)

    h = {
        'alpha': 0.05,
        'batch_size': 1,
        'data_type': 'ellipse_4d',
        'N': 100,
        'cut_to_simplex': False,
        'grid_resolution': 100,
        'plot_grid_2d': False,
        'calculate_volume': True,
        'verbose_progress': True
    }

    # uniform data
    # data = np.random.uniform(0, 1, size=(h['N'], 2))
    # data = np.array([[0.7, 0.3, 0.3, 0.3]] * h['N'])
    # data = np.array([[0.5, 0.5]] * h['N'])
    data, true_mean = DataDistributionChecker.get_data(h['data_type'], h['N'])  # or "fixed" for constant data

    # cs = UniversalPortfolioCS(h, data)
    cs = BoundedVectorUPCS(h, data)

    vol = cs.compute_volume_at(99)
    # ts, means, lowers, uppers, volumes, volume_stds = cs.run()
    print(f"Volume: {vol}")

    # for l, u in zip(lowers, uppers):
    #     print(f"[{l[0]:2f} - {u[0]:2f}], [{l[1]:2f} - {u[1]:2f}]")
    #
    # print("Volumes:", volumes)
    # print("Volume stds:", volume_stds)
    # print("Means:", means)
    # print("Lowers:", lowers)
    # print("Uppers:", uppers)


if __name__ == "__main__":
    main()

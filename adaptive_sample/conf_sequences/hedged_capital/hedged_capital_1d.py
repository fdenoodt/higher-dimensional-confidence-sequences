import math
from typing import Dict
from typing import Type

import numpy as np
import matplotlib.pyplot as plt

from shared.decorators import timer_decorator
from adaptive_sample.conf_sequences import BaseConfidenceSequence
from adaptive_sample.conf_sequences.hedged_capital.f_approximation import bisection
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_utils import find_minimum_of_k


_LOG_MAX_FLOAT = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
_LOG_TINY_FLOAT = math.log(np.nextafter(0.0, 1.0))


def _safe_log_factor(factor: float) -> float:
    if factor <= 0.0:
        return -math.inf
    return math.log(factor)


def _safe_exp(log_value: float) -> float:
    if log_value == -math.inf:
        return 0.0
    if log_value >= _LOG_MAX_FLOAT:
        return math.inf
    return math.exp(log_value)


def _log_weighted_max(log_a: float, log_b: float, weight_a: float, weight_b: float) -> float:
    term_a = math.log(weight_a) + log_a if log_a != -math.inf and weight_a > 0.0 else -math.inf
    term_b = math.log(weight_b) + log_b if log_b != -math.inf and weight_b > 0.0 else -math.inf
    return max(term_a, term_b)


class HedgedCapitalCI(BaseConfidenceSequence):
    def __init__(self, h: dict, data):
        super().__init__(h, data)
        self.c = 0.75  # truncation parameter
        self.theta = 0.5  # hedging parameter

        # while BaseConfidenceSequence initializes means on the fly via .run(),
        # we want to pre-compute them so that this class also works when run() wasn't called,
        # and we just one to evaluate wealth() for a sinigle m.
        self.pre_computed_means = self._extract_means(
            h, data, N=self.N, batch_size=self.batch_size
        )

    def bisection_at_least(self, f, a, b, tol=1e-8, max_iter=100_000):
        """
        Modified bisection that returns the point where f(x) >= 0
        *guaranteed*, even if the exact root is slightly below due to floating error.
        """

        fa, fb = f(a), f(b)

        assert ((fa <= 0 <= fb) or (fb <= 0 <= fa)), \
            f"No sign change on [a,b]. f(a)={fa}, f(b)={fb}, a={a}, b={b}. " \
            f"Please ensure that f(a) and f(b) have opposite signs."

        # we only stop when the interval is tiny AND midpoint satisfies f >= 0.
        for _ in range(max_iter):
            if abs(b - a) < tol:
                # return the 'safe' side, can be either (depending on whether the left
                # part of m0 or right part of m0 was given). Sometimes f is increasing and sometimes decreasing
                if fa >= 0:
                    return a
                else:
                    return b

            c = 0.5 * (a + b)  # mid
            fc = f(c)

            # decide which half to keep
            if (fa <= 0 <= fc) or (fc <= 0 <= fa):
                b, fb = c, fc
            elif (fb <= 0 <= fc) or (fc <= 0 <= fb):
                a, fa = c, fc
            else:
                raise ValueError(f"f(c)={fc} does not have the same sign as f(a)={fa} or f(b)={fb}. "
                                 f"Please check your function and interval.")

        # final check, return safe side
        if f(a) >= 0:
            return a
        else:
            return b

    def find_confidence_interval_1d(self, t, target: float, actual_at_least_target: bool = False) -> (float, float):
        """
        Find the confidence interval for a hedged capital process in one dimension.

        If actual_at_least_target=True, the bisection stops only once f(x) >= 0,
        so that the returned point never undershoots the target due to numerical error.
        """
        log_target = math.log(target)
        m_min, m_max = 0.0, 1.0
        # m0 = self.pre_computed_means[t]
        _, _, a, b = find_minimum_of_k(
            k_plus=lambda x: math.log(self.theta) + self.get_log_kplus(t, x),
            k_minus=lambda x: math.log(1 - self.theta) + self.get_log_kminus(t, x),
            a=m_min,
            b=m_max,
            log_scale=True,
            eps_x=1e-12,
            eps_y=1e-12,
            max_iter=100,
        )
        m0 = 0.5 * (a + b)

        def f_exact(m):
            return self.get_log_wealth(t=t, m=m) - log_target

        def compute_interval(f):
            # --- LOWER BOUND ---
            if f(m_min) < 0 and f(m0) < 0:
                lower = m_min
            else:
                if actual_at_least_target:
                    lower = self.bisection_at_least(f, m_min, m0)
                else:
                    lower = bisection(f, m_min, m0)

            # --- UPPER BOUND ---
            if f(m0) < 0 and f(m_max) < 0:
                upper = m_max
            else:
                if actual_at_least_target:
                    upper = self.bisection_at_least(f, m0, m_max)
                else:
                    upper = bisection(f, m0, m_max)

            return lower, upper

        return compute_interval(f_exact)

    def _get_log_wealth(self, t: int, m: float) -> (float, float, float):
        """
        Computes K_t(m) directly based on observations data without caching or interpolation.
        t : Time index (starts at 0). !!NOTE get_wealth(0) means no data seen yet. so is equivalent to K^max_t=0(m)=0.5 in paper!
        m : float
        """
        # Extract all data seen up to time t
        seen_data = self.data[:t * self.batch_size]  # shape (t * batch_size,)

        # Initialize log capital processes for the specific m
        log_Kplus = 0.0
        log_Kminus = 0.0

        # Initialize running statistics
        sum_x = 0.0
        sum_dev2 = 0.0
        mu_hat = 0.5  # initial mean estimate
        sigma2_hat = 0.25  # initial variance estimate

        # Iterate over the observed data
        for i, x in enumerate(seen_data, start=1):
            sum_x += x
            dev = x - mu_hat
            sum_dev2 += dev ** 2
            mu_hat = (0.5 + sum_x) / (i + 1)
            sigma2_hat = (0.25 + sum_dev2) / (i + 1)

            # Compute lambda_tilde
            lam_tilde = self.lambda_t(i, self.alpha, sigma2_hat)

            # Update capital processes for the specific m
            lam_plus = min(lam_tilde, self.c / (m + 1e-10))  # avoid division by zero
            lam_minus = min(lam_tilde, self.c / (1 - m + 1e-10))  # avoid division by zero
            log_Kplus += _safe_log_factor(1 + lam_plus * (x - m))
            log_Kminus += _safe_log_factor(1 - lam_minus * (x - m))

        log_khedged = _log_weighted_max(log_Kplus, log_Kminus, self.theta, 1 - self.theta)

        return log_khedged, log_Kplus, log_Kminus

    def get_wealth(self, t: int, m: float) -> float:
        """Compute the hedged capital process K_t^max(m) at time t for mean m."""
        return _safe_exp(self.get_log_wealth(t, m))

    def get_log_wealth(self, t: int, m: float) -> float:
        log_khedged, _, _ = self._get_log_wealth(t, m)
        return log_khedged

    def get_kplus(self, t: int, m: float) -> float:
        """Get Kplus value for given (t, m)."""
        return _safe_exp(self.get_log_kplus(t, m))

    def get_log_kplus(self, t: int, m: float) -> float:
        _, log_Kplus, _ = self._get_log_wealth(t, m)
        return log_Kplus

    def get_kminus(self, t: int, m: float) -> float:
        """Get Kminus value for given (t, m)."""
        return _safe_exp(self.get_log_kminus(t, m))

    def get_log_kminus(self, t: int, m: float) -> float:
        _, _, log_Kminus = self._get_log_wealth(t, m)
        return log_Kminus

    @staticmethod
    def lambda_t(i, alpha, sigma2_hat):
        res = np.sqrt(2 * np.log(2 / alpha) /
                      (sigma2_hat * i * np.log(1 + i) + 1e-20))
        return res

    @staticmethod
    def update_capital_processes(x: float, lam_tilde: float, c: float,
                                 m_grid: np.ndarray, log_Kplus: np.ndarray, log_Kminus: np.ndarray):
        """Update log Kplus and log Kminus based on the current observation."""
        epsilon = 10e-10
        lam_plus = np.minimum(lam_tilde, c / (m_grid + epsilon))  # shape based on m_grid
        lam_minus = np.minimum(lam_tilde, c / (1 - m_grid + epsilon))
        plus_factors = 1 + lam_plus * (x - m_grid)
        minus_factors = 1 - lam_minus * (x - m_grid)
        log_Kplus = log_Kplus + np.where(plus_factors > 0.0, np.log(plus_factors), -np.inf)
        log_Kminus = log_Kminus + np.where(minus_factors > 0.0, np.log(minus_factors), -np.inf)

        return log_Kplus, log_Kminus

    def update(self, k: int, batch: np.ndarray):
        """ update() is called for every batch of observations.
        Update the confidence sequence with a new batch of observations.
        """

        lower_t, upper_t = self.find_confidence_interval_1d(
            t=k - 1, target=1.0 / self.alpha
        )  # Assumes self.get_wealth is defined

        self.lowers[k - 1] = lower_t
        self.uppers[k - 1] = upper_t

        # all below is for plotting only
        # f = lambda m: self.get_wealth(t=k - 1, m=m)
        # a_star, b_star, c_star = approximate_parabola_with_c(
        #     f, x_min=0.0, x_max=1.0)

        # Plotter.plot_1d_function(f=[f, lambda x: (x - a_star) ** 2 / b_star ** 2 - c_star],
        #                          x_min=0.0,
        #                          x_max=1.0,
        #                          title=f"Hedged Capital CI at t={k - 1}",
        #                          ylim=100)


# appprox through linear interpolation
class CachedHedgedCapitalCI(HedgedCapitalCI):
    def __init__(self, h: dict, data):
        super().__init__(h, data)
        self.num_cache_points = h['grid_resolution']
        # m grid we will maintain Kplus/Kminus for
        self.m_grid = np.linspace(0.0, 1.0, self.num_cache_points)

        # Cached results for time t
        self._log_Kplus = np.zeros(self.num_cache_points, dtype=float)
        self._log_Kminus = np.zeros(self.num_cache_points, dtype=float)

        # last computed t (in the sense used by get_wealth,
        # i.e. seen_data = data[: t * batch_size ])
        self.last_t: int = 0

        # running scalar stats corresponding to last_t (so we can continue incremental updates)
        self._sum_x_for_last = 0.0
        self._sum_dev2_for_last = 0.0
        self._mu_hat_for_last = 0.5
        self._sigma2_hat_for_last = 0.25
        # last sample index i (counts from 1 to n_obs). For last_t==0, last_i==0.
        self._last_i = 0

        # caches for quick interpolation: cache[t] = khedged_array (len = num_cache_points)
        self.cache: Dict[int, np.ndarray] = {}
        # cache the t=0 trivial khedged
        self.cache[0] = np.full(self.num_cache_points, max(self.theta, 1 - self.theta), dtype=float)

    def _advance_from_last_to(self, target_t: int):
        """
        Advance internal Kplus/Kminus and scalar stats from self.last_t to target_t
        by processing only the new observations (vectorized over m_grid).
        """
        if target_t <= self.last_t:
            return  # nothing to advance

        # number of observations per batch:
        bs = self.batch_size
        # current number of observations represented by last_t:
        start_idx = self.last_t * bs
        target_n_obs = target_t * bs
        # iterate through new observations (index from start_idx to target_n_obs-1)
        sum_x = self._sum_x_for_last
        sum_dev2 = self._sum_dev2_for_last
        mu_hat = self._mu_hat_for_last
        last_i = self._last_i

        log_Kplus = self._log_Kplus
        log_Kminus = self._log_Kminus
        c = self.c

        # small epsilon for division
        eps = 1e-10

        for obs_idx in range(start_idx, target_n_obs):
            x = float(self.data[obs_idx])  # ensure Python float
            # one-based sample index for this new observation
            i = last_i + 1

            # update scalar running statistics exactly as in original implementation
            sum_x += x
            dev = x - mu_hat
            sum_dev2 += dev * dev
            mu_hat = (0.5 + sum_x) / (i + 1)
            sigma2_hat = (0.25 + sum_dev2) / (i + 1)

            # compute lam_tilde for this i
            lam_tilde = self.lambda_t(i, self.alpha, sigma2_hat)

            # vectorized lam_plus / lam_minus over the whole m_grid
            # shape: (num_cache_points,)
            lam_plus = np.minimum(lam_tilde, c / (self.m_grid + eps))
            lam_minus = np.minimum(lam_tilde, c / (1.0 - self.m_grid + eps))

            # vectorized update of K arrays
            # note: broadcasting with (x - m_grid)
            plus_factors = 1.0 + lam_plus * (x - self.m_grid)
            minus_factors = 1.0 - lam_minus * (x - self.m_grid)
            log_Kplus = log_Kplus + np.where(plus_factors > 0.0, np.log(plus_factors), -np.inf)
            log_Kminus = log_Kminus + np.where(minus_factors > 0.0, np.log(minus_factors), -np.inf)

            # move forward
            last_i = i

        # store updated internal state
        self._log_Kplus = log_Kplus
        self._log_Kminus = log_Kminus
        self._sum_x_for_last = sum_x
        self._sum_dev2_for_last = sum_dev2
        self._mu_hat_for_last = mu_hat
        # recompute sigma2 for storage (not strictly necessary)
        self._sigma2_hat_for_last = (0.25 + sum_dev2) / (last_i + 1) if last_i > 0 else 0.25
        self._last_i = last_i
        self.last_t = target_t

        # store cached khedged array for this t so future calls can interpolate immediately
        log_khedged = np.maximum(np.log(self.theta) + self._log_Kplus, np.log(1 - self.theta) + self._log_Kminus)
        self.cache[target_t] = np.exp(np.clip(log_khedged, _LOG_TINY_FLOAT, _LOG_MAX_FLOAT))

    def get_wealth(self, t: int, m: float) -> float:
        raise NotImplementedError(
            "Use the vectorized get_wealth below with caching. due to kplus and kminus not stored separately.")
        """
        Cached + incremental vectorized version.
        - If we've already computed up to t, we only interpolate from cached array.
        - If t > last_t, we incrementally advance from last_t to t using only the new observations,
          updating Kplus/Kminus vectorized across the grid, then cache wealth for t.
        - If t < last_t and we have cache[t], use it. Otherwise fall back to computing from scratch.
        """
        # quick return if we have exact cached array for this t
        if t in self.cache:
            k_arr = self.cache[t]
            return float(np.interp(m, self.m_grid, k_arr))

        # if we can advance incrementally (t > last_t), do so:
        if t > self.last_t:
            self._advance_from_last_to(t)
            k_arr = self.cache[t]
            return float(np.interp(m, self.m_grid, k_arr))

        # else (t < last_t and not cached): fallback to older behaviour (compute directly), then cache
        # This rarely happens in typical forward-only usage.
        # We compute the exact wealth array over the m_grid by calling parent for each grid point.
        k_values = np.array([super().get_wealth(t, m_i) for m_i in self.m_grid])
        self.cache[t] = k_values
        return float(np.interp(m, self.m_grid, k_values))

    def get_kplus(self, t: int, m: float) -> float:
        """Get Kplus value for given (t, m)."""
        raise NotImplementedError(
            "Use the vectorized get_kplus below with caching. due to kplus and kminus not stored separately.")

    def get_kminus(self, t: int, m: float) -> float:
        """Get Kminus value for given (t, m)."""
        raise NotImplementedError(
            "Use the vectorized get_kminus below with caching. due to kplus and kminus not stored separately.")


class ExactCachedHedgedCapitalCI(HedgedCapitalCI):
    """
    Exact cached hedged-capital evaluator that stores (x_i, lam_tilde_i) for
    all observed datapoints up to last_t.  Allows fast, exact evaluation for
    any t <= last_t by slicing the stored arrays and vectorized product.

    Notes:
     - No interpolation: exact (mod floating point) evaluation for any m.
     - For very large t, use log_mode=True to reduce overflow/underflow issues.
    """

    def __init__(self, h: dict, data):
        super().__init__(h, data)
        self._max_obs = int(h["N"])  # pre-alloc capacity

        # storage for observed x and computed lam_tilde per datapoint
        self._xs = np.empty(self._max_obs, dtype=np.float64)
        self._lams = np.empty(self._max_obs, dtype=np.float64)
        self._filled = 0  # how many entries filled

        # running state for advancing
        self.last_t = 0
        self._sum_x_for_last = 0.0
        self._sum_dev2_for_last = 0.0
        self._mu_hat_for_last = 0.5
        self._sigma2_hat_for_last = 0.25
        self._last_i = 0  # number of processed observations

    def _advance_from_last_to(self, target_t: int):
        """Process new observations up to target_t and fill _xs/_lams arrays."""
        if target_t <= self.last_t:
            return

        bs = self.batch_size
        start_idx = self.last_t * bs
        target_n_obs = target_t * bs

        xs = self._xs
        lams = self._lams
        filled = self._filled

        sum_x = self._sum_x_for_last
        sum_dev2 = self._sum_dev2_for_last
        mu_hat = self._mu_hat_for_last
        last_i = self._last_i

        for obs_idx in range(start_idx, target_n_obs):
            x = float(self.data[obs_idx])
            i = last_i + 1

            # update running scalars exactly like original implementation
            sum_x += x
            dev = x - mu_hat
            sum_dev2 += dev * dev
            mu_hat = (0.5 + sum_x) / (i + 1)
            sigma2_hat = (0.25 + sum_dev2) / (i + 1)

            lam_tilde = self.lambda_t(i, self.alpha, sigma2_hat)

            if filled >= self._max_obs:
                # enlarge arrays if pre-allocation is too small
                new_max = int(max(self._max_obs * 1.5, filled + 1000))
                xs_new = np.empty(new_max, dtype=np.float64)
                lams_new = np.empty(new_max, dtype=np.float64)
                xs_new[:filled] = xs[:filled]
                lams_new[:filled] = lams[:filled]
                self._xs = xs = xs_new
                self._lams = lams = lams_new
                self._max_obs = new_max

            xs[filled] = x
            lams[filled] = lam_tilde
            filled += 1
            last_i = i

        # commit state back
        self._filled = filled
        self._xs = xs
        self._lams = lams
        self._sum_x_for_last = sum_x
        self._sum_dev2_for_last = sum_dev2
        self._mu_hat_for_last = mu_hat
        self._sigma2_hat_for_last = (0.25 + sum_dev2) / (last_i + 1) if last_i > 0 else 0.25
        self._last_i = last_i
        self.last_t = target_t

    def _get_log_wealth(self, t: int, m: float) -> (float, float, float):
        """
        Exact log-space evaluation for given (t, m).
        - If t > last_t: advances to t (fills arrays) then evaluates.
        - If t <= last_t: uses stored arrays up to n = t * batch_size and evaluates.
        """
        # ensure we have lam_tilde for all observations up to t
        if t > self.last_t:
            self._advance_from_last_to(t)

        n_obs = t * self.batch_size
        if n_obs == 0:
            return math.log(max(self.theta, 1 - self.theta)), 0.0, 0.0

        x_arr = self._xs[:n_obs]
        lam_arr = self._lams[:n_obs]

        eps = 1e-12
        lam_plus = np.minimum(lam_arr, self.c / (m + eps))
        lam_minus = np.minimum(lam_arr, self.c / (1.0 - m + eps))

        factors_plus = 1.0 + lam_plus * (x_arr - m)
        factors_minus = 1.0 - lam_minus * (x_arr - m)

        log_Kplus = float(np.sum(np.where(factors_plus > 0.0, np.log(factors_plus), -np.inf)))
        log_Kminus = float(np.sum(np.where(factors_minus > 0.0, np.log(factors_minus), -np.inf)))
        log_khedged = _log_weighted_max(log_Kplus, log_Kminus, self.theta, 1 - self.theta)
        return log_khedged, log_Kplus, log_Kminus


if __name__ == "__main__":
    @timer_decorator
    def main(h: Dict, cs_class: Type[HedgedCapitalCI]):
        # data = np.array([0.7] * h['N'])
        data = np.random.uniform(0.0, 1.0, size=h['N'])

        cs = cs_class(h, data)
        ts, means, lowers, uppers, vols, vol_stds = cs.run()

        return lowers, uppers


    def _sanity_checks(h):
        for cs_class in [HedgedCapitalCI, CachedHedgedCapitalCI, ExactCachedHedgedCapitalCI]:
            lw, up = main(h, cs_class=cs_class)

            print(f"Confidence intervals for {cs_class.__name__}:")
            for l, u in zip(lw[-10:], up[-10:]):
                print(f"\t{u - l:.4f}")


    def _test_cached_gap(h):
        def _compute_interval_gap(h, t, grid_resolutions):
            """Compute and print interval gaps for different caching strategies."""
            print(f"\n[Interval Gap] Testing with time step: {t}")
            h_copy = h.copy()
            h_copy['N'] = t
            h_copy['grid_resolution'] = 1_000  # Default grid resolution

            # Non-cached version
            lw_nc, up_nc = main(h_copy, cs_class=HedgedCapitalCI)

            # Cached versions
            for resolution in grid_resolutions:
                h_copy['grid_resolution'] = resolution
                lw_c, up_c = main(h_copy, cs_class=CachedHedgedCapitalCI)
                avg_error = np.mean(np.abs((np.array(up_c) - np.array(lw_c)) - (np.array(up_nc) - np.array(lw_nc))))
                print(f"\tGrid resolution {resolution}: Average error (Cached): {avg_error:.6f}")

            # Exact cached version
            lw_ec, up_ec = main(h_copy, cs_class=ExactCachedHedgedCapitalCI)
            avg_error_ec = np.mean(np.abs((np.array(up_ec) - np.array(lw_ec)) - (np.array(up_nc) - np.array(lw_nc))))
            print(f"\tExactCachedHedgedCapitalCI: Average error: {avg_error_ec:.6f}")

        def _compute_wealth_gap(h, grid_resolutions, data, t_values, m_values):
            """Compute and print wealth gaps for different caching strategies."""
            print("\n[Wealth Gap] Testing wealth differences")
            # Non-cached version
            cs_nc = HedgedCapitalCI(h, data)
            cs_nc.run()

            for resolution in grid_resolutions:
                print(f"\nGrid resolution: {resolution}")
                h['grid_resolution'] = resolution

                # Cached version
                cs_c = CachedHedgedCapitalCI(h, data=data)
                cs_c.run()

                # Exact cached version
                cs_ec = ExactCachedHedgedCapitalCI(h, data=data)
                cs_ec.run()

                # Compare wealths at different time steps and m values
                for t in t_values:
                    wealth_differences_c = []
                    wealth_differences_ec = []
                    for m in m_values:
                        wealth_nc = cs_nc.get_wealth(t, m)
                        wealth_c = cs_c.get_wealth(t, m)
                        wealth_ec = cs_ec.get_wealth(t, m)
                        wealth_differences_c.append(abs(wealth_nc - wealth_c))
                        wealth_differences_ec.append(abs(wealth_nc - wealth_ec))

                    avg_wealth_difference_c = np.mean(wealth_differences_c)
                    avg_wealth_difference_ec = np.mean(wealth_differences_ec)
                    print(f"\tTime step {t}: Avg wealth diff (Cached): {avg_wealth_difference_c:.6f}")
                    print(f"\tTime step {t}: Avg wealth diff (ExactCached): {avg_wealth_difference_ec:.6f}")

        # Parameters for testing
        grid_resolutions = [500, 1_000, 10_000]
        t_values = [10, 100, 500]
        m_values = np.linspace(0.0, 1.0, 100)
        data = np.array([0.7] * h['N'])

        # Run interval gap tests
        for t in t_values:
            _compute_interval_gap(h, t, grid_resolutions)

        # Run wealth gap tests
        _compute_wealth_gap(h, grid_resolutions, data, t_values, m_values)


    def _test_exact_cached_vs_non_cached(h):
        data = np.random.uniform(0.0, 1.0, size=h['N'])  # Random data for testing

        # Initialize both classes
        non_cached = HedgedCapitalCI(h, data)
        exact_cached = ExactCachedHedgedCapitalCI(h, data)

        # Test wealth computation for multiple time steps and m values
        t_values = [1, 10, 50, 100]  # Time steps to test
        m_values = np.linspace(0.0, 1.0, 10)  # Test m values

        for t in t_values:
            for m in m_values:
                wealth_non_cached = non_cached.get_wealth(t, m)
                wealth_exact_cached = exact_cached.get_wealth(t, m)
                assert np.isclose(wealth_non_cached, wealth_exact_cached, atol=1e-6), \
                    f"Wealth mismatch at t={t}, m={m}: {wealth_non_cached} != {wealth_exact_cached}"

        print("All tests passed!")

    # fix seeds:
    np.random.seed(42)

    _h = {
        'alpha': 0.05,
        'batch_size': 1,
        'data_type': 'fixed',
        'N': 550,
        'cut_to_simplex': False,
        'grid_resolution': 1_000,
        'plot_grid_2d': False,
        'calculate_volume': False,
        'save_plots_locally': False,
        'use_wandb': False,
        'verbose_progress': True,
    }
    # _test_cached_gap(_h)
    # _sanity_checks(_h)
    #
    # # Run the test
    # _test_exact_cached_vs_non_cached(_h)

    # Plot wealth/kplus/kminus for ExactCachedHedgedCapitalCI at a fixed t
    data = np.random.uniform(0.0, 1.0, size=_h['N'])
    cs_plot = HedgedCapitalCI(_h, data)
    #ExactCachedHedgedCapitalCI(_h, data)
    t_plot = 50
    m_grid = np.linspace(0.0, 1.0, 200)
    wealth_vals = [cs_plot.get_wealth(t_plot, m) for m in m_grid]
    kplus_vals = [cs_plot.get_kplus(t_plot, m) for m in m_grid]
    kminus_vals = [cs_plot.get_kminus(t_plot, m) for m in m_grid]

    plt.figure()
    plt.plot(m_grid, wealth_vals, label="get_wealth", linestyle='--')
    plt.plot(m_grid, kplus_vals, label="get_kplus")
    plt.plot(m_grid, kminus_vals, label="get_kminus")
    plt.ylim(0, 50)
    plt.xlabel("m")
    plt.ylabel("value")
    plt.title(f"ExactCachedHedgedCapitalCI at t={t_plot}")
    plt.legend()
    plt.show()

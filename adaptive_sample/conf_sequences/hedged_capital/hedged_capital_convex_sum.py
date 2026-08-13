import logging
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from typing import Tuple

import numpy as np
from scipy.special import gamma
from tqdm import tqdm

from adaptive_sample.conf_sequences import BaseConfidenceSequence, MultiDimensionalConfidenceSequence
from adaptive_sample.conf_sequences.conf_sphere.conf_sphere_utils import compute_ellips_bbx_volume
from adaptive_sample.conf_sequences.hedged_capital.f_approximation import ParabolaApproximation, parabola
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_1d import HedgedCapitalCI
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_utils import find_minimum_of_k
from adaptive_sample.conf_sequences.hedged_capital.hedged_factory import HedgedFactory
from adaptive_sample.helper.ellipsoid_tools import EllipsoidTools
from adaptive_sample.helper.grid_cs_helper import GridCSHelper
from adaptive_sample.helper.hypercube import HypercubeGrid
from adaptive_sample.helper.hypercube_tools import HypercubeTools
from adaptive_sample.helper.volume import VolumeCalculator
from adaptive_sample.plotting import Plotter
from adaptive_sample.shared_code import DataDistributionChecker

logger = logging.getLogger(__name__)


def _portfolio_weights(h: dict, D: int) -> np.ndarray:
    """Return a validated, fixed portfolio-weight vector."""
    raw_weights = h.get("portfolio_weights")
    if raw_weights is None:
        return np.full(D, 1.0 / D, dtype=float)

    weights = np.asarray(raw_weights, dtype=float)
    if weights.shape != (D,):
        raise ValueError(
            f"portfolio_weights must have shape ({D},), got {weights.shape}."
        )
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("portfolio_weights must be finite and strictly positive.")
    if not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(
            f"portfolio_weights must sum to one, got {float(np.sum(weights)):.12g}."
        )
    return weights.copy()


def _safe_positive_product(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(values < 0.0):
        raise ValueError("_safe_positive_product expects non-negative inputs.")
    if np.any(values == 0.0):
        return 0.0

    log_prod = float(np.sum(np.log(values)))
    if log_prod < math.log(np.nextafter(0.0, 1.0)):
        return 0.0
    return float(math.exp(log_prod))


def plot_1d_fnc(h: dict, t, data, d, f, a_star, b_star, c_star, alpha,
                x1_touch, x2_touch):
    if h['plot_grid_2d']:  # plot parabola fit
        running_mean1 = np.mean(data[: t + 1, d])
        running_mean2 = np.mean(data[: t, d] if t > 0 else np.nan)
        Plotter.plot_1d_function(f=[f,
                                    lambda x: parabola(x, a_star, b_star, c_star, pow=h['parabola_power']),
                                    ],
                                 x_min=0.0, x_max=1.0,
                                 # x_min=x1_start, x_max=x2_end,
                                 x_coords=[x1_touch, x2_touch],
                                 title=f"Hedged Capital CI at t={t} - mu_bar1={running_mean1:.10f}, mu_bar2={running_mean2:.10f}",
                                 ylim=(1 / alpha) + 5)


def _ellipse_params_from_hparams(
        C: float,
        hparams: np.ndarray,
        weights: np.ndarray,
        pow: int,
) -> np.ndarray:
    """Convert per-axis lower fits into the weighted l_p ellipsoid."""
    a_vals = np.asarray(hparams[:, 0], dtype=float)
    b_vals = np.asarray(hparams[:, 1], dtype=float)
    c_vals = np.asarray(hparams[:, 2], dtype=float)
    weighted_budget = float(C + np.dot(weights, c_vals))
    if not np.isfinite(weighted_budget) or weighted_budget <= 0.0:
        return np.column_stack((a_vals, np.full_like(b_vals, np.nan)))
    radii = b_vals * (weighted_budget / weights) ** (1.0 / pow)
    return np.column_stack((a_vals, radii))


def _ellipsoid_volume_from_hparams(C, D, hparams, pow: int, weights=None):
    # convert to ellipse params
    if weights is None:
        weights = np.full(D, 1.0 / D, dtype=float)
    ellipse_params = _ellipse_params_from_hparams(
        C, hparams, np.asarray(weights, dtype=float), pow
    )
    # extract b_i (radii)
    b_vals = np.array([b for (_, b) in ellipse_params], dtype=float)
    # any invalid radii -> infeasible candidate
    if np.any(~np.isfinite(b_vals)) or np.any(b_vals <= 0.0):
        return None, None
    # volume of axis-aligned ellipsoid: unit-ball-volume * prod(b_i)
    unit_ball_vol = (math.pi ** (D / 2.0)) / gamma(D / 2.0 + 1.0)
    radii_prod = _safe_positive_product(b_vals)
    vol = unit_ball_vol * radii_prod
    return vol, ellipse_params


def perform_volume_sanity_checks_if_requested(t, h, D, q, Q_inv, pow: int, lower, upper, vol_analytical: float,
                                              std_analytical: float):
    if h['perform_volume_sanity_checks']:
        is_in_cs: Callable[[np.ndarray], np.ndarray] = lambda m: (
            # (m - q).T @ Q_inv @ (m - q) <= 1.0 and  # check if in ellipsoid
                float(np.sum(((m - q) ** pow) * np.diag(Q_inv))) <= 1.0 and
                np.all(m >= 0.0) and np.all(m <= 1.0))  # check if in [0,1]^D

        vol_mc, std_mc = VolumeCalculator.monte_carlo_cs_volume(
            h, is_in_cs, D,
            bbx_lower=np.maximum(lower, 0.0),
            bbx_upper=np.minimum(upper, 1.0))

        if not np.isclose(vol_analytical, vol_mc, rtol=0.1):
            print("*" * 20)
            print(f"Warning: Ellipsoid volume sanity check failed at t={t}: "
                  f"analytic vol={vol_analytical:.6g}+-{std_analytical:.6g} vs MC vol={vol_mc:.6g}+/-{std_mc:.6g}")


class HedgedCapitalConvexSum(BaseConfidenceSequence, MultiDimensionalConfidenceSequence, ABC):
    """Base class that centralizes common logic for the different
    convex-sum confidence set approximations.

    Subclasses implement `_bounds_at_t(t, Kd_calcs, C)` which should return
    `(low: np.ndarray[D], up: np.ndarray[D], extra: dict)` where `extra` may
    contain auxiliary data used by plotting or volume-collection.

    Subclasses may also override `plot_grid` to visualize results per-t.
    """

    def __init__(self, h: dict, data):
        super().__init__(h, data)
        self._kd_calcs_cache = None
        self.weights = _portfolio_weights(h, self.D)

    def update(self, k: int, batch: np.ndarray):
        raise NotImplementedError("This method should not be called directly. Use `run` instead.")

    # -------------------- shared utilities --------------------
    def K_convex_combination(
            self,
            m_vec: List[float],
            Kd_calculators: List[HedgedCapitalCI],
            t: int,
    ) -> float:
        wealths = np.asarray(
            [calc.get_wealth(t, float(m)) for calc, m in zip(Kd_calculators, m_vec)],
            dtype=float,
        )
        return float(np.dot(self.weights, wealths))

    def _finalize_and_store(self, lowers: np.ndarray, uppers: np.ndarray, volumes: Optional[np.ndarray],
                            volume_stds: Optional[np.ndarray]):
        self.lowers = lowers
        self.uppers = uppers
        if self.h['calculate_volume']:
            self.volumes = volumes
            self.volume_stds = volume_stds
        else:
            self.volumes = np.zeros(len(self.ts))
            self.volume_stds = np.zeros(len(self.ts))

    # -------------------- template runner --------------------
    def run(self):
        """Generic runner. Subclasses should implement `_bounds_at_t` and
        (optionally) `plot_grid`.

        Returns (ts, means, lowers, uppers) to keep API compatibility.
        """
        data = self.data

        n, D = data.shape
        B = self.batch_size
        T = n // B
        C = 1.0 / self.alpha

        Kd_calcs = HedgedFactory.build_and_cache_Khedgeds(self.h, data)

        lowers = np.zeros((T, D))
        uppers = np.zeros((T, D))

        volumes = np.zeros(T)
        volume_stds = np.zeros(T)

        for t in tqdm(range(T),
                      desc=f"{self.__class__.__name__} (T={T}, D={D})",
                      disable=not self.h['verbose_progress']):
            low_t, up_t, extra = self._bounds_at_t(t, Kd_calcs, C)

            lowers[t, :] = low_t
            uppers[t, :] = up_t

            if self.h['calculate_volume']:
                volumes[t], volume_stds[t] = self.compute_volume(t, Kd_calcs, low_t, up_t, extra)

            # delegate plotting / visualization to subclass
            try:
                self.plot_grid(t, Kd_calcs, low_t, up_t, extra)
            except Exception as e:
                # plotting should never break the run
                logger.debug("plot_grid failed for t=%s: %s", t, e)

        self.means = self._extract_means(self.h, data, self.N, self.batch_size)
        self._finalize_and_store(lowers, uppers, volumes, volume_stds)
        return self.ts, self.means, self.lowers, self.uppers, self.volumes, self.volume_stds

    # -------------------- methods subclasses must/ may override --------------------
    @abstractmethod
    def _bounds_at_t(self, t: int, Kd_calcs: List[HedgedCapitalCI], C: float) -> Tuple[
        np.ndarray, np.ndarray, Dict[str, Any]]:
        """Return (low[D], up[D], extra_dict).

        extra_dict may contain keys used by plot_grid or for volume collection.
        """
        raise NotImplementedError

    def compute_volume_at(self, t: int) -> Tuple[float, float]:
        """Compute volume at batch-index t without requiring run() to have been called."""
        # build calculators from data (safe to call repeatedly; HedgedFactory may cache internally)
        Kd_calcs = HedgedFactory.build_and_cache_Khedgeds(self.h, self.data)
        C = 1.0 / self.alpha

        low, up, extra = self._bounds_at_t(t, Kd_calcs, C)
        if self.h['plot_grid_2d'] and DataDistributionChecker.dim(self.h) <= 3:
            self.plot_grid(t, Kd_calcs, None, None, extra)
        vol, std = self.compute_volume(t, Kd_calcs, low, up, extra)
        return float(vol), float(std)

    def plot_grid(self, t: int, Kd_calcs, low_t: np.ndarray, up_t: np.ndarray, extra: dict):
        if not self.h['plot_grid_2d']:
            return

        ellipse_params = extra.get('ellipse_params', None)  # for Ellip, BBxEllip
        proj_lowers = extra.get('proj_lowers', None)  # for BBxEllip, BBx
        proj_uppers = extra.get('proj_uppers', None)

        is_in_cs_func = lambda m: self.K_convex_combination(m, Kd_calcs, t) < (1.0 / self.alpha)
        grid, in_cs = GridCSHelper.confidence_set(
            self.h,
            is_in_cs_func,
            D=self.D,
            grid_resolution=self.h['grid_resolution']
        )

        if ellipse_params is not None:
            a = ellipse_params[0, 0]
            b = ellipse_params[0, 1]
            a_prime = ellipse_params[1, 0]
            b_prime = ellipse_params[1, 1]
        else:
            a = b = a_prime = b_prime = None

        if proj_lowers is not None:
            low = np.atleast_2d(proj_lowers)
            up = np.atleast_2d(proj_uppers)
        else:
            low = up = []

        if self.D == 2:
            if t <= 3:
                return
            pow = self.h['parabola_power']

            f_tilde = lambda x1, x2: (
                    (np.abs(x1 - a) ** pow) / (b ** pow) +
                    (np.abs(x2 - a_prime) ** pow) / (b_prime ** pow)
            ) if (b is not None and b_prime is not None) else float('inf')

            Plotter.GridPlotter.display_2d_confidence_region(
                f_tilde=f_tilde,
                C=1,
                low=low, up=up,
                title=f"Hedged Capital CI at t={t + 1}",
                grid=grid, in_cs=in_cs,
                # domain_start=0.8, domain_end=1.0
                # domain_start=0.35, domain_end=0.6
            )


        elif self.D == 3:
            if t <= 5 or h['parabola_power'] != 2:
                return

            Plotter.GridPlotter.display_3d_confidence_region(
                ellipse_params=ellipse_params,
                low=low, up=up,
                title=f"Hedged Capital CI at t={t + 1}",
                grid=grid, in_cs=in_cs, max_points=500,
                domain_start=0.3, domain_end=0.6
            )
        else:
            # no plotting for other dimensions
            return

    def is_member_at(self, x, t: int) -> bool:
        """Check if x is in the confidence set at 0-based batch index t."""
        n_samples, D = self.data.shape
        B = self.batch_size
        T = n_samples // B
        if not (0 <= t < T):
            raise IndexError(f"t={t} out of range [0, {T - 1}]")

        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = np.array([float(x)])
        if x.shape[-1] != D:
            raise ValueError(f"Expected x with last dimension {D}, got {x.shape}.")

        if np.any(x < 0.0) or np.any(x > 1.0):
            return False

        if self.h.get('cut_to_simplex', False) and D > 1:
            if not np.isclose(np.sum(x), 1.0, atol=1e-8):
                return False

        if self._kd_calcs_cache is None:
            self._kd_calcs_cache = HedgedFactory.build_and_cache_Khedgeds(self.h, self.data)

        C = 1.0 / self.alpha
        return self.K_convex_combination(x, self._kd_calcs_cache, t) < C


# -------------------- Grid-based approach --------------------
class HedgedCapitalConvexSumGrid(HedgedCapitalConvexSum):
    @staticmethod
    def _log_wealth_many(calc, t: int, candidates: np.ndarray, chunk_size: int = 256) -> np.ndarray:
        """Exact cached log-wealth evaluation for many candidate means."""
        if not all(hasattr(calc, attr) for attr in ("_advance_from_last_to", "_xs", "_lams")):
            return np.asarray([calc.get_log_wealth(t, float(m)) for m in candidates], dtype=float)

        calc._advance_from_last_to(t)
        n_obs = t * calc.batch_size
        if n_obs == 0:
            return np.full(
                len(candidates),
                math.log(max(calc.theta, 1.0 - calc.theta)),
                dtype=float,
            )

        x_arr = calc._xs[:n_obs, None]
        lam_arr = calc._lams[:n_obs, None]
        result = np.empty(len(candidates), dtype=float)
        eps = 1e-12
        for start in range(0, len(candidates), chunk_size):
            stop = min(start + chunk_size, len(candidates))
            m = np.asarray(candidates[start:stop], dtype=float)[None, :]
            lam_plus = np.minimum(lam_arr, calc.c / (m + eps))
            lam_minus = np.minimum(lam_arr, calc.c / (1.0 - m + eps))
            factors_plus = 1.0 + lam_plus * (x_arr - m)
            factors_minus = 1.0 - lam_minus * (x_arr - m)
            log_plus = np.sum(
                np.where(factors_plus > 0.0, np.log(factors_plus), -np.inf),
                axis=0,
            )
            log_minus = np.sum(
                np.where(factors_minus > 0.0, np.log(factors_minus), -np.inf),
                axis=0,
            )
            result[start:stop] = np.maximum(
                math.log(calc.theta) + log_plus,
                math.log(1.0 - calc.theta) + log_minus,
            )
        return result

    def _vectorized_volume(
            self,
            t: int,
            Kd_calcs: List[HedgedCapitalCI],
            lowers: np.ndarray,
            uppers: np.ndarray,
    ) -> Tuple[float, float]:
        """Monte Carlo portfolio volume with batched exact wealth evaluation."""
        n_samples = int(self.h.get("volume_mc_samples", 25_000))
        num_repeats = int(self.h.get("volume_mc_repeats", 5))
        base_seed = self.h.get("volume_mc_seed")
        chunk_size = int(self.h.get("portfolio_mc_chunk_size", 256))
        if n_samples <= 0 or num_repeats <= 0 or chunk_size <= 0:
            raise ValueError("Portfolio Monte Carlo settings must be positive.")

        box_volume = VolumeCalculator.safe_positive_product(uppers - lowers)
        if box_volume == 0.0:
            return 0.0, 0.0

        log_threshold = math.log(1.0 / self.alpha)
        volumes = []
        for repeat_idx in range(num_repeats):
            rng = np.random.default_rng(
                None if base_seed is None else int(base_seed) + repeat_idx
            )
            samples = rng.random(size=(n_samples, self.D))
            samples = samples * (uppers - lowers) + lowers

            log_sum = None
            for d, calc in enumerate(Kd_calcs):
                log_wealth = self._log_wealth_many(
                    calc,
                    t,
                    samples[:, d],
                    chunk_size=chunk_size,
                )
                weighted_log_wealth = math.log(self.weights[d]) + log_wealth
                log_sum = (
                    weighted_log_wealth
                    if log_sum is None
                    else np.logaddexp(log_sum, weighted_log_wealth)
                )
            log_portfolio = log_sum
            volumes.append(float(np.mean(log_portfolio < log_threshold)) * box_volume)
        return float(np.mean(volumes)), float(np.std(volumes))

    def _bounds_at_t(self, t: int, Kd_calcs: List[HedgedCapitalCI], C: float):
        D = len(Kd_calcs)

        is_in_cs_func = lambda m: self.K_convex_combination(m, Kd_calcs, t) < C

        grid, in_cs = GridCSHelper.confidence_set(
            self.h,
            is_in_cs_func,
            D=D,
            grid_resolution=self.h['grid_resolution']
        )

        lowers = np.zeros(D)
        uppers = np.zeros(D)

        if np.any(in_cs):
            # grid[in_cs] has shape (N_in, D)
            selected = grid[in_cs]
            for d in range(D):
                lowers[d] = np.min(selected[:, d])
                uppers[d] = np.max(selected[:, d])
        else:
            lowers[:] = 0.0
            uppers[:] = 1.0

        extra = {
            'grid': grid,
            'in_cs': in_cs,
        }
        return lowers, uppers, extra

    def compute_volume(self, t, Kd_calcs, _, __, extra: dict) -> Tuple[float, float]:
        is_in_cs_func = lambda m: (self.K_convex_combination(m, Kd_calcs, t) < (1.0 / self.alpha) and
                                   np.all(m >= 0.0) and np.all(m <= 1.0))

        # only used here to allow for mc volume calc that is more efficient in higher dimensions
        lowers, uppers = HedgedCapitalConvexSumBoundingBox._projection_hypercube_single(
            self.D, Kd_calcs, 1.0 / self.alpha, t, weights=self.weights)

        if self.h.get("vectorized_portfolio_volume", False) and not self.h["cut_to_simplex"]:
            return self._vectorized_volume(t, Kd_calcs, lowers, uppers)

        vol, std = VolumeCalculator.monte_carlo_cs_volume(self.h, is_in_cs_func, D=self.D,
                                                          bbx_lower=lowers,
                                                          bbx_upper=uppers)
        return vol, std

    def compute_volume_at(self, t: int) -> Tuple[float, float]:
        # avoid recomputing the grid
        Kd_calcs = HedgedFactory.build_and_cache_Khedgeds(self.h, self.data)
        C = 1.0 / self.alpha
        is_in_cs = lambda m: (self.K_convex_combination(m, Kd_calcs, t) < C and
                              np.all(m >= 0.0) and np.all(m <= 1.0))
        lowers, uppers = HedgedCapitalConvexSumBoundingBox._projection_hypercube_single(
            self.D, Kd_calcs, C, t, weights=self.weights)
        if self.h.get("vectorized_portfolio_volume", False) and not self.h["cut_to_simplex"]:
            return self._vectorized_volume(t, Kd_calcs, lowers, uppers)
        vol, std = VolumeCalculator.monte_carlo_cs_volume(self.h, is_in_cs, D=self.D,
                                                          bbx_lower=lowers,
                                                          bbx_upper=uppers)
        return float(vol), float(std)


# -------------------- Bounding-box projection approach --------------------
class HedgedCapitalConvexSumBoundingBox(HedgedCapitalConvexSum):
    @staticmethod
    def _weights(D: int, weights=None) -> np.ndarray:
        return (
            np.full(D, 1.0 / D, dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )

    @staticmethod
    def _f_star_per_coordinate(Kd_calcs: List[HedgedCapitalCI], t: int) -> np.ndarray:
        D = len(Kd_calcs)
        f_star = np.empty(D)
        for d, calc in enumerate(Kd_calcs):
            k_lower, _, _, _ = find_minimum_of_k(
                k_plus=lambda x: np.log(calc.theta) + calc.get_log_kplus(t, x),
                k_minus=lambda x: np.log(1 - calc.theta) + calc.get_log_kminus(t, x),
                a=0.0, b=1.0,
                log_scale=True,
                eps_x=1e-8,
                eps_y=1e-8,
                max_iter=100)
            f_star[d] = k_lower
        return f_star

    @staticmethod
    def _find_argmin(f, x_min, x_max, n=1001):
        raise NotImplementedError("THIS IS OUTDATED: USE find_minimum_of_k INSTEAD")

        xs = np.linspace(x_min, x_max, n)
        ys = np.array([f(x) for x in xs])
        idx = np.argmin(ys)
        return xs[idx], ys[idx]

    @staticmethod
    def _projection_hypercube_single(
            D: int,
            Kd_calcs: List[HedgedCapitalCI],
            C: float,
            t: int,
            weights=None,
    ):
        lowers = np.zeros(D)
        uppers = np.zeros(D)
        w = HedgedCapitalConvexSumBoundingBox._weights(D, weights)
        f_star = HedgedCapitalConvexSumBoundingBox._f_star_per_coordinate(Kd_calcs, t)

        total_min = float(np.dot(w, f_star))

        for d, calc in enumerate(Kd_calcs):
            gamma_d = (C - (total_min - w[d] * f_star[d])) / w[d]
            if gamma_d < f_star[d]:  # no solution
                lowers[d] = 0.0
                uppers[d] = 1.0
            else:
                lower_d, upper_d = calc.find_confidence_interval_1d(t, gamma_d)
                lowers[d] = lower_d
                uppers[d] = upper_d

        # assert between 0 and 1
        assert np.all(lowers >= 0.0) and np.all(lowers <= 1.0), f"lowers not in [0,1]: {lowers}"

        return lowers, uppers

    def _bounds_at_t(self, t: int, Kd_calcs: List[HedgedCapitalCI], C: float):
        D = len(Kd_calcs)
        lowers, uppers = self._projection_hypercube_single(
            D, Kd_calcs, C, t, weights=self.weights
        )

        if self.h['cut_to_simplex'] and D > 1:
            lowers, uppers = HypercubeTools.hypercube_and_simplex_intersection(lowers, uppers, D)

        extra = {}
        return lowers, uppers, extra

    def compute_volume(self, t, Kd_calcs, low: np.ndarray, up: np.ndarray, extra: dict) -> Tuple[float, float]:
        return VolumeCalculator.hypercube_volume(self.h, low, up), 0.0


# -------------------- Ellipsoid approximation approach --------------------
class HedgedCapitalConvexSumEllipsoid(HedgedCapitalConvexSum):
    def __init__(self, h: dict, data):
        super().__init__(h, data)

    @staticmethod
    def _min_smoothen(y: np.ndarray) -> np.ndarray:
        """Set y[i] = min(y[i], y[i-1], y[i+1])"""
        y_new = y.copy()
        y_new[1:-1] = np.minimum(y[1:-1], np.minimum(y[:-2], y[2:]))

        # first element: min of self and right neighbor
        y_new[0] = min(y[0], y[1])

        # last element: min of self and left neighbor
        y_new[-1] = min(y[-1], y[-2])
        return y_new

    def _compute_hyperbola_params(self, Kd_calcs, t: int, D: int, C: float):
        assert C >= 1.0, "C must be at least 1.0"

        pow = self.h['parabola_power']
        max_realistic_C = (C / float(np.min(self.weights))) + 1e-8

        def _compute_for_power(power: int, do_plot: bool):
            assert power > 0

            if self.h['parabola_C_touchpoint'] == 'automatic':
                div = (D ** (1.0 / power))
                touchpointC = C / div
            else:
                touchpointC = float(self.h['parabola_C_touchpoint'])

            hyperbola_params = np.empty((D, 3))
            for d, calc in enumerate(Kd_calcs):
                f = lambda m: calc.get_wealth(t=t, m=m)
                # domain:
                x1_start, x2_end = calc.find_confidence_interval_1d(t=t, target=max_realistic_C,
                                                                    actual_at_least_target=True)
                # touch points:
                x1_touch, x2_touch = calc.find_confidence_interval_1d(t=t, target=touchpointC,
                                                                      actual_at_least_target=True)

                temp = ParabolaApproximation.approximate_parabola(
                    self.h,
                    f,
                    x1_start, x2_end,  # defines domain
                    x1_touch, x2_touch,  # points where parabola must touch f
                    C=touchpointC,
                    method="D", options={
                        'parabola_power': power,
                        # used for optional conservative construction:
                        'k_plus': (lambda x: calc.theta * calc.get_kplus(t, x)),
                        'k_minus': (lambda x: (1 - calc.theta) * calc.get_kminus(t, x))
                    }
                )

                if temp is None:
                    # infeasible parabola approximation; fallback to full interval. Should only happen at t = 0.
                    a_star, b_star, c_star = 0.5, 0.5, 1.0  # touches at (0,0) and (1,0)
                    if t != 0:
                        return None
                else:
                    a_star, b_star, c_star = temp
                    if b_star == float('inf') or math.isnan(b_star):
                        return None

                hyperbola_params[d] = (a_star, b_star, c_star)

                if do_plot:
                    plot_1d_fnc(self.h, t, self.data, d, f, a_star, b_star, c_star, self.alpha,
                                x1_touch, x2_touch)

            return hyperbola_params

        if self.h['parabola_adaptive'] is False:
            return _compute_for_power(pow, do_plot=True)
        else:
            pow = 0  # placeholder

            best_params = None
            best_pow = None
            best_volume = float('inf')
            for candidate_pow in range(6, 17):
                hyperbola_params = _compute_for_power(candidate_pow, do_plot=False)
                if hyperbola_params is None:
                    continue

                ellipse_params = _ellipse_params_from_hparams(
                    C,
                    hyperbola_params,
                    self.weights,
                    candidate_pow,
                )
                b_vals = np.array([b for (_, b) in ellipse_params], dtype=float)
                if np.any(~np.isfinite(b_vals)) or np.any(b_vals <= 0.0):
                    continue

                q = np.array([a for (a, _) in ellipse_params], dtype=float)
                radii = b_vals
                Q = np.diag(b_vals ** candidate_pow)
                Q_inv = np.diag(b_vals ** (-candidate_pow))

                ellip_low = q - (np.diag(Q) ** (1.0 / candidate_pow))
                ellip_up = q + (np.diag(Q) ** (1.0 / candidate_pow))
                low = np.maximum(ellip_low, 0.0)
                up = np.minimum(ellip_up, 1.0)

                vol, _ = compute_ellips_bbx_volume(
                    self.h, D, q=q, Q=Q, Q_inv=Q_inv, radii=radii,
                    bbx_low=low, bbx_high=up, pow=candidate_pow
                )
                if vol < best_volume:
                    best_volume = vol
                    best_params = hyperbola_params
                    best_pow = candidate_pow

            ####

            self.h['parabola_power'] = int(best_pow)
            return _compute_for_power(int(best_pow), do_plot=True)

    def _compute_hyperbola_params_OPTIMIZE_C_BASED_ON_VOL(self, Kd_calcs, t: int, D: int, C: float):
        # C per dimension
        assert C >= 1.0, "C must be at least 1.0"

        # domain upper bound for realistic C used for interval querying
        max_realistic_C = (C / float(np.min(self.weights))) + 1e-8

        touchpoint_candidates = np.linspace(1, max_realistic_C, num=100)

        best_volume = float('inf')
        best_hyperbola_params = None
        best_touchpointC = None

        # iterate candidate touchpointC values
        for tpC in touchpoint_candidates:
            hyperbola_params = np.empty((D, 3))
            feasible = True

            for d, calc in enumerate(Kd_calcs):
                f = lambda m: calc.get_wealth(t=t, m=m)

                # domain endpoints for per-d approximation (same as original)
                try:
                    x1_start, x2_end = calc.find_confidence_interval_1d(
                        t=t, target=max_realistic_C, actual_at_least_target=True
                    )
                    # touching points w.r.t. the original target C (the target level in your algorithm)
                    x1_touch, x2_touch = calc.find_confidence_interval_1d(
                        t=t, target=tpC, actual_at_least_target=True
                    )
                except Exception as e:
                    # if interface raises or returns invalid intervals, mark infeasible
                    feasible = False
                    break

                # attempt parabola approximation using current tpC
                temp = ParabolaApproximation.approximate_parabola(
                    self.h,
                    f,
                    x1_start, x2_end,
                    x1_touch, x2_touch,
                    C=tpC,
                    method="D", options={
                        'parabola_power': self.h['parabola_power'],
                        'k_plus': (lambda x: calc.theta * calc.get_kplus(t, x)),
                        'k_minus': (lambda x: (1 - calc.theta) * calc.get_kminus(t, x))
                    }
                )

                if temp is None:
                    feasible = False
                    break

                a_star, b_star, c_star = temp
                if b_star == float('inf') or math.isnan(b_star) or not np.isfinite(b_star) or b_star <= 0.0:
                    feasible = False
                    break

                hyperbola_params[d] = (a_star, b_star, c_star)

            if not feasible:
                # skip this tpC candidate (some dimension had infeasible parabola)
                continue

            # compute ellipsoid volume from this candidate hyperbola params
            vol, ellipse_params = _ellipsoid_volume_from_hparams(
                C,
                D,
                hyperbola_params,
                self.h['parabola_power'],
                weights=self.weights,
            )
            # print(f"Tried touchpointC={tpC:.6g} at t={t} -> volume={vol:.6g}")
            if vol is None:  # invalid radii -> skip
                continue

            # keep candidate with smallest volume
            if vol < best_volume:
                best_volume = vol
                best_hyperbola_params = hyperbola_params.copy()
                best_touchpointC = float(tpC)
                best_ellipse_params = ellipse_params

        # If we found at least one feasible candidate, use it. Otherwise fall back to original single-C
        if best_hyperbola_params is not None:
            touchpointC = best_touchpointC
            hyperbola_params = best_hyperbola_params
            for d, calc in enumerate(Kd_calcs):
                f = lambda m: calc.get_wealth(t=t, m=m)
                x1_touch, x2_touch = calc.find_confidence_interval_1d(
                    t=t, target=touchpointC, actual_at_least_target=True
                )
                a_star, b_star, c_star = hyperbola_params[d]
                plot_1d_fnc(self.h, t, self.data, d, f, a_star, b_star, c_star, self.alpha,
                            x1_touch, x2_touch)

            if self.h.get('verbose_progress', False):
                print(f"[ellip] Chosen touchpointC={touchpointC:.6g} at t={t} (volume={best_volume:.6g})")
        else:
            # fallback: try original single tpC == C behaviour (exactly as original code)
            touchpointC = C
            if self.h.get('verbose_progress', False):
                print(f"[ellip] No feasible grid candidate found, falling back to touchpointC={touchpointC} at t={t}")
            hyperbola_params = np.empty((D, 3))
            for d, calc in enumerate(Kd_calcs):
                f = lambda m: calc.get_wealth(t=t, m=m)
                x1_start, x2_end = calc.find_confidence_interval_1d(t=t, target=max_realistic_C,
                                                                    actual_at_least_target=True)
                x1_touch, x2_touch = calc.find_confidence_interval_1d(t=t, target=C, actual_at_least_target=True)
                temp = ParabolaApproximation.approximate_parabola(
                    self.h,
                    f,
                    x1_start, x2_end,
                    x1_touch, x2_touch,
                    C=touchpointC,
                    method="D", options={
                        'parabola_power': self.h['parabola_power'],
                        'k_plus': (lambda x: calc.theta * calc.get_kplus(t, x)),
                        'k_minus': (lambda x: (1 - calc.theta) * calc.get_kminus(t, x))
                    }
                )
                if temp is None:
                    # infeasible parabola approximation; fallback safe default
                    a_star, b_star, c_star = 0.5, 0.5, 1.0
                    assert t == 0, "infeasible parabola should only happen at t=0"
                else:
                    a_star, b_star, c_star = temp
                hyperbola_params[d] = (a_star, b_star, c_star)
                plot_1d_fnc(self.h, t, self.data, d, f, a_star, b_star, c_star, self.alpha,
                            x1_touch, x2_touch)

        print(f"[ellip] Using touchpointC={touchpointC:.6g} at t={t} --- D={D}")

        return hyperbola_params

    def _bounds_at_t(self, t: int, Kd_calcs: List[HedgedCapitalCI], C: float):
        D = len(Kd_calcs)

        hyperbola_params = self._compute_hyperbola_params(Kd_calcs, t, D, C)

        ellipse_params = _ellipse_params_from_hparams(
            C,
            hyperbola_params,
            self.weights,
            self.h['parabola_power'],
        )  # ellipse_params[d] = (a_d, b_d) with center a_d and radius b_d

        ellipsoid_lowers = ellipse_params[:, 0] - ellipse_params[:, 1]
        ellipsoid_uppers = ellipse_params[:, 0] + ellipse_params[:, 1]

        # clamp to [0, 1]
        lowers = np.maximum(ellipsoid_lowers, 0.0)
        uppers = np.minimum(ellipsoid_uppers, 1.0)

        q = np.array([a for (a, b) in ellipse_params])
        Q = np.diag([b ** self.h['parabola_power'] for (a, b) in ellipse_params])

        if self.h['cut_to_simplex'] and D > 1:
            c = np.ones(D)
            gamma = 1
            w, W = EllipsoidTools.intersection_ellipsoid_hyperplane(q, Q, c, gamma, D)
            evals, evecs = EllipsoidTools.extract_non_zero_eigens(W)
            bounding_box = EllipsoidTools.compute_bounding_box(w, evals, evecs, D)

            for d_idx in range(D):
                ellipse_intersec_lower, ellipse_intersec_upper = bounding_box[d_idx]
                lowers[d_idx] = np.maximum(lowers[d_idx], ellipse_intersec_lower)
                uppers[d_idx] = np.minimum(uppers[d_idx], ellipse_intersec_upper)

        extra = {
            'ellipse_params': ellipse_params,
            'ellipse_params_mtx': (q, Q),
            'radii': np.asarray([b for (a, b) in ellipse_params], dtype=float)
        }

        return lowers, uppers, extra

    def compute_volume(self, t, Kd_calcs, low: np.ndarray[float], up: np.ndarray[float], extra: dict) -> Tuple[
        float, float]:
        if self.h['calculate_volume']:
            q, Q = extra['ellipse_params_mtx']
            Q_inv = np.linalg.inv(Q)  # shape (D, D)
            radii = extra['radii']  # shape (D,)

            assert np.allclose(Q_inv @ Q, np.eye(self.D)), "Q_inv @ Q is not close to the identity matrix"

            pow = self.h['parabola_power']
            # get bbx of ellipsoid
            # lower = q - np.sqrt(np.diag(Q))
            # upper = q + np.sqrt(np.diag(Q))  # bounding box of ellipsoid

            # get bbx of power ellipsoid
            ellip_low = q - (np.diag(Q) ** (1.0 / pow))  # shape (D,)
            ellip_up = q + (np.diag(Q) ** (1.0 / pow))  # bounding box of ellipsoid

            low = np.maximum(np.maximum(ellip_low, low), np.zeros_like(low))
            up = np.minimum(np.minimum(ellip_up, up), np.ones_like(up))

            vol, std = compute_ellips_bbx_volume(
                self.h, self.D, q=q, Q=Q, Q_inv=Q_inv, radii=radii,
                bbx_low=low, bbx_high=up, pow=pow)

            # # also do MC volume calc for comparison and check if approximately the same
            # perform_volume_sanity_checks_if_requested(t, self.h, self.D, q, Q_inv, pow, low, up, vol, std)

            return vol, std

        else:
            return 0.0, 0.0

    @staticmethod
    def ellip_targ(alpha, D):
        epsilon = 1e-8
        return ((1.0 / alpha) * D) + epsilon


# -------------------- Bounding-box X Ellipsoid (combined) --------------------
class HedgedCapitalConvexSumBBxEllip(HedgedCapitalConvexSumBoundingBox, HedgedCapitalConvexSumEllipsoid):
    def __init__(self, h: dict, data):
        super().__init__(h, data)

    def _bounds_at_t(self, t: int, Kd_calcs: List[HedgedCapitalCI], C: float):
        D = len(Kd_calcs)

        # 1) compute hyperbola params per-d

        hyperbola_params = self._compute_hyperbola_params(Kd_calcs, t, D, C)
        ellipse_params = _ellipse_params_from_hparams(
            C,
            hyperbola_params,
            self.weights,
            self.h['parabola_power'],
        )  # ellipse_params[d] = (a_d, b_d) with center a_d and radius b_d

        ellipsoid_lowers = ellipse_params[:, 0] - ellipse_params[:, 1]
        ellipsoid_uppers = ellipse_params[:, 0] + ellipse_params[:, 1]

        proj_lowers, proj_uppers = self._projection_hypercube_single(
            D, Kd_calcs, C, t, weights=self.weights
        )

        lowers = np.zeros(D)
        uppers = np.zeros(D)
        for d in range(D):
            mx = np.max([ellipsoid_lowers[d], proj_lowers[d], 0.0])
            mn = np.min([ellipsoid_uppers[d], proj_uppers[d], 1.0])
            if mx > mn:
                # fallback to full interval if numerical issue
                lowers[d], uppers[d] = 0.0, 1.0
            else:
                lowers[d], uppers[d] = mx, mn

        q = np.array([a for (a, b) in ellipse_params])
        Q = np.diag([b ** self.h['parabola_power'] for (a, b) in ellipse_params])

        # optional simplex intersections for both hypercube and ellipsoid
        if self.h['cut_to_simplex'] and D > 1:
            # hypercube part
            lowers_proj_inters, uppers_proj_inters = HypercubeTools.hypercube_and_simplex_intersection(proj_lowers,
                                                                                                       proj_uppers, D)
            lowers = np.maximum(lowers, lowers_proj_inters)
            uppers = np.minimum(uppers, uppers_proj_inters)

            # ellipsoid intersection bounding box
            c = np.ones(D)
            gamma = 1
            w, W = EllipsoidTools.intersection_ellipsoid_hyperplane(q, Q, c, gamma, D)
            evals, evecs = EllipsoidTools.extract_non_zero_eigens(W)
            bounding_box = EllipsoidTools.compute_bounding_box(w, evals, evecs, D)
            for d_idx in range(D):
                ellipse_intersec_lower, ellipse_intersec_upper = bounding_box[d_idx]
                lowers[d_idx] = np.maximum(lowers[d_idx], ellipse_intersec_lower)
                uppers[d_idx] = np.minimum(uppers[d_idx], ellipse_intersec_upper)

        extra = {'ellipse_params': ellipse_params,
                 'ellipse_params_mtx': (q, Q),
                 'radii': np.asarray([b for (a, b) in ellipse_params], dtype=float),
                 'proj_lowers': proj_lowers,
                 'proj_uppers': proj_uppers}

        # check if ellipse_params contains inf or nan
        if np.any(np.isinf(ellipse_params)) or np.any(np.isnan(ellipse_params)):
            print(f"Warning: ellipse_params contains inf or nan at t={t}: {ellipse_params}")

        return lowers, uppers, extra

    def compute_volume(self, t, Kd_calcs, low: np.ndarray[float], up: np.ndarray[float], extra: dict) -> Tuple[
        float, float]:
        """Estimate volume of intersection between the ellipsoid and the projection hypercube
        (and optionally the simplex) using Monte Carlo. Returns 0.0 when volume calculation is disabled.
        """
        if not self.h['calculate_volume']:
            return 0.0, 0.0

        # get ellipse matrix params
        q, Q = extra['ellipse_params_mtx']  # q shape (D,), Q shape (D,D) diagonal(b^2)
        # use pseudo-inverse in case Q is singular (robust to degenerate ellipsoid axes)
        Q_inv = np.linalg.inv(Q)  # should be invertible because b > 0 always
        radii = extra['radii']  # shape (D,)

        assert np.allclose(Q_inv @ Q, np.eye(self.D)), "Q_inv @ Q is not close to the identity matrix"

        # projection hypercube bounds from extra
        proj_lowers = extra['proj_lowers']
        proj_uppers = extra['proj_uppers']
        low = np.maximum(np.maximum(low, proj_lowers), np.zeros_like(low))
        up = np.minimum(np.minimum(up, proj_uppers), np.ones_like(up))

        pow = self.h['parabola_power']
        vol, std = compute_ellips_bbx_volume(
            self.h, self.D, q=q, Q=Q, Q_inv=Q_inv, radii=radii,
            bbx_low=low, bbx_high=up, pow=pow
        )

        return vol, std


if __name__ == "__main__":
    # for dataa in ['ellipse_2d', 'circle_2d', 'fixed_2d']:
    # fix seed

    np.random.seed(42)

    data_type = 'fixed_2d'
    h = {
        'alpha': 0.05,
        # 'alpha': 0.01,
        'batch_size': 1,
        'data_type': data_type,  # fixed_2d | circle_2d | ellipse_2d | uniform_2d
        'N': 100,
        'cut_to_simplex': False,
        'grid_resolution': 500 if data_type.endswith('_2d') else 100,
        'calculate_volume': True,
        'save_plots_locally': False,
        'use_wandb': False,
        'verbose_progress': True,
        'cache_hedged': 'EXACT',  # 'EXACT' | None | 'APPROX'
        'plot_grid_2d': True,
        'parabola_power': 4,
        'extend_f_domain_for_parabola_fit': True,
    }

    # print(f"trying C={MY_C_TOUCHPOINT}")

    data, true_mean = DataDistributionChecker.get_data(h['data_type'], h['N'])  # or "fixed" for constant data

    # data, true_mean = 0.98 * np.ones((h['N'], 2)), np.array([1.0, 1.0])
    # data, true_mean = 0.02 * np.ones((h['N'], 2)), np.array([1.0, 1.0])

    # cs = HedgedCapitalConvexSumBBxEllip(h, data)
    # vol_ell_bbx, std_ell_bbx = cs.compute_volume_at(min(100, h['N'] - 1))
    # print(f"{vol_ell_bbx=} += {std_ell_bbx}")

    cs_ellip = HedgedCapitalConvexSumEllipsoid(h, data)
    # cs_ellip.run()
    vol_ell, std_ell = cs_ellip.compute_volume_at(5)  # min(100, h['N'] - 1))
    print(f"{vol_ell=} += {std_ell}")

    # cs_bonf = HedgedCapitalConvexSumBoundingBox(h, data)
    # vol_bonf, std_bonf = cs_bonf.compute_volume_at(min(100, h['N'] - 1))
    # print(f"{vol_bonf=}")
    #
    # cs_grid = HedgedCapitalConvexSumGrid(h, data)
    # vol_grid, std_grid = cs_grid.compute_volume_at(min(100, h['N'] - 1))
    # print(f"{vol_grid=}")

    # print(f"{vol_bonf/vol_bonf=:6f} \n"
    #       f"{vol_ell/vol_bonf=:6f} \n"
    #       f"{vol_ell_bbx / vol_bonf=:6f} \n"
    #       f"{vol_grid/vol_bonf=:6f}"
    #       )

    ###

import math
from typing import Tuple, Optional, Dict, Any

import numpy as np

from adaptive_sample.conf_sequences import (
    BaseConfidenceSequence,
    MultiDimensionalConfidenceSequence,
)
from adaptive_sample.conf_sequences.conf_sphere.conf_sphere_utils import (
    rescale_data,
    rescale_data_optimistic,
)
from adaptive_sample.helper.volume import VolumeCalculator
from adaptive_sample.shared_code import DataDistributionChecker


def spectral_clip(V: np.ndarray, rho: float) -> np.ndarray:
    """Return V ∨ rho I by clipping eigenvalues below rho."""
    evals, evecs = np.linalg.eigh(V)
    evals = np.maximum(evals, rho)
    return evecs @ np.diag(evals) @ evecs.T


def gamma_min(V: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(V)))


def gamma_max(V: np.ndarray) -> float:
    return float(np.max(np.linalg.eigvalsh(V)))


def condition_number(V: np.ndarray) -> float:
    evals = np.linalg.eigvalsh(V)
    return float(np.max(evals) / np.min(evals))


def h_poly_continuous(x: float, s: float = 2.0) -> float:
    """
    Continuous extension of h(k) = zeta(s) (k+1)^s.
    We instantiate the paper's suggested family with s=2 for simplicity.
    """
    if s != 2.0:
        raise ValueError("Only s=2.0 implemented in this helper.")
    return (np.pi ** 2 / 6.0) * (x + 1.0) ** s


def L_cov_delta(
    V: np.ndarray,
    delta: float,
    rho: float,
    alpha: float = 1.05,
    beta: float = 2.0,
    eps: float = 0.5,
    h_fn=h_poly_continuous,
    C_d: float = 1.0,
) -> float:
    """
    Practical simplified log term based on Corollary 4.3:
      L_cov(V) = log h(log_alpha(gmax/rho))
               + log(C_d / (delta * (1 - 1/beta)))
               + d * log(3 * beta * sqrt(kappa) / eps)
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    if not (alpha > 1):
        raise ValueError("alpha must be > 1")
    if not (beta > 1):
        raise ValueError("beta must be > 1")
    if not (0 < eps < 1):
        raise ValueError("eps must be in (0,1)")

    Vclip = spectral_clip(V, rho)
    gmax = gamma_max(Vclip)
    kappa = condition_number(Vclip)

    epoch = np.log(gmax / rho) / np.log(alpha)  # log_alpha(gmax/rho)
    return float(
        np.log(h_fn(epoch))
        + np.log(C_d / (delta * (1.0 - 1.0 / beta)))
        + V.shape[0] * np.log(3.0 * beta * np.sqrt(kappa) / eps)
    )


def omegaE1_star(u: float) -> float:
    """Convex conjugate of omega_E,1(lambda) = -log(1-lambda) - lambda."""
    u = np.asarray(u)
    return u - np.log1p(u)


def inv_omegaE1_star(y: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Solve omega_E,1^*(u) = y for u >= 0 by bisection."""
    if y <= 0:
        return 0.0

    lo, hi = 0.0, 1.0
    while omegaE1_star(hi) < y:
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = omegaE1_star(mid)
        if abs(val - y) < tol:
            return mid
        if val < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def unit_ball_volume(D: int) -> float:
    """Volume of the Euclidean unit ball in R^D."""
    return math.pi ** (D / 2.0) / math.gamma(D / 2.0 + 1.0)


def ellipsoid_volume_from_shape_matrix(M: np.ndarray) -> float:
    """
    Ellipsoid:
        {x : (x-q)^T M^{-1} (x-q) <= 1}
    has volume vol(B_2^D) * sqrt(det(M)).
    """
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return 0.0
    D = M.shape[0]
    log_unit_ball_vol = (D / 2.0) * math.log(math.pi) - math.lgamma(D / 2.0 + 1.0)
    log_vol = log_unit_ball_vol + 0.5 * float(logdet)
    if log_vol < math.log(np.nextafter(0.0, 1.0)):
        return 0.0
    return float(math.exp(log_vol))


def sample_uniform_rotated_ellipsoid(
    center: np.ndarray,
    M: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample uniformly from {x : (x-center)^T M^{-1} (x-center) <= 1}.
    """
    D = center.shape[0]
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0.0)
    transform = eigvecs @ np.diag(np.sqrt(eigvals))

    dirs = rng.normal(size=(n_samples, D))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-300)
    radii = rng.random(n_samples) ** (1.0 / D)
    unit_ball_samples = dirs * radii[:, None]
    return center + unit_ball_samples @ transform.T


def is_in_ellipsoid(point: np.ndarray, center: np.ndarray, M_inv: np.ndarray, tol: float = 1e-12) -> bool:
    diff = point - center
    return float(diff.T @ M_inv @ diff) <= 1.0 + tol


def box_corners(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """All 2^D corners of the axis-aligned box [low, high]."""
    D = len(low)
    grids = np.meshgrid(*[[low[i], high[i]] for i in range(D)], indexing="ij")
    return np.stack(grids, axis=-1).reshape(-1, D)


def ellipsoid_coord_halfwidths(M: np.ndarray) -> np.ndarray:
    """
    Coordinate-wise halfwidths of the rotated ellipsoid:
        max |x_j - q_j| = sqrt(M_jj).
    """
    return np.sqrt(np.maximum(np.diag(M), 0.0))


def compute_rotated_ellipsoid_box_volume(
    h: dict,
    D: int,
    *,
    center: np.ndarray,
    M: np.ndarray,
    bbx_low: np.ndarray,
    bbx_high: np.ndarray,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    """
    Volume of ellipsoid ∩ box, where ellipsoid is
        (x-center)^T M^{-1} (x-center) <= 1
    and the box is [bbx_low, bbx_high].

    Returns (volume, mc_std).
    """
    center = np.asarray(center, dtype=float)
    M = np.asarray(M, dtype=float)
    bbx_low = np.asarray(bbx_low, dtype=float)
    bbx_high = np.asarray(bbx_high, dtype=float)

    M_inv = np.linalg.inv(M)
    radii_coord = ellipsoid_coord_halfwidths(M)

    # 1) Entire ellipsoid inside box?
    if np.all(center - radii_coord + tol >= bbx_low) and np.all(center + radii_coord - tol <= bbx_high):
        return float(ellipsoid_volume_from_shape_matrix(M)), 0.0

    # 2) Entire box inside ellipsoid?
    if D <= 16:
        corners = box_corners(bbx_low, bbx_high)
        if np.all([is_in_ellipsoid(c, center, M_inv, tol=tol) for c in corners]):
            return VolumeCalculator.safe_positive_product(bbx_high - bbx_low), 0.0

    # 3) Monte Carlo intersection volume
    num_samples = int(h.get("volume_mc_samples", 100_000))
    rng_seed = int(h.get("volume_mc_seed", 49))
    rng = np.random.default_rng(rng_seed)
    ellip_vol = ellipsoid_volume_from_shape_matrix(M)
    box_vol = VolumeCalculator.safe_positive_product(bbx_high - bbx_low)

    # Sample from the smaller proposal set to avoid zero-hit estimates in high dimension.
    if ellip_vol > 0.0 and ellip_vol <= box_vol:
        volumes = []
        num_repeats = int(h.get("volume_mc_repeats", 5))
        for _ in range(num_repeats):
            samples = sample_uniform_rotated_ellipsoid(center, M, num_samples, rng)
            inside_box = np.all(samples >= bbx_low, axis=1) & np.all(samples <= bbx_high, axis=1)
            p_hat = float(np.mean(inside_box))
            volumes.append(ellip_vol * p_hat)
        return float(np.mean(volumes)), float(np.std(volumes))

    samples = rng.uniform(low=bbx_low, high=bbx_high, size=(num_samples, D))
    diffs = samples - center
    quad_vals = np.einsum("ni,ij,nj->n", diffs, M_inv, diffs)
    inside = quad_vals <= 1.0 + tol

    p_hat = float(np.mean(inside))
    vol = box_vol * p_hat
    vol_std = box_vol * math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / num_samples)

    return float(vol), float(vol_std)


class NormalizedConfEllipsoid(BaseConfidenceSequence, MultiDimensionalConfidenceSequence):
    """
    Whitehouse et al. Theorem 6.1 (simple variant) confidence ellipsoid
    for the mean of bounded vector observations.

    Internally works in scaled coordinates where ||x_t|| <= 1/2,
    but all reported centers, widths, membership checks, and volumes
    are in the original mu-space.
    """

    def __init__(self, h: dict, data: np.ndarray):
        super().__init__(h, data)

        self._tol = 1e-12
        self.domain_low = float(h.get("domain_low", 0.0))
        self.domain_high = float(h.get("domain_high", 1.0))
        self.target_B = 0.5  # theorem assumption after rescaling

        # Paper-style defaults
        self.rho = float(h.get("normalized_conf_rho", 1.0))
        self.alpha_stitch = float(h.get("normalized_conf_alpha_stitch", 1.05))
        self.beta = float(h.get("normalized_conf_beta", 2.0))
        self.eps = float(h.get("normalized_conf_eps", 0.5))
        self.variant = h.get("normalized_conf_variant", "simple")  # "simple" or "complex"

        if self.variant not in {"simple", "complex"}:
            raise ValueError("normalized_conf_variant must be 'simple' or 'complex'.")

        if h.get("conf_sphere_optimistic_rescale", False):
            print("Warning: Using optimistic rescaling for NormalizedConfEllipsoid.")
            self.rescale_fn = rescale_data_optimistic
        else:
            print("Warning: Using standard rescaling for NormalizedConfEllipsoid.")
            self.rescale_fn = rescale_data

        # Scale observations so theorem assumption ||x_t|| <= 1/2 holds.
        self.data_sc, self.rescale_factor, self.box_center = self.rescale_fn(
            self.data,
            self.D,
            domain_low=self.domain_low,
            domain_high=self.domain_high,
            target_B=self.target_B,
        )

        # Streaming state in scaled space
        self.n_seen_sc = 0
        self.sum_x_sc = np.zeros(self.D, dtype=float)
        self.mu_hat_sc = np.zeros(self.D, dtype=float)
        self.V_sc = np.zeros((self.D, self.D), dtype=float)

        # Cache for compute_volume_at / is_member_at:
        # num_obs -> dict with scaled-state snapshot
        self._state_cache: Dict[int, Dict[str, Any]] = {}

    def _L(self, V_sc: np.ndarray) -> float:
        return L_cov_delta(
            V_sc,
            delta=self.alpha,  # BaseConfidenceSequence alpha is failure probability
            rho=self.rho,
            alpha=self.alpha_stitch,
            beta=self.beta,
            eps=self.eps,
        )

    def _radius_sc(self, V_sc: np.ndarray) -> float:
        """
        Radius in self-normalized S_t-space.
        """
        Vclip = spectral_clip(V_sc, self.rho)
        gmin = gamma_min(Vclip)
        L = self._L(V_sc)

        if self.variant == "simple":
            return float(np.sqrt(2.0 * self.alpha_stitch * L) + (self.alpha_stitch * L) / gmin)

        # Optional more complex first display of Theorem 6.1
        y = self.alpha_stitch * L / gmin
        return float((math.sqrt(gmin) / (1.0 - self.eps)) * inv_omegaE1_star(y))

    def _ellipsoid_state_from_scaled_state(
        self,
        n_seen_sc: int,
        mu_hat_sc: np.ndarray,
        V_sc: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Convert scaled-space streaming state into mean-space ellipsoid state.
        """
        if n_seen_sc <= 0:
            raise ValueError("Need at least one observation.")

        Vclip_sc = spectral_clip(V_sc, self.rho)
        r_sc = self._radius_sc(V_sc)

        # Mean-space ellipsoid in scaled coords:
        #   (theta_sc - mu_hat_sc)^T [n^2 Vclip^{-1}] (theta_sc - mu_hat_sc) <= r_sc^2
        # Equivalently:
        #   (theta_sc - mu_hat_sc)^T M_sc^{-1} (theta_sc - mu_hat_sc) <= 1
        # with M_sc = (r_sc^2 / n^2) * Vclip_sc
        M_sc = ((r_sc ** 2) / (n_seen_sc ** 2)) * Vclip_sc

        # Transform back to original mu-space:
        # theta_orig = box_center + rescale_factor * theta_sc
        center_orig = mu_hat_sc * self.rescale_factor + self.box_center
        M_orig = M_sc * (self.rescale_factor ** 2)

        # Coordinate halfwidths in original mu-space
        halfwidths_orig = ellipsoid_coord_halfwidths(M_orig)

        return {
            "n_seen_sc": n_seen_sc,
            "mu_hat_sc": mu_hat_sc.copy(),
            "V_sc": V_sc.copy(),
            "radius_sc": float(r_sc),
            "center_orig": center_orig.copy(),
            "M_orig": M_orig.copy(),
            "M_inv_orig": np.linalg.inv(M_orig),
            "halfwidths_orig": halfwidths_orig.copy(),
            "gmin_sc": gamma_min(Vclip_sc),
            "gmax_sc": gamma_max(Vclip_sc),
            "kappa_sc": condition_number(Vclip_sc),
        }

    def _cache_current_state(self) -> None:
        self._state_cache[self.n_seen_sc] = self._ellipsoid_state_from_scaled_state(
            self.n_seen_sc, self.mu_hat_sc, self.V_sc
        )

    def _scaled_batch_for_raw_batch(self, batch: np.ndarray) -> np.ndarray:
        batch_sc, _, _ = self.rescale_fn(
            batch,
            self.D,
            domain_low=self.domain_low,
            domain_high=self.domain_high,
            target_B=self.target_B,
        )
        return batch_sc

    def update(self, k: int, batch: np.ndarray):
        """
        Update class state using one batch of raw observations.
        Stores axis-aligned lower/upper bounds of the rotated ellipsoid in mu-space.
        """
        batch = np.asarray(batch, dtype=float)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        batch_sc = self._scaled_batch_for_raw_batch(batch)

        for x_sc in batch_sc:
            mu_prev_sc = self.mu_hat_sc.copy()

            self.n_seen_sc += 1
            diff_sc = x_sc - mu_prev_sc
            self.V_sc += np.outer(diff_sc, diff_sc)
            self.sum_x_sc += x_sc
            self.mu_hat_sc = self.sum_x_sc / self.n_seen_sc

        self._cache_current_state()
        st = self._state_cache[self.n_seen_sc]

        center = st["center_orig"]
        halfwidths = st["halfwidths_orig"]

        lower = np.maximum(self.domain_low, center - halfwidths)
        upper = np.minimum(self.domain_high, center + halfwidths)

        if self.h.get("calculate_volume", False):
            vol, std = self.compute_volume(
                k - 1,
                None,
                lower,
                upper,
                {
                    "center_orig": center,
                    "M_orig": st["M_orig"],
                },
            )
        else:
            vol, std = 0.0, 0.0

        self.lowers[k - 1, :] = lower
        self.uppers[k - 1, :] = upper
        self.volumes[k - 1] = float(vol)
        self.volume_stds[k - 1] = float(std)

    def compute_volume(
        self,
        _t,
        _Kd_calcs,
        _low: np.ndarray,
        _up: np.ndarray,
        _extra: dict,
    ) -> Tuple[float, float]:
        """
        Volume in original mu-space of the confidence ellipsoid intersected with the domain box.
        """
        center = np.asarray(_extra["center_orig"], dtype=float)
        M = np.asarray(_extra["M_orig"], dtype=float)
        return compute_rotated_ellipsoid_box_volume(
            self.h,
            self.D,
            center=center,
            M=M,
            bbx_low=np.asarray(_low, dtype=float),
            bbx_high=np.asarray(_up, dtype=float),
            tol=self._tol,
        )

    def _rebuild_state_up_to(self, num_obs: int) -> Dict[str, Any]:
        """
        Rebuild scaled-space state for the first num_obs observations.
        Used by compute_volume_at and is_member_at independently of run().
        """
        if num_obs in self._state_cache:
            return self._state_cache[num_obs]

        if num_obs <= 0:
            raise ValueError("num_obs must be >= 1.")

        sum_x_sc = np.zeros(self.D, dtype=float)
        mu_hat_sc = np.zeros(self.D, dtype=float)
        V_sc = np.zeros((self.D, self.D), dtype=float)

        for i in range(num_obs):
            x_sc = self.data_sc[i]
            mu_prev_sc = mu_hat_sc.copy()
            diff_sc = x_sc - mu_prev_sc
            V_sc += np.outer(diff_sc, diff_sc)
            sum_x_sc += x_sc
            mu_hat_sc = sum_x_sc / float(i + 1)

        st = self._ellipsoid_state_from_scaled_state(num_obs, mu_hat_sc, V_sc)
        self._state_cache[num_obs] = st
        return st

    def compute_volume_at(self, t: int) -> Tuple[float, float]:
        """
        Compute (volume, std) at 0-based batch index t.
        """
        assert self.h["calculate_volume"] is True

        B = self.batch_size
        T = self.N // B
        if not (0 <= t < T):
            raise IndexError(f"t={t} out of range [0, {T - 1}]")

        num_obs = (t + 1) * B
        st = self._rebuild_state_up_to(num_obs)

        center = st["center_orig"]
        halfwidths = st["halfwidths_orig"]
        lower = np.maximum(self.domain_low, center - halfwidths)
        upper = np.minimum(self.domain_high, center + halfwidths)

        vol, std = self.compute_volume(
            t,
            None,
            lower,
            upper,
            {
                "center_orig": center,
                "M_orig": st["M_orig"],
            },
        )
        return float(vol), float(std)

    def is_member_at(self, x, t: int) -> bool:
        """
        Check whether x belongs to the confidence set at 0-based batch index t.
        Membership is checked in original mu-space.
        """
        B = self.batch_size
        T = self.N // B
        if not (0 <= t < T):
            raise IndexError(f"t={t} out of range [0, {T - 1}]")

        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = np.array([float(x)])
        if x.shape[-1] != self.D:
            raise ValueError(f"Expected x with last dimension {self.D}, got {x.shape}.")

        if not (np.all(x >= self.domain_low) and np.all(x <= self.domain_high)):
            return False

        num_obs = (t + 1) * B
        st = self._rebuild_state_up_to(num_obs)
        return is_in_ellipsoid(x, st["center_orig"], st["M_inv_orig"], tol=self._tol)



import time
import numpy as np

# from adaptive_sample.data.data_distribution_checker import DataDistributionChecker
# from adaptive_sample.conf_sequences.normalized_conf_sphere import NormalizedConfEllipsoid


def sample_mixture_1(N, d, seed=None):
    data, true_mean = DataDistributionChecker.get_data(f"fixed_{d}d", N)
    return data, true_mean


def main(ConfClass, optimistic_rescale, D, N):
    h = {
        "alpha": 0.05,
        "batch_size": 1,
        "data_type": f"fixed_{D}d",
        "N": N,
        "cut_to_simplex": False,
        "plot_grid_2d": False,
        "calculate_volume": True,
        "verbose_progress": True,
        "D": D,
        "conf_sphere_optimistic_rescale": optimistic_rescale,

        # optional knobs for this class
        "normalized_conf_variant": "simple",   # or "complex"
        "normalized_conf_rho": 1.0,
        "normalized_conf_alpha_stitch": 1.05,
        "normalized_conf_beta": 2.0,
        "normalized_conf_eps": 0.5,

        # MC volume settings
        "volume_mc_samples": 100_000,
        "volume_mc_seed": 49,
    }

    rng_global = np.random.RandomState(41)
    X, true_mean = sample_mixture_1(N=h["N"], d=h["D"], seed=rng_global.randint(1_000_000))

    cs = ConfClass(h, X)

    start_time = time.time()
    ts, means, lowers, uppers, volumes, volume_stds = cs.run()
    end_time = time.time()

    print(f"Execution time for cs.run(): {end_time - start_time:.4f} seconds")
    print("means.shape =", means.shape)
    print("lowers.shape =", lowers.shape)
    print("uppers.shape =", uppers.shape)
    print("volumes[:5] =", volumes[:5])
    print("true_mean =", true_mean)
    print("sample_mean =", X.mean(axis=0))

    final_center = cs._rebuild_state_up_to(N)["center_orig"]
    print("final_center =", final_center)

    # Membership at the true mean is the relevant sanity check.
    print("is_member_at(true_mean) final:", cs.is_member_at(true_mean, t=len(ts) - 1))

    # Keep a midpoint probe as a diagnostic, but label it clearly.
    x_test = np.full(D, 0.5)
    print("is_member_at([0.5]*D) final:", cs.is_member_at(x_test, t=len(ts) - 1))

    # Example volume recomputation
    vol_t10, std_t10 = cs.compute_volume_at(10)
    print("volume_at(10) =", vol_t10, "std =", std_t10)


if __name__ == "__main__":
    main(
        ConfClass=NormalizedConfEllipsoid,
        optimistic_rescale=False,
        D=5,
        N=1000,
    )

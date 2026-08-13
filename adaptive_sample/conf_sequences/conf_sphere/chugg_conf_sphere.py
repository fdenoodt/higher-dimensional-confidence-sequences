import itertools
import math
from typing import Tuple
import numpy as np
from adaptive_sample.conf_sequences import BaseConfidenceSequence, MultiDimensionalConfidenceSequence
from adaptive_sample.conf_sequences.conf_sphere.conf_sphere_utils import (
    rescale_data,
    compute_ellips_bbx_volume,
    rescale_data_optimistic,
)


class EmpiricalBernsteinConfSphere(BaseConfidenceSequence, MultiDimensionalConfidenceSequence):
    # https://arxiv.org/abs/2311.08168
    """
    Time-uniform Empirical-Bernstein confidence *sphere* (Appendix E / Theorem E.2).

    This version fixes sigma_hat2 and lambda_t to be consistent with the appendix-style
    empirical proxy based on ||X_t - mu_bar_{t-1}||^2 (UNWEIGHTED empirical mean),
    and chooses lambda_t with the suggested sequential scaling ~ sqrt(A / (sigma^2 t log t)).
    """

    def __init__(self, h: dict, data: np.ndarray):
        assert h["batch_size"] == 1, "EmpiricalBernsteinCS requires batch_size=1 for unambiguous t indexing."
        super().__init__(h, data)

        self.B_scaled = 0.5  # we want scaled data with sup ||x|| <= 1/2, Theorem E.2 (55)
        self._tol = 1e-12
        self.eps = 0.5
        self.c = 2.0

        # domain bounds (assume box domain by default)
        self.domain_low = 0.0
        self.domain_high = 1.0

        if h["conf_sphere_optimistic_rescale"]:
            print("Warning: Using optimistic rescaling for BanachSphere. Confsphere is at advantage.")
            self.rescale_fn = rescale_data_optimistic
        else:
            print("Warning: Using standard rescaling for BanachSphere. (Confsphere is at a disadvantage here.)")
            self.rescale_fn = rescale_data

        self.data_sc, self.rescale_factor, self.box_center = self.rescale_fn(
            self.data,
            self.D,
            domain_low=self.domain_low,
            domain_high=self.domain_high,
            target_B=self.B_scaled,
        )

        # --- stats used in eq (55) ---
        self.running_sum = np.zeros(self.D)  # sum_{j<=t} X_j   (scaled)
        self.sum_sq_terms = np.zeros(self.N)  # ||X_t - mu_bar_{t-1}||^2

        self.log_term_const = self.D * np.log(1.0 / self.eps) + np.log(1.0 / self.alpha)

        self.lambdas = np.empty(self.N)
        self.psi_vals = np.empty(self.N)

        # --- FIXED: lambda/sigma state (appendix-consistent) ---
        # Weighted center uses Σ λ_t X_t / Σ λ_t (as in eq (55))
        self.weighted_sum = np.zeros(self.D, dtype=float)
        self.sum_lambdas = 0.0

        # lambda clip (must be < 1)
        self.c1 = 0.5

        # sigma_hat2 tracks empirical proxy based on UNWEIGHTED mean:
        # cum_sq_emp = Σ ||X_t - mu_bar_{t-1}||^2
        # sigma_hat2[t] = max(sigma_floor, cum_sq_emp / t)
        self.c2 = 0.5  # floor multiplier (kept from your code)
        self.sigma_floor = float(self.c2 * (self.B_scaled ** 2))
        self.cum_sq_emp = 0.0
        self.sigma_hat2 = np.empty(self.N + 1, dtype=float)
        self.sigma_hat2[0] = self.sigma_floor

        # sequential lambda scaling parameters
        self.mode = "cs"  # "cs" uses sqrt(t log(1+t)) denom; "batch" uses sqrt(t)
        self.A = float(self.D * np.log(1.0 / self.eps) + np.log(1.0 / self.alpha))
        self.c_lambda = 1.0  # tune if desired

        # caches for compute_volume_at (keyed by num_obs)
        self._state_cache = {}  # num_obs -> (mu_hat_scaled, radius_scaled)

    @staticmethod
    def _psi_E(lmbda: float) -> float:
        """psi_E(lambda) = | log(1 - lambda) + lambda | for lambda in [0,1)."""
        return np.abs(np.log(1.0 - lmbda) + lmbda)

    def update(self, t_one: int, batch: np.ndarray):
        """
        t_one is 1-based index for which we're updating; batch must be length 1.
        """
        t0 = t_one - 1
        assert len(batch) == 1
        assert t0 >= 0

        batch, _, _ = self.rescale_fn(
            batch,
            self.D,
            domain_low=self.domain_low,
            domain_high=self.domain_high,
            target_B=self.B_scaled,
        )
        x = batch[0]  # scaled

        # --- compute empirical proxy term v_t = ||X_t - mu_bar_{t-1}||^2 (UNWEIGHTED mean) ---
        mu_prev = np.zeros(self.D) if t0 == 0 else (self.running_sum / float(t0))
        diff = x - mu_prev
        v_emp = float(np.sum(diff**2))
        self.sum_sq_terms[t0] = v_emp

        # update unweighted running sum for next time step
        self.running_sum += x

        # --- FIXED lambda_t ---
        sigma_prev2 = float(self.sigma_hat2[t0])
        sigma_prev2 = max(self._tol, sigma_prev2)

        if self.mode == "batch":
            denom_t = math.sqrt(max(1.0, float(t_one)))
        else:
            denom_t = math.sqrt(max(1.0, float(t_one) * math.log1p(float(t_one))))

        lambda_t = float(self.c_lambda * math.sqrt(self.A / sigma_prev2) / denom_t)
        lambda_t = float(min(lambda_t, float(self.c1)))
        self.lambdas[t0] = lambda_t

        # update weighted sums (center uses these)
        self.weighted_sum += lambda_t * x
        self.sum_lambdas += lambda_t

        # --- FIXED sigma_hat2 update (consistent with v_emp) ---
        self.cum_sq_emp += v_emp
        self.sigma_hat2[t0 + 1] = max(self.sigma_floor, self.cum_sq_emp / float(t_one))

        # --- remaining eq (55) pieces ---
        self.psi_vals[t0] = float(self._psi_E(lambda_t))
        psi_vals = self.psi_vals[: t0 + 1]

        numerator = float(np.sum(psi_vals * self.sum_sq_terms[: t0 + 1]) + self.log_term_const)
        denominator = float((1.0 - self.eps) * max(self._tol, self.sum_lambdas))
        r_scaled = max(0.0, numerator / denominator)  # Euclidean radius in SCALED space

        mu_hat_scaled = (self.weighted_sum / self.sum_lambdas) if (self.sum_lambdas > 0.0) else np.zeros(self.D)

        # map back to original scale
        mu_hat = mu_hat_scaled * self.rescale_factor + self.box_center
        r = float(r_scaled * self.rescale_factor)

        lower = np.maximum(self.domain_low, mu_hat - r)
        upper = np.minimum(self.domain_high, mu_hat + r)
        vol, std = self.compute_volume(t0 + 1, None, lower, upper, {"mu_hat": mu_hat, "radius": r})

        self.lowers[t0, :] = lower
        self.uppers[t0, :] = upper
        self.volumes[t0] = vol
        self.volume_stds[t0] = std

        # invalidate cache entries that depend on future data (simplest safe option)
        # (optional: you can do smarter incremental caching)
        self._state_cache.clear()

    def compute_volume(self, _t, _Kd_calcs, _low: np.ndarray, _up: np.ndarray, _extra: dict) -> Tuple[float, float]:
        """
        Given center & radius in ORIGINAL coordinates, return (volume, std).
        Uses exact d-ball formula when ball fully inside box, or covers box, else calls VolumeCalculator.
        """
        mu_hat_orig = _extra["mu_hat"]
        r_orig = _extra["radius"]

        if r_orig <= 0.0:
            return 0.0, 0.0

        q = np.asarray(mu_hat_orig, dtype=float)
        radii = np.full(self.D, r_orig, dtype=float)
        Q = np.eye(self.D, dtype=float) * (r_orig**2)
        Q_inv = np.eye(self.D, dtype=float) * (1.0 / (r_orig**2))

        vol, std = compute_ellips_bbx_volume(
            self.h, self.D, q=q, Q=Q, Q_inv=Q_inv, radii=radii, bbx_low=_low, bbx_high=_up, pow=2
        )
        return vol, std

    def _rebuild_state_up_to(self, num_obs: int) -> Tuple[np.ndarray, float]:
        """
        Rebuild streaming state for first num_obs observations (num_obs >= 0) in SCALED space.
        Return (mu_hat_scaled, r_scaled).

        This mirrors update() exactly (with the fixed sigma/lambda logic).
        """
        if num_obs in self._state_cache:
            return self._state_cache[num_obs]

        if num_obs == 0:
            mu_hat_local = np.zeros(self.D)
            r_local = 0.0
            self._state_cache[0] = (mu_hat_local, float(r_local))
            return self._state_cache[0]

        sum_sq_terms = np.zeros(num_obs, dtype=float)
        psi_vals = np.zeros(num_obs, dtype=float)

        # state variables
        running_sum = np.zeros(self.D, dtype=float)
        weighted_sum = np.zeros(self.D, dtype=float)
        sum_lambdas = 0.0

        cum_sq_emp = 0.0
        sigma_hat2 = np.empty(num_obs + 1, dtype=float)
        sigma_hat2[0] = self.sigma_floor

        for i in range(num_obs):
            x = self.data_sc[i]
            t_one = i + 1

            # empirical proxy v_emp = ||x - mu_bar_{t-1}||^2
            mu_prev = np.zeros(self.D) if i == 0 else (running_sum / float(i))
            v_emp = float(np.sum((x - mu_prev) ** 2))
            sum_sq_terms[i] = v_emp
            running_sum += x

            # lambda_t from sigma_hat2[i]
            sigma_prev2 = float(sigma_hat2[i])
            sigma_prev2 = max(self._tol, sigma_prev2)

            if self.mode == "batch":
                denom_t = math.sqrt(max(1.0, float(t_one)))
            else:
                denom_t = math.sqrt(max(1.0, float(t_one) * math.log1p(float(t_one))))

            lambda_i = float(self.c_lambda * math.sqrt(self.A / sigma_prev2) / denom_t)
            lambda_i = float(min(lambda_i, float(self.c1)))
            psi_vals[i] = float(self._psi_E(lambda_i))

            weighted_sum += lambda_i * x
            sum_lambdas += lambda_i

            # sigma update
            cum_sq_emp += v_emp
            sigma_hat2[i + 1] = max(self.sigma_floor, cum_sq_emp / float(t_one))

        numerator = float(np.sum(psi_vals * sum_sq_terms) + self.log_term_const)
        denom_r = float((1.0 - self.eps) * max(self._tol, sum_lambdas))
        r_local = max(0.0, numerator / denom_r)

        mu_hat_local = (weighted_sum / sum_lambdas) if (sum_lambdas > 0.0) else np.zeros(self.D)

        self._state_cache[num_obs] = (mu_hat_local.copy(), float(r_local))
        return self._state_cache[num_obs]

    def compute_volume_at(self, t: int) -> Tuple[float, float]:
        """
        Compute (volume, std) for the confidence set at batch-index t (0-based).
        Rebuilds state up to num_obs = (t+1)*batch_size, maps back to ORIGINAL coords and
        delegates to compute_volume(...).
        """
        assert self.h.get("calculate_volume", True) is True
        B = self.batch_size
        T = self.N // B
        if not (0 <= t < T):
            raise IndexError(f"t={t} out of range [0, {T - 1}]")
        num_obs = (t + 1) * B

        mu_hat_scaled, r_scaled = self._rebuild_state_up_to(num_obs)

        mu_hat_orig = mu_hat_scaled * self.rescale_factor + self.box_center
        r_orig = float(r_scaled * self.rescale_factor)

        lower = np.maximum(self.domain_low, mu_hat_orig - r_orig)
        upper = np.minimum(self.domain_high, mu_hat_orig + r_orig)

        vol, vol_std = self.compute_volume(t, None, lower, upper, {"mu_hat": mu_hat_orig, "radius": r_orig})
        return float(vol), float(vol_std)

    def is_member_at(self, x, t: int) -> bool:
        """Check if x is in the confidence set at 0-based batch index t."""
        B = self.batch_size
        T = self.N // B
        if not (0 <= t < T):
            raise IndexError(f"t={t} out of range [0, {T - 1}]")

        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = np.array([float(x)])
        if x.shape[-1] != self.D:
            raise ValueError(f"Expected x with last dimension {self.D}, got {x.shape}.")

        num_obs = (t + 1) * B
        mu_hat_scaled, r_scaled = self._rebuild_state_up_to(num_obs)
        mu_hat_orig = mu_hat_scaled * self.rescale_factor + self.box_center
        r_orig = float(r_scaled * self.rescale_factor)

        if r_orig <= 0.0:
            return False

        in_box = np.all(x >= self.domain_low) and np.all(x <= self.domain_high)
        if not in_box:
            return False
        diff = x - mu_hat_orig
        return float(np.sum(diff**2)) <= (r_orig**2 + self._tol)

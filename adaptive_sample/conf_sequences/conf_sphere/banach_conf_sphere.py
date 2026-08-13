import math
from typing import Tuple
import numpy as np
from adaptive_sample.conf_sequences import (
    BaseConfidenceSequence,
    MultiDimensionalConfidenceSequence,
)
from adaptive_sample.conf_sequences.conf_sphere.conf_sphere_utils import (
    compute_ellips_bbx_volume,
    rescale_data, rescale_data_optimistic,
)


class BanachSphere(BaseConfidenceSequence, MultiDimensionalConfidenceSequence):
    # Implementation of Corollary 1 (empirical-Bernstein) from
    # https://arxiv.org/pdf/2409.06060
    def __init__(self, h: dict, data: np.ndarray):
        assert h["batch_size"] == 1, "This implementation requires batch_size=1."
        super().__init__(h, data)

        self._tol = 1e-12
        # Paper-suggested defaults
        self.B_scaled = math.sqrt(5)  # 0.25  # enforce sup ||x|| <= B_scaled
        self.c1 = 0.5  # lambda upper clip (≤ 0.8)
        self.c2 = 0.25  # variance shrinkage

        # domain bounds (default box [0,1]^D as in prior code)
        self.domain_low = 0.0
        self.domain_high = 1.0

        ### temporarily add one observation that is [self.domain_high, ....]
        # self.N += 1
        # self.data = np.vstack([self.data, np.ones((1, self.D)) * self.domain_high])
        ###

        # rescale data so sup ||x|| <= self.B_scaled
        if h['conf_sphere_optimistic_rescale']:
            print("Warning: Using optimistic rescaling for BanachSphere. Confsphere is at advantage.")
            self.rescale_fn = rescale_data_optimistic
        else:
            print("Warning: Using standard rescaling for BanachSphere. (Confsphere is at a disadvantage here.)")
            self.rescale_fn = rescale_data

        self.data_sc, self.rescale_factor, self.box_center = self.rescale_fn(
            self.data, self.D,
            domain_low=self.domain_low,
            domain_high=self.domain_high,
            target_B=self.B_scaled,
            # do_sanity_check=True  # validate all points within ball, else confsphere potential wouldn't be valid
        )

        # max_norm = np.max(np.linalg.norm(self.data_sc, axis=1)) # Since I temporarily added one point max_norm is correctly sqrt(5)

        # State needed for streaming updates (scaled coordinates)
        self.lambdas = np.zeros(self.N, dtype=float)
        self.psi_vals = np.zeros(self.N, dtype=float)
        self.weighted_sum = np.zeros(self.D, dtype=float)  # sum_{j<=t} lambda_j * X_j (scaled)
        self.sum_lambdas = 0.0
        self.cumulative = 0.0
        self.sigma_hat2 = np.empty(self.N + 1)
        self.sigma_hat2[0] = float(self.c2 * (self.B_scaled ** 2))

        # λ numerator constant from paper
        self._lambda_numer_const = math.sqrt(
            2.0 * (4.0 * self.B_scaled) ** 2 * math.log(2.0 / self.alpha))

        # log term in numerator of r
        self._log_term_const = math.log(2.0 / self.alpha)
        self.sum_sq_terms = np.zeros(self.N)  # ||X_i - mu_{i-1}||^2 for scaled data

        self.mode = "cs"  # cs | batch

        # caches for compute_volume_at
        self._state_cache = {}  # num_obs -> (mu_hat_scaled, r_scaled)

    @staticmethod
    def _psi_E(lmbda: np.ndarray) -> np.ndarray:
        """psi_E(lambda) = -log(1 - lambda) - lambda, vectorized."""
        return -np.log1p(-lmbda) - lmbda

    def _compute_lambdas_for_first_n(self, n: int, sigma_hat2: np.ndarray) -> np.ndarray:
        """Return lambdas[0:n] where lambda_i uses sigma_hat2[i-1] (predictable).
           sigma_hat2 should have length at least n (and sigma_hat2[0] is initial c2*B^2).
        """
        lambdas = np.zeros(n, dtype=float)
        for i in range(1, n + 1):
            sh = max(1e-12, math.sqrt(float(sigma_hat2[i - 1])))  # sigma_hat_{i-1}
            # denom = math.sqrt(max(1.0, float(i) * math.log1p(i)))

            if self.mode == "batch":
                denom = math.sqrt(max(1.0, float(i)))
            else:
                denom = math.sqrt(max(1.0, float(i) * math.log1p(i)))

            lambdas[i - 1] = float(self._lambda_numer_const / (sh * denom))

        # clip to (0, c1]
        np.clip(lambdas, 1e-12, float(self.c1), out=lambdas)
        return lambdas

    def update(self, t: int, batch: np.ndarray):
        """Streaming update (1-based index t)."""
        t0 = t - 1
        assert len(batch) == 1
        assert t0 >= 0

        # scale incoming batch
        batch_sc, _, _ = self.rescale_fn(  # the data here is still unscaled, so we need to scale it here too.
            batch, self.D,
            domain_low=self.domain_low,
            domain_high=self.domain_high,
            target_B=self.B_scaled
        )
        x = batch_sc[0]

        # 1) compute lambda_t using sigma_hat2[t-1] (predictable)
        sh_prev = math.sqrt(max(1e-12, float(self.sigma_hat2[t0])))  # sigma_hat2 stored so index t0 is sigma_{t-1}

        # denom_t = math.sqrt(max(1.0, float(t) * math.log1p(t)))
        if self.mode == "batch":
            denom_t = math.sqrt(max(1.0, float(t)))
        else:
            denom_t = math.sqrt(max(1.0, float(t) * math.log1p(t)))

        lambda_t = float(self._lambda_numer_const / (max(1e-12, sh_prev) * denom_t))
        lambda_t = float(min(lambda_t, float(self.c1)))
        self.lambdas[t0] = lambda_t
        self.psi_vals[t0] = float(self._psi_E(lambda_t))

        # 2) mu_bar_{t-1} (weighted)
        mu_bar_prev = (self.weighted_sum / self.sum_lambdas) if (self.sum_lambdas > 0.0) else np.zeros(self.D)

        # 3) squared difference relative to weighted previous mean:
        diff = x - mu_bar_prev
        v_t = float(np.sum(diff ** 2))
        self.sum_sq_terms[t0] = v_t

        # 4) update cumulative V_t
        self.cumulative += v_t

        # 5) update sigma_hat2[t] = c2 B^2 + cumulative/(t+1)
        self.sigma_hat2[t0 + 1] = float(self.c2 * (self.B_scaled ** 2) + (self.cumulative / float(t + 1)))

        # 6) update weighted sums with lambda_t (now that lambda_t is known)
        self.weighted_sum += lambda_t * x
        self.sum_lambdas += lambda_t

        # 7) compute numerator (with correct constants) and radius (scaled)
        # numerator = (1/(4B)) * sum_i psi_i * v_i  +  4B * log(2/alpha)
        psi_v_sum = float(np.sum(self.psi_vals[:t] * self.sum_sq_terms[:t]))
        numerator = (1.0 / (4.0 * self.B_scaled)) * psi_v_sum + (4.0 * self.B_scaled * self._log_term_const)
        denom_r = max(self._tol, self.sum_lambdas)
        r_scaled = max(0.0, float(numerator / denom_r))  # self.D *

        # 8) weighted center (scaled)
        mu_hat_scaled = self.weighted_sum / max(self.sum_lambdas, 1e-12)

        # 9) map back to original scale
        mu_hat = mu_hat_scaled * self.rescale_factor + self.box_center
        r = float(r_scaled * self.rescale_factor)

        lower = np.maximum(self.domain_low, mu_hat - r)
        upper = np.minimum(self.domain_high, mu_hat + r)
        vol, std = self.compute_volume(t - 1, None, lower, upper, {"mu_hat": mu_hat, "radius": r})
        self.lowers[t - 1, :] = lower
        self.uppers[t - 1, :] = upper
        self.volumes[t - 1] = float(vol)
        self.volume_stds[t - 1] = float(std)

    def compute_volume(
            self, _t, _Kd_calcs, _low: np.ndarray, _up: np.ndarray, _extra: dict
    ) -> Tuple[float, float]:
        mu_hat_orig = _extra["mu_hat"]
        r_orig = _extra["radius"]
        if r_orig <= 0.0:
            return 0.0, 0.0

        q = np.asarray(mu_hat_orig, dtype=float)
        radii = np.full(self.D, r_orig, dtype=float)
        Q = np.eye(self.D, dtype=float) * (r_orig ** 2)
        Q_inv = np.eye(self.D, dtype=float) * (1.0 / (r_orig ** 2))

        vol, std = compute_ellips_bbx_volume(
            self.h, self.D, q=q, Q=Q, Q_inv=Q_inv, radii=radii,
            bbx_low=_low, bbx_high=_up, pow=2
        )
        return vol, std

    def _rebuild_state_up_to(self, num_obs: int) -> Tuple[np.ndarray, float]:
        """Rebuild state for first num_obs observations (scaled space)."""
        if num_obs in self._state_cache:
            return self._state_cache[num_obs]
        if num_obs == 0:
            self._state_cache[0] = (np.zeros(self.D), 0.0)
            return self._state_cache[0]

        # init
        weighted_sum = np.zeros(self.D, dtype=float)
        sum_lambdas = 0.0
        cumulative = 0.0
        sigma_hat2 = np.empty(num_obs + 1, dtype=float)
        sigma_hat2[0] = self.c2 * (self.B_scaled ** 2)
        lambdas_local = np.zeros(num_obs, dtype=float)
        psi_vals_local = np.zeros(num_obs, dtype=float)
        sum_sq_local = np.zeros(num_obs, dtype=float)

        for i in range(1, num_obs + 1):
            x = self.data_sc[i - 1]
            # compute lambda_i from sigma_hat2[i-1]
            sh_prev = math.sqrt(max(1e-12, float(sigma_hat2[i - 1])))
            # denom_i = math.sqrt(max(1.0, float(i) * math.log1p(i)))
            if self.mode == "batch":
                denom_i = math.sqrt(max(1.0, float(i)))
            else:
                denom_i = math.sqrt(max(1.0, float(i) * math.log1p(i)))
            lambda_i = float(self._lambda_numer_const / (max(1e-12, sh_prev) * denom_i))
            lambda_i = float(min(lambda_i, float(self.c1)))
            lambdas_local[i - 1] = lambda_i
            psi_vals_local[i - 1] = float(self._psi_E(lambda_i))

            # mu_bar_{i-1}:
            mu_bar_prev = (weighted_sum / sum_lambdas) if (sum_lambdas > 0.0) else np.zeros(self.D)

            # squared difference
            v = float(np.sum((x - mu_bar_prev) ** 2))
            sum_sq_local[i - 1] = v
            cumulative += v

            # update sigma_hat2[i] = c2*B^2 + cumulative/(i+1)
            sigma_hat2[i] = float(self.c2 * (self.B_scaled ** 2) + cumulative / float(i + 1))

            # update weighted
            weighted_sum += lambda_i * x
            sum_lambdas += lambda_i

        # radius
        psi_v_sum = float(np.sum(psi_vals_local * sum_sq_local))
        numerator = (1.0 / (4.0 * self.B_scaled)) * psi_v_sum + (4.0 * self.B_scaled * self._log_term_const)
        r_local = max(0.0, float(numerator / max(self._tol, sum_lambdas)))  # self.D *
        mu_hat_local = weighted_sum / max(sum_lambdas, 1e-12)
        self._state_cache[num_obs] = (mu_hat_local.copy(), float(r_local))
        return self._state_cache[num_obs]

    def compute_volume_at(self, t: int) -> Tuple[float, float]:
        """Compute (volume, std) at 0-based batch index t."""
        assert self.h["calculate_volume"] is True

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

        vol, vol_std = self.compute_volume(
            t, None, lower, upper, {"mu_hat": mu_hat_orig, "radius": r_orig}
        )
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
        return float(np.sum(diff ** 2)) <= (r_orig ** 2 + self._tol)

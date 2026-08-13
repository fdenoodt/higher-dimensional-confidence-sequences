import itertools
import math
from typing import Tuple, Optional
import numpy as np

from adaptive_sample.helper.volume import VolumeCalculator
from adaptive_sample.shared_code import generate_samples, DistributionType

# Implementation of https://arxiv.org/pdf/2310.09100
# it implements the simplified CS after Theorem 5.4, starting from "In particular, since sub-psi..."


# rescale data so sup ||x|| <= self.B_scaled
def rescale_data(X, D: int, domain_low=0.0, domain_high=1.0, target_B=0.5):
    """
    Shift by box center and scale so that sup ||x||_2 <= target_B.
    Returns (X_scaled, rescale_factor, box_center), where internally we do:
      X_shifted = X - box_center
      Xs = X_shifted / scale   with scale = max(1.0, rescale_factor)
    Note: rescale_factor = B_raw_center / target_B, where
      B_raw_center = sqrt(D) * (domain_high - domain_low) / 2
    """
    # let \vect{x} \in [domain_low, domain_high]^D. Then ||x||_2 <= sqrt(D) * (domain_high - domain_low) = B_raw
    # but actually, we should first shift by center of box, so x' = x - 0.5*(domain_low + domain_high)
    # so ||x'||_2 <= sqrt(D) * (domain_high - domain_low) / 2 = B_raw / 2

    box_center = 0.5 * (domain_low + domain_high)
    X_shifted = X - box_center

    # bound after centering
    B_raw_center = np.sqrt(D) * (domain_high - domain_low) / 2.0

    # how much to divide by so that max norm <= target_B
    # i.e. we want scale = B_raw_center / target_B, but we will use scale = max(1.0, rescale_factor)
    rescale_factor = B_raw_center / float(target_B)

    Xs = X_shifted / rescale_factor
    return Xs, rescale_factor, box_center


# rescale, but don't care about corners
def rescale_data_optimistic(X, D: int, domain_low=0.0, domain_high=1.0, target_B=0.5, do_sanity_check: bool = True):
    org_box_center = 0.5 * (domain_low + domain_high)
    X_shifted = X - org_box_center

    rescale_factor = (domain_high - domain_low) / (2.0 * target_B)
    Xs = X_shifted / rescale_factor

    if do_sanity_check:  # validate whether all points are within circle of radius target_B
        max_norm = np.max(np.linalg.norm(Xs, axis=1))
        assert max_norm <= target_B + 1e-12, \
            f"The data doesn't lie in the expected ball after rescaling: max_norm={max_norm:.6f} > target_B={target_B:.6f}"

    return Xs, rescale_factor, org_box_center


def check_through_rnd_samples_if_in_ellipse(
        is_in_ellipsoid: callable, q: np.ndarray, Q_inv: np.ndarray,
        domain_low: np.ndarray, domain_high: np.ndarray,
        D: int, pow: int,
        tol: float = 1e-9,
) -> Optional[str]:
    """
    Return one of:
      - "ENTIRE_ELLIPSE_INSIDE_BOX"
      - "ENTIRE_BOX_IN_ELLIPSE"
      - "ELLIPSE_INTERSECTS_BOX"
    """
    # compute radii
    radii = 1.0 / (np.diag(Q_inv) ** (1.0 / pow))

    # 1) deterministic check: is the whole ellipsoid inside the box?
    if np.all(q - radii + tol >= domain_low) and np.all(q + radii - tol <= domain_high):
        return "ENTIRE_ELLIPSE_INSIDE_BOX"

    # 2) deterministic (exact) check: is the whole box inside the
    # axis-aligned L_p ellipsoid? The farthest corner is obtained by choosing
    # the endpoint farthest from q independently in every coordinate.
    farthest = np.maximum(np.abs(domain_low - q), np.abs(domain_high - q))
    max_ellip_value = float(np.sum((farthest ** pow) * np.diag(Q_inv)))
    if max_ellip_value <= 1.0 + tol:
        return "ENTIRE_BOX_IN_ELLIPSE"

    # 3) If neither deterministic containment holds -> the shapes intersect in some non-trivial way
    return "ELLIPSE_INTERSECTS_BOX"


def sample_uniform_axis_aligned_lp_ellipsoid(
        D: int,
        n_samples: int,
        radii_per_dim: np.ndarray,
        center: np.ndarray,
        pow: int,
        rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample uniformly from the axis-aligned L_p ellipsoid
        sum_i |(x_i-center_i)/r_i|^p <= 1.

    This mirrors the general branch in generate_samples but works for any p >= 1,
    including odd integers, which some hedged-capital parabola approximations use.
    """
    if pow < 1:
        raise ValueError("pow must be >= 1")

    if rng is None:
        rng = np.random.default_rng()

    radii_per_dim = np.asarray(radii_per_dim, dtype=float).reshape(1, D)
    center = np.asarray(center, dtype=float).reshape(1, D)

    if pow == 2:
        vecs = rng.normal(size=(n_samples, D))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-300)
    else:
        shape = 1.0 / float(pow)
        T = rng.gamma(shape=shape, scale=1.0, size=(n_samples, D))
        R = T ** (1.0 / float(pow))
        signs = rng.choice([-1.0, 1.0], size=(n_samples, D))
        vecs = signs * R
        lp_norms = (np.sum(np.abs(vecs) ** float(pow), axis=1, keepdims=True)) ** (1.0 / float(pow))
        vecs = vecs / np.maximum(lp_norms, 1e-300)

    base_radii = rng.random(n_samples) ** (1.0 / D)
    scaled_dirs = vecs * base_radii[:, None]
    return center + scaled_dirs * radii_per_dim


# ellipse
def compute_ellips_bbx_volume(
        h: dict,
        D: int,
        *,
        q: np.ndarray,
        Q: np.ndarray,  # shape (D,D)
        Q_inv: np.ndarray,  # shape (D,D)
        radii: np.ndarray,  # shape (D,)

        # e.g. for [0,1]^D or for bbx of ellipsoid etc, just to know where to sample. It can also be the bbx computed for ellipse intersection
        bbx_low: np.ndarray[float],
        bbx_high: np.ndarray[float],
        pow: int
) -> Tuple[float, float]:
    q = np.asarray(q, dtype=float)
    assert q.shape == (D,)
    assert Q.shape == (D, D)
    assert bbx_low.shape == (D,)

    def is_in_ellipsoid(point: np.ndarray) -> bool:
        v = np.abs(point - q)
        p = pow
        ellip_val = float(np.sum((v ** p) * np.diag(Q_inv)))
        return (ellip_val <= 1.0)

    def is_in_cs(point: np.ndarray) -> bool:
        return ((is_in_ellipsoid(point)) and
                np.all(point >= bbx_low) and np.all(point <= bbx_high))

    def ellipsoid_volume() -> float:
        log_V_unit_lp = D * math.log(2.0 * math.gamma(1.0 + 1.0 / pow)) - math.log(math.gamma(1.0 + D / pow))
        log_prod_radii = float(np.sum(np.log(radii)))
        log_vol = log_V_unit_lp + log_prod_radii

        # Smallest positive subnormal float64 is about exp(-744.44). Below that the reported
        # volume cannot be represented in float64, so returning 0 is the only faithful option.
        if log_vol < math.log(np.nextafter(0.0, 1.0)):
            return 0.0
        return float(math.exp(log_vol))

    def monte_carlo_intersection_via_ellipsoid(
            n_samples: int = 25_000,
            num_repeats: int = 5,
    ) -> Tuple[float, float]:
        n_samples = int(h.get("volume_mc_samples", n_samples))
        num_repeats = int(h.get("volume_mc_repeats", num_repeats))
        if n_samples <= 0:
            raise ValueError("volume_mc_samples must be positive.")
        if num_repeats <= 0:
            raise ValueError("volume_mc_repeats must be positive.")
        ellip_vol = ellipsoid_volume()
        if ellip_vol == 0.0:
            return 0.0, 0.0

        volumes = []
        base_seed = h.get("volume_mc_seed")
        for repeat_idx in range(num_repeats):
            rng = np.random.default_rng(
                None if base_seed is None else int(base_seed) + repeat_idx
            )
            samples = sample_uniform_axis_aligned_lp_ellipsoid(
                D, n_samples, radii_per_dim=radii, center=q, pow=pow, rng=rng
            )
            inside_box = np.all(samples >= bbx_low, axis=1) & np.all(samples <= bbx_high, axis=1)
            volumes.append(float(np.mean(inside_box)) * ellip_vol)
        return float(np.mean(volumes)), float(np.std(volumes))

    rel = check_through_rnd_samples_if_in_ellipse(
        is_in_ellipsoid, q, Q_inv, bbx_low, bbx_high, D, pow
    )
    if rel == 'ENTIRE_ELLIPSE_INSIDE_BOX':  # entire sphere in box
        return ellipsoid_volume(), 0.0
    elif rel == 'ENTIRE_BOX_IN_ELLIPSE':  # entire box is in
        return VolumeCalculator.safe_positive_product(bbx_high - bbx_low), 0.0
    else:  # ELLIPSE_INTERSECTS_BOX
        ellip_vol = ellipsoid_volume()
        box_vol = VolumeCalculator.safe_positive_product(bbx_high - bbx_low)

        # Sample from the smaller proposal set to avoid zero-hit estimates in high dimension.
        if ellip_vol > 0.0 and ellip_vol <= box_vol:
            return monte_carlo_intersection_via_ellipsoid()

        vol, vol_std = VolumeCalculator.monte_carlo_cs_volume(
            h, is_in_cs, D=D,
            bbx_lower=bbx_low, bbx_upper=bbx_high)
        return float(vol), float(vol_std)


if __name__ == "__main__":
    def rescale_data_demo(X, rescale_f: callable):
        import matplotlib.pyplot as plt

        np.random.seed(0)

        # parameters
        domain_low = 0.0
        domain_high = 1.0
        target_B = 0.5

        # call rescale
        Xs, rescale_factor, box_center = rescale_f(X, D, domain_low, domain_high, target_B)

        # checks
        X_shifted = X - box_center
        max_norm_shifted = np.max(np.linalg.norm(X_shifted, axis=1))
        max_norm_scaled = np.max(np.linalg.norm(Xs, axis=1))

        print(f"rescale_factor: {rescale_factor:.6f}")
        print(f"box_center: {box_center}")
        print(f"max ||x - center|| (before scaling): {max_norm_shifted:.6f}")
        print(f"max ||x|| (after scaling):  {max_norm_scaled:.6f}  (should be <= target_B = {target_B})")

        # plot original and transformed side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.6)
        ax.set_title("Original data in [0, 1]^2")
        ax.set_xlim(domain_low - 0.05, domain_high + 0.05)
        ax.set_ylim(domain_low - 0.05, domain_high + 0.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

        ax = axes[1]
        ax.scatter(Xs[:, 0], Xs[:, 1], s=6, alpha=0.6)
        ax.set_title("Centered & rescaled data")
        # draw circle of radius target_B to show bound
        circle = plt.Circle((0.0, 0.0), target_B, fill=False, linewidth=1.5)
        ax.add_patch(circle)
        lim = target_B * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

        plt.tight_layout()
        plt.show()


    # generate uniform data in [0,1]^2
    N = 1000
    D = 2

    ### Data
    # X = np.random.rand(N, D) * (domain_high - domain_low) + domain_low
    # generate data in [0, 1]^2 but inside a circle of radius 0.5
    angles = np.random.rand(N) * 2.0 * np.pi
    radii = np.sqrt(np.random.rand(N)) * 0.5
    X = np.zeros((N, D))
    X[:, 0] = 0.5 + radii * np.cos(angles)
    X[:, 1] = 0.5 + radii * np.sin(angles)

    # rescale_data | rescale_data_optimistic
    rescale_data_demo(X, rescale_data_optimistic)

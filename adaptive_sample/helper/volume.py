import numpy as np
import math
from typing import Callable, Tuple, Optional, Dict, Any
from itertools import product, combinations
from typing import Tuple
from scipy.spatial import ConvexHull
from joblib import Parallel, delayed


class VolumeCalculator:
    @staticmethod
    def safe_positive_product(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if np.any(values < 0.0):
            raise ValueError("safe_positive_product expects non-negative inputs.")
        if np.any(values == 0.0):
            return 0.0

        log_prod = float(np.sum(np.log(values)))
        if log_prod < math.log(np.nextafter(0.0, 1.0)):
            return 0.0
        return float(math.exp(log_prod))

    @staticmethod
    def _sample_uniform_simplex(n: int, D: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample n points uniformly from the (D-1)-simplex (probability simplex) in R^D.
        Uses the standard method: sample Exp(1) independent coordinates and normalize.
        Returns array shape (n, D).
        """
        if rng is None:
            rng = np.random.default_rng()
        # exponential(1) samples; normalize rows to sum 1
        x = rng.exponential(scale=1.0, size=(n, D))
        x /= x.sum(axis=1, keepdims=True)
        return x

    @staticmethod
    def monte_carlo_cs_volume(
            h: dict,
            is_in_cs: Callable[[np.ndarray], bool],
            D: int,
            n_samples: int = 25_000,
            num_repeats: int = 5,
            bbx_lower: Optional[np.ndarray] = None,
            bbx_upper: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Monte Carlo estimate of the volume of a confidence set, repeated 5 times to compute
        the average volume and standard deviation.

        if bbx_lower and bbx_upper are provided, the sampling is done within the bounding box to
        potentially improve accuracy for small volumes. The volume is still corrected to the full domain volume.

        Warning: bbx only used when cut_to_simplex is False.
        """

        # Optional experiment-level overrides. Keeping the signature defaults
        # preserves the paper settings while allowing large ablations to make
        # an explicit, reproducible precision/runtime tradeoff.
        n_samples = int(h.get("volume_mc_samples", n_samples))
        num_repeats = int(h.get("volume_mc_repeats", num_repeats))
        if n_samples <= 0:
            raise ValueError("volume_mc_samples must be positive.")
        if num_repeats <= 0:
            raise ValueError("volume_mc_repeats must be positive.")
        base_seed = h.get("volume_mc_seed")

        # if hedged capital, and dim > 50, do more samples
        interval_types = h['interval_types']
        if "volume_mc_samples" not in h and D >= 50 and 'HEDGE_nd_GRID' in interval_types:
            n_samples = 100_000

        if h['cut_to_simplex'] is False:
            assert bbx_lower is not None and bbx_upper is not None, \
                "It's better to use bbx_lower and bbx_upper to improve accuracy for small volumes."

        # Parallel execution
        volumes = Parallel(n_jobs=-1)(
            delayed(VolumeCalculator._monte_carlo_cs_volume)(h, is_in_cs, D, n_samples,
                                                             bbx_lower=bbx_lower, bbx_upper=bbx_upper,
                                                             rng_seed=(None if base_seed is None
                                                                       else int(base_seed) + repeat_idx))
            for repeat_idx in range(num_repeats)
        )

        # synchronous execution
        # volumes = []
        # for _ in range(num_repeats):
        #     vol = VolumeCalculator._monte_carlo_cs_volume(
        #         h, is_in_cs, D, n_samples, bbx_lower=bbx_lower, bbx_upper=bbx_upper
        #     )
        #     volumes.append(vol)

        avg_volume = float(np.mean(volumes))
        std_dev_volume = float(np.std(volumes))

        return avg_volume, std_dev_volume

    @staticmethod
    def _monte_carlo_cs_volume(
            h: dict,
            is_in_cs: Callable[[np.ndarray], bool],
            D: int,
            n_samples: int,
            bbx_lower: Optional[np.ndarray] = None,
            bbx_upper: Optional[np.ndarray] = None,
            rng_seed: Optional[int] = None,
    ) -> float:
        if h['calculate_volume'] is False:
            return 0.0

        rng = np.random.default_rng(rng_seed)

        cut_to_simplex = h['cut_to_simplex']

        # Generate samples
        if cut_to_simplex:
            samples = VolumeCalculator._sample_uniform_simplex(n_samples, D, rng)
            domain_vol = math.sqrt(D) / math.factorial(D - 1)  # (D-1)-volume of simplex
        else:

            samples = rng.random(size=(n_samples, D))
            domain_vol = 1.0  # hypercube [0,1]^D

            if bbx_lower is not None and bbx_upper is not None:
                # Adjust samples to fit within the bounding box
                samples = samples * (bbx_upper - bbx_lower) + bbx_lower
                domain_vol = VolumeCalculator.safe_positive_product(bbx_upper - bbx_lower)

        # 1. quicky santity check: check for a few points whether they are in [0,1]^D or the specified bbx
        VolumeCalculator._sanity_check_points_in_bbx(samples, bbx_lower, bbx_upper, cut_to_simplex)
        # 2. also quickly check for a few points that are not in the bbx, that `is_in_cs` returns False
        VolumeCalculator._sanity_check_points_not_in_bbx(is_in_cs, bbx_lower, bbx_upper, cut_to_simplex, D)

        mask_list = [bool(is_in_cs(samples[i])) for i in range(n_samples)]
        mask = np.asarray(mask_list, dtype=bool)

        fraction = float(np.mean(mask))
        volume_est = fraction * domain_vol

        return volume_est

    @staticmethod
    def _sanity_check_points_in_bbx(points: np.ndarray, bbx_lower: Optional[np.ndarray],
                                    bbx_upper: Optional[np.ndarray], cut_to_simplex: bool):
        """Sanity check that points are within the bounding box or [0,1]^D."""
        if bbx_lower is not None and bbx_upper is not None:
            if not np.all((points >= bbx_lower) & (points <= bbx_upper)):
                raise ValueError("Some points are outside the specified bounding box.")
        elif not cut_to_simplex:
            if not np.all((points >= 0.0) & (points <= 1.0)):
                raise ValueError("Some points are outside the unit hypercube [0,1]^D.")
        # else: no check needed for simplex since samples are generated correctly

    @staticmethod
    def _sanity_check_points_not_in_bbx(is_in_cs: Callable[[np.ndarray], np.ndarray],
                                        bbx_lower: Optional[np.ndarray],
                                        bbx_upper: Optional[np.ndarray],
                                        cut_to_simplex: bool, D: int):
        """Sanity check that points outside the bounding box or [0,1]^D are not in the confidence set."""
        if cut_to_simplex:
            return True

        eps = 1e-8
        outside_points = []

        # If a bounding box is provided, create points slightly outside each face of the box.
        if bbx_lower is not None and bbx_upper is not None:
            center = (bbx_lower + bbx_upper) / 2.0
            span = bbx_upper - bbx_lower
            n_checks = min(5, D)
            for i in range(n_checks):
                p_above = center.copy()
                p_below = center.copy()
                offset = max(eps, 0.01 * float(span[i]) if span[i] != 0 else 0.01)
                p_above[i] = float(bbx_upper[i]) + offset
                p_below[i] = float(bbx_lower[i]) - offset
                outside_points.append(p_above)
                outside_points.append(p_below)
        else:
            # Domain is the unit hypercube [0,1]^D: produce slightly out-of-range points on some axes
            center = np.full(D, 0.5, dtype=float)
            n_checks = min(5, D)
            for i in range(n_checks):
                p_above = center.copy()
                p_below = center.copy()
                p_above[i] = 1.0 + 0.01
                p_below[i] = -0.01
                outside_points.append(p_above)
                outside_points.append(p_below)

        # Check each outside point
        for pt in outside_points:
            try:
                res = is_in_cs(pt)
            except Exception as e:
                raise ValueError(f"is_in_cs raised an exception for outside point {pt!r}: {e}") from e

            # Treat any truthy / any-True return as a violation
            arr = np.asarray(res)
            reported_inside = bool(arr.any()) if arr.size != 1 else bool(arr.item())

            if reported_inside:
                raise ValueError(
                    f"Sanity check failed: point {pt!r} is outside the sampling domain but "
                    f"is_in_cs returned {res!r} (treated as True)."
                )

        return True

    @staticmethod
    def hypercube_volume(h: dict, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-12) -> float:
        if h['calculate_volume'] is False:
            return 0.0

        """
        Compute the volume of a Bonferroni axis-aligned region.

        Parameters
        ----------
        lower : (D,) array
            Lower bounds per dimension.
        upper : (D,) array
            Upper bounds per dimension.
        cut_to_simplex : bool
            If False: compute hypercube box volume = prod(u-l).
            If True: compute volume of intersection with simplex (D-1 dimensional volume).
        tol : float
            Numerical tolerance for feasibility tests.

        Returns
        -------
        vol : float
            Estimated geometric volume (hypercube or simplex (D-1)-volume).
        """
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape (D,)")

        D = lower.shape[0]

        cut_to_simplex = h['cut_to_simplex']
        if not cut_to_simplex:
            # hypercube: exact product of edge lengths
            diffs = upper - lower
            diffs = np.maximum(diffs, 0.0)  # guard against negative width
            return VolumeCalculator.safe_positive_product(diffs)

        # else: compute intersection with simplex
        # quick feasibility check: if sum of lowers > 1 or sum of uppers < 1 -> empty
        if lower.sum() - tol > 1.0 or upper.sum() + tol < 1.0:
            return 0.0

        vertices = []
        # iterate free index r
        for r in range(D):
            fixed_idx = [i for i in range(D) if i != r]
            # iterate all assignments (lower/upper) for the D-1 fixed indices
            for bits in product((0, 1), repeat=D - 1):
                x = np.empty(D, dtype=float)
                # assign fixed indices
                for k, idx in enumerate(fixed_idx):
                    x[idx] = lower[idx] if bits[k] == 0 else upper[idx]
                # compute remaining variable from sum constraint
                x[r] = 1.0 - x[fixed_idx].sum()
                # check feasibility within bounds (allow small tol)
                if x[r] + tol >= lower[r] and x[r] <= upper[r] + tol:
                    # ensure all coordinates are within [0,1] plus tolerance
                    if np.all(x >= -tol) and np.all(x <= 1.0 + tol):
                        # clip tiny numerical noise and append
                        x_clipped = np.clip(x, 0.0, 1.0)
                        vertices.append(tuple(np.round(x_clipped, 12)))  # rounding helps dedupe

        if len(vertices) == 0:
            return 0.0

        # unique vertices
        verts = np.array(sorted(set(vertices)), dtype=float)  # shape (m, D)
        # project to first D-1 coordinates (u-space). This is a valid linear coordinate map
        # from hyperplane sum=1 into R^{D-1}. ConvexHull on these coords yields vol_u.
        proj = verts[:, : (D - 1)]
        if proj.shape[0] <= D - 1:
            # degenerate: not enough points to form (D-1)-volume
            return 0.0

        try:
            hull = ConvexHull(proj)
            vol_u = hull.volume  # volume in u-space (dim = D-1)
            # scaling factor from u-space to embedded hyperplane Euclidean measure:
            scale = math.sqrt(D)
            return scale * vol_u
        except Exception:
            # fallback: numerical issue computing hull -> try simpler polygon area for D=3
            if D == 3:
                # 2D polygon area via shoelace
                poly = proj
                # compute convex hull manually for 2D
                hull2 = VolumeCalculator._convex_hull_2d_for_fallback(poly)
                if hull2.size:
                    return math.sqrt(3) * VolumeCalculator._polygon_area(hull2)
                else:
                    return 0.0
            else:
                return 0.0

    @staticmethod
    def hypercube_volumes(h: dict, lowers: np.ndarray, uppers: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        if h['calculate_volume'] is False:
            return np.zeros(lowers.shape[0], dtype=float)

        """
        Compute volumes of Bonferroni axis-aligned regions.

        Parameters
        ----------
        lowers : (T, D) array
            Lower bounds per time and dimension.
        uppers : (T, D) array
            Upper bounds per time and dimension.
        cut_to_simplex : bool
            If False: compute hypercube box volume = prod(u-l).
            If True: compute volume of intersection with simplex (D-1 dimensional volume).
        tol : float
            Numerical tolerance for feasibility tests.

        Returns
        -------
        vols : (T,) array
            Estimated geometric volumes (hypercube or simplex (D-1)-volume).
        """
        lowers = np.asarray(lowers, dtype=float)
        uppers = np.asarray(uppers, dtype=float)
        if lowers.shape != uppers.shape:
            raise ValueError("lowers and uppers must have same shape (T, D)")
        T, D = lowers.shape
        vols = np.zeros(T, dtype=float)

        cut_to_simplex = h['cut_to_simplex']
        if not cut_to_simplex:
            # hypercube: exact product of edge lengths
            diffs = uppers - lowers
            diffs = np.maximum(diffs, 0.0)  # guard against negative width
            vols = np.array([VolumeCalculator.safe_positive_product(row) for row in diffs], dtype=float)
            return vols

        # else: compute intersection with simplex for each t
        # enumeration method: for each free index r (one variable determined by sum),
        # set remaining D-1 variables to either lower or upper bounds (2^(D-1) combos).
        # if resulting remaining variable in its bounds -> vertex.
        for t in range(T):
            l = lowers[t]
            u = uppers[t]
            # quick feasibility check: if sum of lowers > 1 or sum of uppers < 1 -> empty
            if l.sum() - tol > 1.0 or u.sum() + tol < 1.0:
                vols[t] = 0.0
                continue

            vertices = []
            # iterate free index r
            for r in range(D):
                fixed_idx = [i for i in range(D) if i != r]
                # iterate all assignments (lower/upper) for the D-1 fixed indices
                for bits in product((0, 1), repeat=D - 1):
                    x = np.empty(D, dtype=float)
                    # assign fixed indices
                    for k, idx in enumerate(fixed_idx):
                        x[idx] = l[idx] if bits[k] == 0 else u[idx]
                    # compute remaining variable from sum constraint
                    x[r] = 1.0 - x[fixed_idx].sum()
                    # check feasibility within bounds (allow small tol)
                    if x[r] + tol >= l[r] and x[r] <= u[r] + tol:
                        # ensure all coordinates are within [0,1] plus tolerance
                        if np.all(x >= -tol) and np.all(x <= 1.0 + tol):
                            # clip tiny numerical noise and append
                            x_clipped = np.clip(x, 0.0, 1.0)
                            vertices.append(tuple(np.round(x_clipped, 12)))  # rounding helps dedupe
            if len(vertices) == 0:
                vols[t] = 0.0
                continue

            # unique vertices
            verts = np.array(sorted(set(vertices)), dtype=float)  # shape (m, D)
            # project to first D-1 coordinates (u-space). This is a valid linear coordinate map
            # from hyperplane sum=1 into R^{D-1}. ConvexHull on these coords yields vol_u.
            proj = verts[:, : (D - 1)]
            if proj.shape[0] <= D - 1:
                # degenerate: not enough points to form (D-1)-volume
                vols[t] = 0.0
                continue

            try:
                hull = ConvexHull(proj)
                vol_u = hull.volume  # volume in u-space (dim = D-1)
                # scaling factor from u-space to embedded hyperplane Euclidean measure:
                scale = math.sqrt(D)
                vols[t] = scale * vol_u
            except Exception:
                # fallback: numerical issue computing hull -> try simpler polygon area for D=3
                if D == 3:
                    # 2D polygon area via shoelace
                    poly = proj
                    # compute convex hull manually for 2D
                    hull2 = VolumeCalculator._convex_hull_2d_for_fallback(poly)
                    if hull2.size:
                        vols[t] = math.sqrt(3) * VolumeCalculator._polygon_area(hull2)
                    else:
                        vols[t] = 0.0
                else:
                    vols[t] = 0.0

        return vols

    @staticmethod
    # small helpers used in fallback (copy these if you want fallback behavior)
    def _convex_hull_2d_for_fallback(pts):
        """Simple 2D monotone chain for fallback when scipy fails (returns closed hull)."""
        pts = np.unique(np.round(np.asarray(pts, float), 12), axis=0)
        if pts.shape[0] <= 1:
            return pts
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))
        upper = []
        for p in pts[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(tuple(p))
        hull = np.array(lower[:-1] + upper[:-1], dtype=float)
        if hull.shape[0] > 0:
            hull = np.vstack([hull, hull[0]])
        return hull

    @staticmethod
    def _polygon_area(poly):
        """Shoelace formula for polygon area; poly is (m,2) closed (first==last)."""
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def ellipsoid_volume(h: dict, semi_axes: np.ndarray) -> float:
        raise NotImplementedError  # I originally planned on exact, but what if ellipsoid is partly out of [0, 1]^D?

        # https://math.stackexchange.com/questions/332391/volume-of-hyperellipsoid
        if h['calculate_volume'] is False:
            return 0.0

        """
        Compute the volume of an ellipsoid defined by its semi-axis lengths.

        Parameters
        ----------
        semi_axes : (D,) array
            Lengths of the semi-axes of the ellipsoid.

        Returns
        -------
        vol : float
            Estimated geometric volume of the ellipsoid.
        """
        semi_axes = np.asarray(semi_axes, dtype=float)
        if semi_axes.ndim != 1:
            raise ValueError("semi_axes must be a 1D array of shape (D,)")

        D = semi_axes.shape[0]
        if D < 1:
            raise ValueError("Dimension D must be at least 1")

        # Volume formula: V = (pi^(D/2) / Gamma(D/2 + 1)) * prod(a_i)
        # where a_i are the semi-axis lengths
        volume = (math.pi ** (D / 2)) / math.gamma(D / 2 + 1) * np.prod(semi_axes)
        return volume

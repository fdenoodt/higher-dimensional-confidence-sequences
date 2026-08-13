from typing import Callable, Optional, Tuple, List, Union
import numpy as np
from numpy import ndarray

from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_utils import find_minimum_of_k


def bisection(f, a, b, tol=1e-8, max_iter=100_000):
    """
    Find m in [a, b] such that f(m)=0 by bisection.
    Requires f(a) and f(b) to have opposite signs.
    """
    fa, fb = f(a), f(b)

    assert ((fa <= 0 <= fb) or (fb <= 0 <= fa)), \
        f"No sign change on [a,b]. f(a)={fa}, f(b)={fb}, a={a}, b={b}. " \
        f"Please ensure that f(a) and f(b) have opposite signs."

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol:
            return c

        # decide which half to keep
        if (fa <= 0 <= fc) or (fc <= 0 <= fa):
            b, fb = c, fc
        elif (fb <= 0 <= fc) or (fc <= 0 <= fb):
            a, fa = c, fc
        else:
            raise ValueError(f"f(c)={fc} does not have the same sign as f(a)={fa} or f(b)={fb}. "
                             f"Please check your function and interval.")

    return 0.5 * (a + b)  # best we got


# parabola function in your preferred algebraic form
def parabola(x, a_param, b_param, c_param, pow: int):
    return (np.abs(x - a_param) ** pow) / (b_param ** pow) - c_param


def construct_conservative_k(k_plus: Callable[[float], float],
                             k_minus: Callable[[float], float],
                             x1: float, x2: float,
                             xs: ndarray, f: Callable[[float], float],
                             ) -> Tuple[ndarray, ndarray]:
    """
    Construct conservative grid values for k based on k_plus and k_minus callables.
    
    Args:
        !!! assumes k_plus and k_minus to be weighted already (so f(x) = max(k_plus, k_minus) !!!
        k_plus: Callable that returns k_plus values
        k_minus: Callable that returns k_minus values
        x1: Left boundary of domain
        x2: Right boundary of domain
        xs: Grid points
        f: Function to evaluate
    """
    if k_plus is None or k_minus is None:
        raise ValueError("conservative=True requires k_plus and k_minus callables.")

    # Sanity check: after the log-space hedged-capital rewrite these can differ by
    # tiny rounding/clipping effects, so exact float equality is too strict here.
    f_mid = float(f(0.5))
    k_mid = float(max(k_plus(0.5), k_minus(0.5)))
    if not np.isclose(f_mid, k_mid, rtol=1e-10, atol=1e-12, equal_nan=False):
        raise AssertionError(
            f"f must be defined as max(k_plus, k_minus). Got f(0.5)={f_mid}, max={k_mid}."
        )

    k_lower, k_upper, a, b = find_minimum_of_k(
        k_plus=k_plus,
        k_minus=k_minus,
        a=0, b=1,
        # warning! k_minus and k_plus are not extended,
        # so setting a=x1, b=x2 may cause an incorrect root out of the domain [0, 1] to be found.
        # instead, i hardcode it as [0, 1] here as the minimum should in any case be there
    )

    phi = k_lower

    # add the two points a, b to the grid and keep sorted order
    xs = np.sort(np.unique(np.concatenate([xs, np.array([a, b], dtype=np.float64)])))
    ys = np.array([f(xx) for xx in xs], dtype=np.float64)

    # Determine l, r from the grid argmin as in the theorem.
    m_idx = int(np.argmin(ys))
    k_min = ys[m_idx]

    # working with k_vals would be more faithful, but my extend_f function has issue that it doesn't extend the individual k_plus, k_minus functions, so better to work with f
    # k_vals = np.array([max(k_plus(xx), k_minus(xx)) for xx in xs], dtype=np.float64)
    # m_idx = int(np.argmin(k_vals))
    # k_min = k_vals[m_idx]

    # left_candidates = np.where(k_vals[:m_idx] > k_min)[0]
    left_candidates = np.where(ys[:m_idx] > k_min)[0]
    if left_candidates.size > 0:
        l_idx = int(left_candidates.max())
    else:
        l_idx = 0

    # right_candidates = np.where(k_vals[m_idx + 1:] > k_min)[0]
    right_candidates = np.where(ys[m_idx + 1:] > k_min)[0]
    if right_candidates.size > 0:
        r_idx = int(m_idx + 1 + right_candidates.min())
    else:
        r_idx = len(xs) - 1

    # Construct conservative grid values k_i^s.
    # ys_safe = k_vals.copy()
    ys_safe = ys.copy()
    if l_idx > 0:
        # ys_safe[:l_idx] = k_vals[1:l_idx + 1]
        ys_safe[:l_idx] = ys[1:l_idx + 1]
    if r_idx < len(xs) - 1:
        # ys_safe[r_idx + 1:] = k_vals[r_idx:-1]
        ys_safe[r_idx + 1:] = ys[r_idx:-1]
    ys_safe[l_idx:r_idx + 1] = phi

    return xs, ys_safe


def extend_domain_for_parabola_fit(f, x1_touch, x2_touch, C: float, is_entirely_kplus: bool, is_entirely_kminus):
    # in case that f(x) = max(k_plus, k_minus) is enirely k_plus or k_minus on one side, just copy the other side entirely
    if is_entirely_kplus:
        def f_extended(x):
            if x <= x2_touch:
                return f(x)
            else:
                return f(-x + 2 * x2_touch)

        return f_extended, x1_touch, 2 * x2_touch - x1_touch

    if is_entirely_kminus:
        def f_extended(x):
            if x >= x1_touch:
                return f(x)
            else:
                return f(-x + 2 * x1_touch)

        return f_extended, 2 * x1_touch - x2_touch, x2_touch

    # 1. if both x1_start and x2_end surpass C, no need to extend
    f_x1 = f(x1_touch)
    f_x2 = f(x2_touch)
    if f_x1 >= C and f_x2 >= C:
        return f, x1_touch, x2_touch

    # 2. at least one of the endpoints is below C, need to extend.
    if f_x1 > f_x2:  # left side is ok, so extend right side
        # x1_touch is fine, extend x2_end.
        assert x2_touch == 1, "we could only have gotten here if x2_touch was at the end"
        # assert f_x1 approx c:
        assert np.isclose(f_x1, C, atol=1e-1) \
               or (x1_touch == 0 and x2_touch == 1), \
            f"f_x1 should be close to C here. got f_x1={f_x1}, C={C}"

        x_alpha_left = x1_touch
        x_beta_left = bisection(f=lambda x: f(x) - f_x2,
                                a=x_alpha_left,
                                # small epsilon to avoid edge case where we find the same x2_touch again
                                b=x2_touch - 1e-6
                                )
        x_beta_right = x2_touch
        x_alpha_right = x_beta_right + (x_beta_left - x_alpha_left)

        # print(f"{x_alpha_left=}, {x_beta_left=}, {x_beta_right=}, {x_alpha_right=}")

        def f_extended(x):
            if x <= x_beta_right:
                return f(x)
            else:
                return f(-x + x_beta_right + x_beta_left)

        return (
            f_extended,
            x1_touch,
            x_alpha_right)


    elif f_x2 > f_x1:  # right is ok, extend left
        # x2_touch is fine, extend x1_start.
        assert x1_touch == 0, "we could only have gotten here if x1_touch was at the start"
        # assert f_x2 approx c:
        assert np.isclose(f_x2, C, atol=1e-2) \
               or (x1_touch == 0 and x2_touch == 1), \
            f"f_x2 should be close to C here. got f_x2={f_x2}, C={C}"

        x_alpha_right = x2_touch
        x_beta_right = bisection(f=lambda x: f(x) - f_x1,
                                 a=x_alpha_right,
                                 b=x1_touch + 1e-6  # small epsilon to avoid finding the same x1_touch again
                                 )
        x_beta_left = x1_touch
        x_alpha_left = x_beta_left - (x_alpha_right - x_beta_right)

        def f_extended(x):
            if x_beta_left <= x:
                return f(x)
            else:
                return f(-x + x_beta_left + x_beta_right)

        return (
            f_extended,
            x_alpha_left,
            x2_touch)
    else:
        # both sides are equal: no point extending
        return f, x1_touch, x2_touch


class ParabolaApproximation:

    @staticmethod
    def approximate_parabola(h: dict, f: Callable[[float], float],
                             x1_start: float, x2_end: float,
                             x1_touch: float, x2_touch: float,
                             C: float,
                             method: str = "C", options: dict = None
                             ) -> \
            Optional[Tuple[float, float, float]]:
        assert options is not None, "Options with `parabola_power` must be provided for method D."
        pow = options['parabola_power']

        if h['extend_f_domain_for_parabola_fit']:
            kplus = options['k_plus']
            kminus = options['k_minus']
            is_entirely_kplus = kplus(1) >= kminus(
                1)  # only need to check these 2 pts since kplus is decreasing and kminus is increasing
            is_entirely_kminus = kminus(0) >= kplus(0)
            f, x1_extended, x2_extended = extend_domain_for_parabola_fit(f, x1_touch, x2_touch, C, is_entirely_kplus,
                                                                         is_entirely_kminus)
            if x1_extended != x1_touch:  # left side extended
                assert x1_start <= 0.0, "we could only have extended left if we started at 0"
                x1_start = x1_extended
                x1_touch = x1_extended
                # print(f"Extended f domain for parabola fit to [{x1_start}, {x2_end}]")

            elif x2_extended != x2_touch:  # right side extended
                assert x2_end >= 1.0, "we could only have extended right if we ended at 1"
                x2_end = x2_extended
                x2_touch = x2_extended
                # print(f"Extended f domain for parabola fit to [{x1_start}, {x2_end}]")
            else:
                pass  # no extension happened

        # N_grid = 2001
        N_grid = h['parabola_grid_size']

        if method == "A":
            raise NotImplementedError("Method A not implemented.")
        elif method == "B":
            assert C > 20, "C must be larger than 20 due to quasiconvexity not being strict."
            assert pow == 2, "Method B only implemented for standard parabola (power=2)."
            # It should be C+\delta. This code assumes self.alpha = 0.05 :/
            return ParabolaApproximation._approximate_parabola_only_compare_lt_C_regions(
                f, x1_touch, x2_touch,
                # x1_start, x2_end,
                C, N_grid
            )
        elif method == "C":  # touch C
            assert pow == 2, "Method C only implemented for standard parabola (power=2)."
            return ParabolaApproximation._approximate_parabola_touch_C(
                f,
                x1_start, x2_end,  # defines domain
                x1_touch, x2_touch,  # defines touching points
                C, N_grid
            )
        ###############################################
        elif method == "D":  # THIS IS THE MAIN METHOD
            assert isinstance(pow, int) and pow >= 1
            return ParabolaPowApproximation._approximate_parabola_touch_C(
                f, pow,
                x1_start, x2_end,  # defines domain
                x1_touch, x2_touch,  # defines touching points
                C, N_grid,
                conservative=any("SAFE" in interval for interval in h['interval_types']),
                k_plus=options['k_plus'],
                k_minus=options['k_minus']
            )
        elif method == "TWO_DIFF_TOUCH":
            assert pow == 2, "Method TWO_DIFF_TOUCH only implemented for standard parabola (power=2)."
            alpha = options['alpha']
            beta = options['beta']
            return ParabolaTwoDifferentTouchPointsApproximation._approximate_parabola_touch_ALPHA_BETA(
                f,
                x1_start, x2_end,
                x1_touch, x2_touch,
                alpha, beta,
                N_grid
            )
        else:
            raise ValueError(f"Unknown method {method} for parabola approximation.")

    @staticmethod
    def _try_C(x: np.ndarray, y: np.ndarray,
               x1_touch: float, x2_touch: float,
               C: float, zero_tol: float = 1e-12,
               b2_min_cut: float = 1e-12) -> Optional[Tuple[float, float, float]]:
        """
        Try to build the parabola for given C.
        Returns (a, b, c) if feasible, else None.

        This version chooses the *smallest* feasible b (i.e. steepest parabola).
        """
        a = 0.5 * (x1_touch + x2_touch)
        d = 0.5 * (x2_touch - x1_touch)

        numer = (x - a) ** 2 - d ** 2  # N(x)
        denom = y - C  # D(x)

        # 1) Points where D == 0: must have N <= 0
        zero_mask = np.abs(denom) <= zero_tol
        if np.any(zero_mask & (numer > 0.0)):
            return None

        # 2) safe division for nonzero denom
        nonzero_mask = ~zero_mask
        S = np.full_like(numer, np.nan, dtype=np.float64)
        S[nonzero_mask] = numer[nonzero_mask] / denom[nonzero_mask]

        # 3) lower bounds come from D > 0: b^2 >= S(x)
        pos_mask = denom > zero_tol
        if np.any(pos_mask):
            L = np.max(S[pos_mask])
        else:
            L = -np.inf

        # 4) upper bounds come from D < 0: b^2 <= S(x)
        neg_mask = denom < -zero_tol
        if np.any(neg_mask):
            # If any S on negative-denom side is <= 0 then infeasible (upper bound non-positive)
            if np.any(S[neg_mask] <= 0.0):
                return None
            U = np.min(S[neg_mask])
        else:
            U = np.inf

        # 5) feasibility check: L <= U and some positive b^2 in [L, U]
        if not (L <= U):
            return None

        # pick smallest feasible positive b^2 (steepest): b2 = max(L, b2_min_cut)
        b2_candidate = max(L, b2_min_cut)

        # ensure the chosen candidate respects the upper bound U
        if b2_candidate > U:
            return None

        b2 = float(b2_candidate)
        b = float(np.sqrt(b2))
        c = float((d * d) / b2 - C)

        # final safety check on the discrete grid with a tiny tolerance
        approx_vals = ((x - a) ** 2) / b2 - c
        if np.any(y + 1e-12 < approx_vals):
            return None

        return (a, b, c)

    @staticmethod
    def _approximate_parabola_only_compare_lt_C_regions(f: Callable[[float], float], x1, x2, C: float,
                                                        N_grid: int = 2001) -> \
            Optional[Tuple[float, float, float]]:
        """
        Approximate f by tilde f(x) = (x-a)^2 / b^2 - c s.t.
          - for all x in [x1 , x2]: f(x) >= tilde f(x). !! No requirement outside this interval. !!
          - and tilde f(x1) = tilde f(x1) = C
        """

        # Sample grid
        x = np.linspace(x1, x2, N_grid)
        y = np.array([f(xx) for xx in x], dtype=np.float64)  # f(x)
        fx1, fx2 = f(x1), f(x2)
        assert (fx1 > 20 or x1 == 0) and (
                fx2 > 20 or x2 == 1), "f(x1) and f(x2) must be larger than 20 due to quasiconvexity not being strict."
        C = min(f(x1), f(x2), C)  # should be close to C, but now a little bit smaller

        zero_tol = 1e-12

        a = 0.5 * (x1 + x2)
        d = 0.5 * (x2 - x1)

        numer = (x - a) ** 2 - d ** 2
        denom = y - C

        # Step 1: any exact/near-zero denom where numer > 0 => infeasible
        zero_mask = np.abs(denom) <= zero_tol
        if np.any(zero_mask & (numer > 0.0)):
            raise Exception("Should not have any D~0 with N>0 here?")

        # Step 2: compute S everywhere denom != 0 (including small denominators).
        # Use safe division; allow +/- inf if denom ~ 0.
        nonzero_mask = ~zero_mask

        S = np.full_like(numer, np.nan, dtype=np.float64)
        S[nonzero_mask] = numer[nonzero_mask] / denom[nonzero_mask]  # S = N / D

        # if denom is neg: b2 should be <= S
        # if denom is pos: b2 should be >= S (happens rarely, but will happen due to C=min(f(x1), f(x2), C) adjustment)
        # choose b2 as the smallest feasible positive value, filtering out the positive-denom regions
        pos_mask = denom > 0.0
        min_b2_candidates = S[~pos_mask]  # only consider S where denom <= 0

        min_b2_candidates = min_b2_candidates[np.isfinite(min_b2_candidates) & (min_b2_candidates > 0.0)]
        if len(min_b2_candidates) == 0:
            # raise Exception("Should have some feasible b^2 candidates here?")
            return None

        b2 = np.min(min_b2_candidates)
        b2 = np.nextafter(b2,
                          np.inf)  # move to the next representable float upward for numerical safety (not strictly needed)

        # also ensure the rare positive-denom regions are satisfied
        max_b2_candidates = S[pos_mask]
        max_b2_candidates = max_b2_candidates[np.isfinite(max_b2_candidates)]
        if len(max_b2_candidates) > 0:
            assert b2 >= np.max(max_b2_candidates), "Chosen b^2 does not satisfy positive-denom constraints."

        b = float(np.sqrt(b2))
        c = float((d * d) / b2 - C)

        f_approx = lambda x_val: parabola(x_val, a, b, c, pow=2)
        ParabolaApproximation._sanity_check_pts_leq(f, f_approx, x1, x2, N_grid)

        return (a, b, c)

    @staticmethod
    def _approximate_parabola_touch_C(f: Callable[[float], float], x1, x2, x1_touch: float, x2_touch: float,
                                      C: float,
                                      N_grid: int = 2001, adaptive: bool = True) -> \
            Optional[Tuple[float, float, float]]:
        """
        Wrapper that uses binary search on x (touch pts). let \bar x = 0.5 * (x1_touch + x2_touch).
        will iteratively try larger/smaller distances from \bar x until a feasible parabola is found.
        """

        # Sample grid once
        xs = np.linspace(x1, x2, N_grid)
        ys = np.array([f(xx) for xx in xs], dtype=np.float64)

        # check if ys are constant 0.5, then no feasible parabola. I didn't bother to handle this case properly earlier
        if np.all(np.abs(ys - ys[0]) <= 1e-12):
            # print("Warning: f appears to be constant; no feasible parabola exists.")
            return None

        C = min(f(x1_touch), f(x2_touch), C)
        # print(f"Adjusted C for parabola fit: {C}")

        # Define initial distance
        low_dist = 0
        max_dist = x2 - x1

        tol = 1e-4  # for binary search
        max_iters = 50

        best_parabola = None
        best_dist = max_dist

        ### First check if low_dist is feasible, can immediately return
        res = ParabolaApproximation._try_touch_pts(
            xs, ys, x1, x2,
            x1_touch - low_dist, x2_touch + low_dist,
            C)

        if (res is not None) or (not adaptive):
            return res

        ### Also assert that max_dist produces a feasible parabola, otherwise there is no point searching
        res = ParabolaApproximation._try_touch_pts(
            xs, ys, x1, x2,
            max(x1_touch - max_dist, x1), min(x2_touch + max_dist, x2),  # ensure within [x1,x2]
            C)
        assert res is not None, "No feasible parabola found even with max distance for touching points."

        ### Binary search for smallest feasible distance
        lo, hi = low_dist, max_dist  # lo is infeasible, hi is feasible, but lo is preferred
        for i in range(max_iters):
            # print(f"BS iter {i}: lo={lo}, hi={hi}")
            mid = 0.5 * (lo + hi)  # new distance to try
            res_mid = ParabolaApproximation._try_touch_pts(
                xs, ys, x1, x2,
                max(x1_touch - mid, x1), min(x2_touch + mid, x2),  # ensure within [x1,x2]
                C)

            if res_mid is not None:
                # mid feasible -> move hi down
                best_parabola = res_mid
                best_dist = mid
                hi = mid
            else:
                # mid infeasible -> increase distance
                lo = mid

            if (hi - lo) <= tol * max(1.0, abs(hi)):
                break

        # print(f"Chosen touching point distance: {best_dist}")
        a, b, c = best_parabola
        f_approx = lambda x_val: parabola(x_val, a, b, c, pow=2)
        ParabolaApproximation._sanity_check_pts_leq(f, f_approx, x1, x2, N_grid)

        return best_parabola

    @staticmethod
    def _try_touch_pts(xs: ndarray, ys: ndarray, x1, x2, x1_touch: float, x2_touch: float,
                       C: float) -> \
            Optional[Tuple[float, float, float]]:
        """
        Approximate f by tilde f(x) = (x-a)^2 / b^2 - c s.t.
          - for all x in [x1 , x2]: f(x) >= tilde f(x). !! No requirement outside this interval. !!
          - !! and tilde f(**x1_touch**) = tilde f(**x1_touch**) = C
        """

        # x1, x2 define domain. (likely f(x1)=D*C)
        # x1_touch, x2_touch define touching points where f(x)=C

        zero_tol = 1e-12

        #### actual fitting logic
        a = 0.5 * (x1_touch + x2_touch)
        d = 0.5 * (x2_touch - x1_touch)
        numer = (xs - a) ** 2 - d ** 2
        denom = ys - C

        ### checking if any b2 satisfies all constraints:

        # if denom == 0: numer must be <= 0
        zero_mask = (np.abs(denom) <= zero_tol)
        if np.any(zero_mask & (numer > 0.0)):
            raise Exception("Should not have any D~0 with N>0 here!")

        # Step 2: compute S everywhere denom != 0 (including small denominators).
        # Use safe division; allow +/- inf if denom ~ 0.
        nonzero_denom_mask = ~zero_mask
        S = np.full_like(numer, np.nan, dtype=np.float64)  # S = numer/denom
        S[nonzero_denom_mask] = numer[nonzero_denom_mask] / denom[nonzero_denom_mask]  # S = N / D

        ### denom < 0: b2 <= S. (relevant for xs between touching points)
        # if denom is neg: b2 should be <= S
        pos_mask = denom > 0.0
        min_b2_candidates = S[~pos_mask]  # gives upperbounds on b2

        min_b2_candidates = min_b2_candidates[np.isfinite(min_b2_candidates) & (min_b2_candidates > 0.0)]
        if len(min_b2_candidates) == 0:  # probably in case of f(x) = const, then no constraints and np.isfinite filters all out but i accept for now. in practice not really a problem
            print("Warning: No feasible upper-bound candidates for b^2 found.")
            return None

        ###  denom > 0: b2 >= S. (relevant for xs outside touching points)
        max_b2_candidates = S[pos_mask]
        max_b2_candidates = max_b2_candidates[np.isfinite(max_b2_candidates)]
        if len(max_b2_candidates) == 0:
            # print("Warning: No lower-bound candidates for b^2 found; assuming no lower bound.")
            return None

        # assert the set of legal b2 is non-empty
        largest_legal_b2 = np.min(min_b2_candidates)
        smallest_legal_b2 = np.max(max_b2_candidates) if len(max_b2_candidates) > 0 else zero_tol  # 0.0
        # assert largest_legal_b2 >= smallest_legal_b2, "No feasible b^2 exists that satisfies all constraints."
        if largest_legal_b2 < smallest_legal_b2:
            # print("Warning: No feasible b^2 exists that satisfies all constraints.")
            return None

        # select b2 as the midpoint between feasible min and max
        # b2 = np.min(min_b2_candidates)  # choose steepest parabola
        b2 = 0.5 * (np.min(min_b2_candidates) + np.max(max_b2_candidates))
        # b2 = np.max(max_b2_candidates)  # choose flattest parabola

        b = float(np.sqrt(b2))
        c = float((d * d) / b2 - C)

        return (a, b, c)

    @staticmethod
    def _sanity_check_pts_leq(f: Callable[[float], float], f_approx: Callable[[float], float],
                              x1: float, x2: float, N_grid: int, tol: float = 1e-8) -> bool:
        for method in ['lin']:  # , 'rnd']:
            if method == 'lin':
                xs = np.linspace(x1, x2, N_grid)
            else:
                # sample a few points in [0,1] to check f(x) >= (x-a)^2 / b^2 - c
                rng = np.random.default_rng(seed=42)
                xs = rng.uniform(x1, x2, size=1000)

            for x in xs:
                f_orig_pt = f(x)
                f_approx_pt = f_approx(x)
                # f_approx = (x - a) ** 2 / (b ** 2) - c
                assert f_orig_pt + tol >= f_approx_pt, f"{method=} - Parabola approximation violated at x={x}: f(x)={f_orig_pt}, approx={f_approx_pt}"

        return True


class ParabolaTwoDifferentTouchPointsApproximation:
    @staticmethod
    def _approximate_parabola_touch_ALPHA_BETA(
            f: Callable[[float], float],
            x1, x2,
            x1_touch: float, x2_touch: float,
            alpha: float, beta: float,
            N_grid: int = 2001, adaptive: bool = True
    ) -> Optional[Tuple[float, float, float]]:

        # Sample grid once
        xs = np.linspace(x1, x2, N_grid)
        ys = np.array([f(xx) for xx in xs], dtype=np.float64)

        # check trivial constant-case
        if np.all(np.abs(ys - ys[0]) <= 1e-12):
            # print("Warning: f appears to be constant; no feasible parabola exists.")
            return None

        # initial search parameters (distance we expand around touching points)
        low_dist = 0.0
        max_dist = x2 - x1

        tol = 1e-4  # for binary search
        max_iters = 50

        best_parabola = None
        best_dist = max_dist

        # first try with low_dist
        res = ParabolaTwoDifferentTouchPointsApproximation._try_touch_pts(
            xs, ys, x1, x2,
            x1_touch - low_dist, x2_touch + low_dist,
            alpha, beta
        )
        if (res is not None) or (not adaptive):
            return res

        # ensure feasibility at max_dist (otherwise no point searching)
        res = ParabolaTwoDifferentTouchPointsApproximation._try_touch_pts(
            xs, ys, x1, x2,
            max(x1_touch - max_dist, x1),
            min(x2_touch + max_dist, x2),
            alpha, beta
        )
        assert res is not None, "No feasible parabola found even with max distance for touching points."

        # binary search smallest feasible distance
        lo, hi = low_dist, max_dist
        for i in range(max_iters):
            mid = 0.5 * (lo + hi)
            res_mid = ParabolaTwoDifferentTouchPointsApproximation._try_touch_pts(
                xs, ys, x1, x2,
                max(x1_touch - mid, x1),
                min(x2_touch + mid, x2),
                alpha, beta
            )
            if res_mid is not None:
                best_parabola = res_mid
                best_dist = mid
                hi = mid
            else:
                lo = mid

            if (hi - lo) <= tol * max(1.0, abs(hi)):
                break

        if best_parabola is None:
            return None

        a, b, c = best_parabola
        f_approx = lambda x_val: parabola(x_val, a, b, c, pow=2)
        ParabolaApproximation._sanity_check_pts_leq(f, f_approx, x1, x2, N_grid)

        return best_parabola

    @staticmethod
    def _try_touch_pts(
            xs: ndarray, ys: ndarray,
            x1, x2, x1_touch: float, x2_touch: float,
            alpha: float, beta: float
    ) -> Optional[Tuple[float, float, float]]:
        """
        Approximate f by tilde f(x) = (x-a)^2 / b^2 - c s.t.
          - for all x in [x1 , x2]: f(x) >= tilde f(x).
          - tilde f(x1_touch) = alpha, tilde f(x2_touch) = beta.
        """

        zero_tol = 1e-12

        # handle degenerate touching points
        if abs(x2_touch - x1_touch) <= 1e-15:
            # touching points too close / identical -> fallback to symmetric case
            a_mid = 0.5 * (x1_touch + x2_touch)
            # behave like alpha==beta: but here alpha and beta must be equal-ish
            if abs(alpha - beta) > 1e-9:
                # very ill-posed: cannot determine k
                return None
            d = 0.5 * (x2_touch - x1_touch)
            numer = (xs - a_mid) ** 2 - d ** 2
            denom = ys - alpha
            # delegate to old logic for symmetric case (not included here)
            # but for brevity, return None to indicate no solution found
            return None

        # compute k and midpoint alpha0 (names consistent with derivation)
        k = (beta - alpha) / (2.0 * (x2_touch - x1_touch))
        midpoint = 0.5 * (x1_touch + x2_touch)

        # For each x: constraint is
        #   b^2 * D(x) >= RHS(x),
        # where RHS(x) = (x-x1_touch)*(x-x2_touch)
        # and D(x) = f(x) - alpha - 2*k*(x - x1_touch)

        RHS = (xs - x1_touch) * (xs - x2_touch)
        D = ys - alpha - 2.0 * k * (xs - x1_touch)

        # case D ~= 0: require RHS <= 0 (since left side is 0)
        zero_mask = np.abs(D) <= zero_tol
        if np.any(zero_mask & (RHS > 0.0)):
            # infeasible because 0 >= positive RHS
            return None

        # D > 0 => b^2 >= RHS / D  (lower bounds)
        pos_mask = D > zero_tol
        lb_vals = np.array([], dtype=np.float64)
        if np.any(pos_mask):
            lb_vals = RHS[pos_mask] / D[pos_mask]

        # D < 0 => b^2 <= RHS / D  (upper bounds)
        neg_mask = D < -zero_tol
        ub_vals = np.array([], dtype=np.float64)
        if np.any(neg_mask):
            ub_vals = RHS[neg_mask] / D[neg_mask]

        # Filter only finite candidates
        lb_vals = lb_vals[np.isfinite(lb_vals)]
        ub_vals = ub_vals[np.isfinite(ub_vals)]

        # We require b^2 >= 0. Lower bounds below 0 are irrelevant; we clamp them to 0
        if lb_vals.size > 0:
            LB = max(0.0, float(np.max(lb_vals)))
        else:
            LB = 0.0

        # Upper bound: if no ub_vals, then no explicit finite upper bound (set +inf)
        if ub_vals.size > 0:
            UB = float(np.min(ub_vals))
        else:
            UB = np.inf

        # If UB <= 0 then impossible to pick positive b^2
        if UB <= 0.0:
            return None

        # Final feasibility check
        if LB > UB:
            return None

        # choose b^2 (midpoint between LB and UB if UB finite, else LB + a small margin)
        if np.isfinite(UB):
            b2 = 0.5 * (LB + UB)
        else:
            # no upper bound: pick b2 = max(LB, small)
            b2 = max(LB, 1e-12)

        # ensure positive
        if b2 <= 0.0:
            b2 = 1e-12

        # compute a and c
        a = midpoint - k * b2
        b = float(np.sqrt(b2))
        c = float((x1_touch - a) ** 2 / b2 - alpha)

        return (a, b, c)


class ParabolaPowApproximation:  # (x-a)^p / b^p - c
    @staticmethod
    def _approximate_parabola_touch_C(f: Callable[[float], float], pow: int, x1, x2, x1_touch: float, x2_touch: float,
                                      C: float,
                                      N_grid: int = 2001, adaptive: bool = True, conservative: bool = False,
                                      k_plus: Optional[Callable[[float], float]] = None,
                                      k_minus: Optional[Callable[[float], float]] = None) -> \
            Optional[Tuple[float, float, float]]:

        assert isinstance(pow, int) and pow >= 1

        # Sample grid once
        xs = np.linspace(x1, x2, N_grid)
        ys = np.array([f(xx) for xx in xs], dtype=np.float64)

        xs_backup = xs.copy()
        ys_backup = ys.copy()

        if conservative:
            xs, ys = construct_conservative_k(k_plus, k_minus, x1, x2, xs, f)

        # check if ys are constant 0.5, then no feasible parabola. I didn't bother to handle this case properly earlier
        if np.all(np.abs(ys - ys[0]) <= 1e-12):
            # print("Warning: f appears to be constant; no feasible parabola exists.")
            return None

        C = min(f(x1_touch), f(x2_touch), C)
        if conservative:  # adjust C further downwards if needed
            zero_tol = 1e-12
            outside_mask = (xs < x1_touch - zero_tol) | (xs > x2_touch + zero_tol)
            outside_vals = ys[outside_mask]
            if outside_vals.size > 0 and np.any(outside_vals <= C + zero_tol):
                min_outside = float(np.min(outside_vals))
                margin = max(1e-8, 1e-6 * max(1.0, abs(min_outside)))
                C = min(C, min_outside - margin)

        # Define initial distance
        low_dist = 0
        max_dist = x2 - x1

        tol = 1e-4  # for binary search
        max_iters = 50

        best_parabola = None
        best_dist = max_dist

        ### First check if low_dist is feasible, can immediately return
        res = ParabolaPowApproximation._try_touch_pts(
            xs, ys,
            x1_touch - low_dist, x2_touch + low_dist,
            C, pow)

        if (res is not None) or (not adaptive):
            return res

        ### Also assert that max_dist produces a feasible parabola, otherwise there is no point searching
        res = ParabolaPowApproximation._try_touch_pts(
            xs, ys,
            max(x1_touch - max_dist, x1),
            min(x2_touch + max_dist, x2),  # ensure within [x1,x2]
            C, pow)

        assert res is not None, "No feasible parabola found even with max distance for touching points. This could also mean a solution was found but under no constraints"

        ### Binary search for smallest feasible distance
        lo, hi = low_dist, max_dist  # lo is infeasible, hi is feasible, but lo is preferred
        for i in range(max_iters):
            # print(f"BS iter {i}: lo={lo}, hi={hi}")
            mid = 0.5 * (lo + hi)  # new distance to try
            res_mid = ParabolaPowApproximation._try_touch_pts(
                xs, ys,
                max(x1_touch - mid, x1), min(x2_touch + mid, x2),  # ensure within [x1,x2]
                C, pow)

            if res_mid is not None:
                # mid feasible -> move hi down
                best_parabola = res_mid
                best_dist = mid
                hi = mid
            else:
                # mid infeasible -> increase distance
                lo = mid

            if (hi - lo) <= tol * max(1.0, abs(hi)):
                break

        a, b, c = best_parabola
        f_approx = lambda x_val: parabola(x_val, a, b, c, pow=pow)
        ParabolaApproximation._sanity_check_pts_leq(f, f_approx, x1, x2, N_grid)

        return best_parabola

    @staticmethod
    def _try_touch_pts(xs: ndarray, ys: ndarray, x1_touch: float, x2_touch: float,
                       C: float, pow: int) -> \
            Optional[Tuple[float, float, float]]:
        """
        Approximate f by tilde f(x) = (x-a)^2 / b^2 - c s.t.
          - for all x in [x1 , x2]: f(x) >= tilde f(x). !! No requirement outside this interval. !!
          - !! and tilde f(**x1_touch**) = tilde f(**x1_touch**) = C
        """

        # x1, x2 define domain. (likely f(x1)=D*C)
        # x1_touch, x2_touch define touching points where f(x)=C

        zero_tol = 1e-12

        #### actual fitting logic
        a = 0.5 * (x1_touch + x2_touch)
        d = 0.5 * (x2_touch - x1_touch)
        # numer = (xs - a) ** pow - d ** pow
        numer = np.abs(xs - a) ** pow - np.abs(d) ** pow
        denom = ys - C

        ### checking if any bpow satisfies all constraints:

        # if denom == 0: numer must be <= 0
        zero_mask = (np.abs(denom) <= zero_tol)
        if np.any(zero_mask & (numer > (0.0 + 1e-15))):  # I needed to add the 1e-15 here to avoid numerical issues
            print(f"Debug info: numer={numer[zero_mask & (numer > 0.0)]}, denom={denom[zero_mask & (numer > 0.0)]}")
            raise Exception("Should not have any D~0 with N>0 here! This indicates no feasible parabola exists.")

        # Step 2: compute S everywhere denom != 0 (including small denominators).
        # Use safe division; allow +/- inf if denom ~ 0.
        nonzero_denom_mask = ~zero_mask
        S = np.full_like(numer, np.nan, dtype=np.float64)  # S = numer/denom
        S[nonzero_denom_mask] = numer[nonzero_denom_mask] / denom[nonzero_denom_mask]  # S = N / D

        ### denom < 0: b^pow <= S. (relevant for xs between touching points)
        # if denom is neg: b^pow should be <= S
        pos_mask = denom > 0.0
        min_bpow_candidates = S[~pos_mask]  # gives upperbounds on b^pow

        min_bpow_candidates = min_bpow_candidates[np.isfinite(min_bpow_candidates) & (min_bpow_candidates > 0.0)]
        if len(min_bpow_candidates) == 0:  # probably in case of f(x) = const, then no constraints and np.isfinite filters all out but i accept for now. in practice not really a problem
            print("Warning: No feasible upper-bound candidates for b^2 found.")
            return None

        ###  denom > 0: bpow >= S. (poses constraints for xs outside touching points)
        max_bpow_candidates = S[pos_mask]
        max_bpow_candidates = max_bpow_candidates[np.isfinite(max_bpow_candidates)]
        if len(max_bpow_candidates) == 0:
            # print("Warning: No lower-bound candidates for b^2 found; assuming no lower bound.")
            # print("This can happen when no values outside touching points have f(x) > C.")
            # return None
            pass

        # assert the set of legal bpow is non-empty
        largest_legal_bpow = np.min(min_bpow_candidates)
        smallest_legal_bpow = np.max(max_bpow_candidates) if len(max_bpow_candidates) > 0 else zero_tol  # 0.0
        # assert largest_legal_bpow >= smallest_legal_bpow, "No feasible b^2 exists that satisfies all constraints."
        if largest_legal_bpow < smallest_legal_bpow:
            # print("Warning: No feasible b^2 exists that satisfies all constraints.")
            return None

        # if len(max_bpow_candidates) was zero, average doesn't make sense; just pick min candidate
        if len(max_bpow_candidates) == 0:
            bpow = np.min(min_bpow_candidates)
        else:  # conventional case:
            # select bpow as the midpoint between feasible min and max
            # bpow = np.min(min_bpow_candidates)  # choose steepest parabola
            bpow = 0.5 * (np.min(min_bpow_candidates) + np.max(max_bpow_candidates))
            # bpow = np.max(max_bpow_candidates)  # choose flattest parabola

        b = float(bpow ** (1.0 / pow))
        c = float((np.abs(d) ** pow) / bpow - C)

        return (a, b, c)


import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Test conservative k construction vs original function

    # Parameter for mixing
    theta = 0.6


    def k_plus(x):
        # Non-increasing function: starts high, ends low
        return theta * (3.0 - 2.0 * x)


    def k_minus(x):
        # Non-decreasing function: starts low, ends high
        return (1 - theta) * (0.5 + 2.5 * x)


    # Define test function as max(theta * k_plus(x), (1-theta) * k_minus(x))
    def f(x):
        return max(k_plus(x), k_minus(x))


    # Domain
    x1, x2 = 0.0, 1.0
    N_grid = 50

    # Original grid and function values
    xs_original = np.linspace(x1, x2, N_grid)
    ys_original = np.array([f(xx) for xx in xs_original], dtype=np.float64)

    # Construct conservative k values
    xs_conservative, ys_conservative = construct_conservative_k(
        k_plus=k_plus,
        k_minus=k_minus,
        x1=x1,
        x2=x2,
        xs=xs_original.copy(),
        f=f
    )

    # Also compute k_plus and k_minus values for visualization
    k_plus_vals = np.array([k_plus(xx) for xx in xs_original], dtype=np.float64)
    k_minus_vals = np.array([k_minus(xx) for xx in xs_original], dtype=np.float64)
    k_max_vals = np.array([max(k_plus(xx), k_minus(xx)) for xx in xs_original], dtype=np.float64)

    plt.plot(xs_original, ys_original, label='Original f(x)', linewidth=2, color='blue', alpha=0.5)
    plt.scatter(xs_conservative, ys_conservative, label='Conservative k(x)', linewidth=2, color='red')

    plt.xlabel('x')
    plt.ylabel('Value')
    plt.title('Original Function and Conservative Construction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # def f(x):
    #     a = (x - 0.1) ** 8
    #     b = (x - 0.1 - 1) ** 2
    #     return 2 * (a + b) - 0.2
    # xs = np.linspace(0, 1, 1001)
    # ys = np.array([f(x) for x in xs], dtype=np.float64)
    #
    # f_extended, x1_extended, x2_extended = extend_domain_for_parabola_fit(
    #     f, x1_touch=0.325, x2_touch=1.0, C=1.0
    # )
    #
    # xs_ext = np.linspace(x1_extended, x2_extended, 1001)
    # ys_ext = np.array([f_extended(x) for x in xs_ext], dtype=np.float64)
    #
    # plt.plot(xs, ys, label='f(x)')
    # plt.plot(xs_ext, ys_ext, label='extended f(x)', linestyle='--')
    # plt.show()

    # Compute approximate parabola

    # def f(x):
    #     # a = (x - 0.1) ** 8
    #     # b = (x - 0.1 - 1) ** 2
    #     # return 2 * (a + b) - 0.2
    #     return 5*(x-0.5)**2 + 0.5

    plt.plot(np.linspace(0, 1, 100), [f(x) for x in np.linspace(0, 1, 100)], label='f(x)')
    plt.grid()
    plt.yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    plt.show()

    xs = np.linspace(0, 1, 1001)
    ys = np.array([f(x) for x in xs], dtype=np.float64)
    h = {'interval_types': 'HEDGE_nd_ELLIP_SAFE', 'extend_f_domain_for_parabola_fit': False}
    options = {'parabola_power': 2, 'k_plus': k_plus, 'k_minus': k_minus}
    result = ParabolaApproximation.approximate_parabola(  # C= 1.5
        h, f, x1_start=0.0, x2_end=1.0, x1_touch=0.6, x2_touch=0.8, C=1.5, method="D", options=options
    )


    # Extract parameters (a, b, c) if the approximation was successful
    if result:
        a, b, c = result


        # Define the approximate parabola function
        def approx_parabola(x):
            return parabola(x, a, b, c, pow=options['parabola_power'])
            # return (np.abs(x - a) ** 5) / (b ** 5) - c


        # Plot the original function and the approximate parabola
        plt.plot(xs, ys, label='f(x)')
        plt.plot(xs, [approx_parabola(x) for x in xs], label='approximate parabola', linestyle='--')
        plt.legend()
        plt.show()
    else:
        print("Failed to compute approximate parabola.")

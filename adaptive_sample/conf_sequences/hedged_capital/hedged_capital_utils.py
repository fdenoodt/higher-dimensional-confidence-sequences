import math
from typing import Callable, Tuple


_LOG_MAX_FLOAT = math.log(float.fromhex("0x1.fffffffffffffp+1023"))


def _safe_exp(log_value: float) -> float:
    if log_value == -math.inf:
        return 0.0
    if log_value >= _LOG_MAX_FLOAT:
        return math.inf
    return math.exp(log_value)


def find_minimum_of_k(
        k_plus: Callable[[float], float], # decreasing
        k_minus: Callable[[float], float], # increasing
        a: float,
        b: float,
        *,
        log_scale: bool = False,
        eps_x: float = 1e-10,
        eps_y: float = 1e-12,
        max_iter: int = 10_000,
) -> Tuple[float, float, float, float]:
    """
    !!! EXPECTS K_PLUS AND K_MINUS TO BE WEIGHTED ALREADY, SO k(x) = max(k_minus(x), k_plus(x)).
    Assumptions (not checked):
      - k_minus(x) >= 0 and is continuous and non-decreasing on [a,b]
      - k_plus(x) >= 0 and is continuous and non-increasing on [a,b]
    Define k(x) = max(k_minus(x), k_plus(x)).
    Returns (k_lower, k_upper, k_est) such that the true minimum of k(x) on [a,b]
    lies in [k_lower, k_upper], and k_est = 0.5*(k_lower+k_upper).

    Notes:
      - If the curves do not intersect on [a,b] (in the sense h(a)>0 or h(b)<0
        where h(x)=k_minus(x)-k_plus(x)), the minimizer is at an endpoint.
    """
    if not (a < b):
        raise ValueError(f"Require a < b, got a={a}, b={b}")
    if eps_x <= 0 or eps_y <= 0:
        raise ValueError("eps_x and eps_y must be positive.")

    def h(x: float) -> float:
        if log_scale:
            return k_minus(x) - k_plus(x)
        return k_minus(x) - k_plus(x)

    def k(x: float) -> float:
        if log_scale:
            return _safe_exp(max(k_minus(x), k_plus(x)))
        return max(k_minus(x), k_plus(x))

    ha = h(a)
    hb = h(b)
    if not (math.isfinite(ha) and math.isfinite(hb)):
        raise ValueError("Non-finite values encountered at the interval endpoints.")

    # No intersection (or not bracketed as expected) -> endpoint solution.
    # (Matches: if h(a)>0 or h(b)<0 then return argmin{k(a),k(b)}.)
    if ha > 0.0 or hb < 0.0:
        ka = k(a)
        kb = k(b)
        if not (math.isfinite(ka) and math.isfinite(kb)):
            raise ValueError("Non-finite k encountered at the interval endpoints.")
        k_star = ka if ka <= kb else kb
        # return k_star, k_star, k_star  # bounds collapse
        # k_lower, k_upper, a, b
        return k_star, k_star, a, b

    # Now h(a) <= 0 <= h(b): intersection exists (possibly an interval).
    # Bisection on h, while tightening bounds on the minimum value.
    k_lower = - math.inf
    k_upper = math.inf

    it = 0
    while (b - a > eps_x) or (k_upper - k_lower > eps_y):
        it += 1
        if it > max_iter:
            # Return best bounds we have so far.
            break

        m = 0.5 * (a + b)
        hm = h(m)
        if not math.isfinite(hm):
            raise ValueError("Non-finite h encountered during bisection.")

        # Update bracket to keep h(a) <= 0 <= h(b)
        if hm < 0.0:
            a = m  # keep h(a) <= 0
        elif hm > 0.0:
            b = m  # keep h(b) >= 0 (also covers hm == 0)
        else:
            x_star = m
            k_star = k(x_star)
            return k_star, k_star, x_star, x_star

        # Update bounds on the minimum value
        if log_scale:
            k_lower = _safe_exp(max(k_minus(a), k_plus(b)))
            k_upper = _safe_exp(min(k_minus(b), k_plus(a)))
        else:
            k_lower = max(k_minus(a), k_plus(b))
            k_upper = min(k_minus(b), k_plus(a))

        if not (math.isfinite(k_lower) and math.isfinite(k_upper)):
            raise ValueError("Non-finite bounds encountered during iteration.")

    # Final bounds and estimate
    if log_scale:
        k_lower = _safe_exp(max(k_minus(a), k_plus(b)))
        k_upper = _safe_exp(min(k_minus(b), k_plus(a)))
    else:
        k_lower = max(k_minus(a), k_plus(b))
        k_upper = min(k_minus(b), k_plus(a))

    return k_lower, k_upper, a, b


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    def k_plus(x):
        a = (x - 0.1) ** 8
        b = (x - 0.1 - 1) ** 2
        return 2 * (a + b) - 0.2
        # return 0.7


    def k_minus(x):
        return -0.9 * (x - 0.9) ** 2 + 1.0
        # return 0.7


    def k_max(x):
        return max(k_plus(x), k_minus(x))


    xs = np.linspace(0, 1, 1001)
    ys = np.array([k_max(x) for x in xs], dtype=np.float64)

    tol = 1e-10
    lb, ub, a, b = find_minimum_of_k(k_plus, k_minus, 0.0, 1.0, eps_x=tol, eps_y=tol)  # 1e-8)

    plt.plot(xs, ys, label='k(x)')
    plt.plot(xs, [k_plus(x) for x in xs], '--', label='k_plus(x)')
    plt.plot(xs, [k_minus(x) for x in xs], '--', label='k_minus(x)')

    plt.axvline(a, color='red', linestyle=':', label='a (min bound)')
    plt.axvline(b, color='green', linestyle=':', label='b (min bound)')

    plt.axhline(lb, color='orange', linestyle='--', label='k_lower')
    plt.axhline(ub, color='purple', linestyle='--', label='k_upper')

    plt.legend()
    plt.show()

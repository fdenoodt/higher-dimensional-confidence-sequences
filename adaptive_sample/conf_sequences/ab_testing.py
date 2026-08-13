from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from adaptive_sample.conf_sequences.interval_factory import IntervalFactory
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_bonf import (
    HedgedCapitalCIMultiDimensionalBonferroni,
)
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_convex_sum import (
    HedgedCapitalConvexSum,
)
from adaptive_sample.conf_sequences.hedged_capital.hedged_factory import HedgedFactory
from adaptive_sample.helper.volume import VolumeCalculator


def get_batched_ab_stream(
        T: int = 200,
        n_per_arm: int = 500,
        seed: int = 0,
        noise_dims: int = 2,
        # Which (named) metrics are truly affected by treatment:
        effect_metrics: Sequence[str] = ("time", "clicks", "scroll", "add_to_cart"),
        # Strength is on the *latent engagement shift* used only for affected metrics.
        # Values around 0.03–0.08 are reasonable; larger => quicker detection.
        engagement_shift: float = 0.06,
        variable_batch_sizes: bool = True,
        min_batch: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """
    Returns:
        Z: (T, D) array in [0,1], where Z_t = 0.5 * ( (mean_B - mean_A) + 1 ).
           Null corresponds to 0.5 per coordinate.
        meta: dict with metric names, batch sizes, and diagnostic arrays.

    Dimensions:
        Base metrics: 8
        + noise_dims (default 2) => D = 10 by default.
    """

    rng = np.random.default_rng(seed)

    # ---------- Metric configuration ----------
    # Caps for clipping (raw -> bounded)
    CAPS = {
        "time_sec": 300.0,
        "clicks": 20.0,
        "pages": 10.0,
    }

    base_metric_names = [
        "time",  # scaled time_sec / 300
        "clicks",  # scaled clicks / 20
        "pages",  # scaled pages / 10
        "scroll",  # already in [0,1]
        "add_to_cart",  # in {0,1}
        "purchase",  # in {0,1}
        "non_bounce",  # in {0,1}, equals 1 - bounce
        "return_7d",  # in {0,1}
    ]
    noise_metric_names = [f"noise_{j + 1}" for j in range(noise_dims)]
    metric_names = base_metric_names + noise_metric_names

    name_to_idx = {n: i for i, n in enumerate(base_metric_names)}

    # Which base metrics get treatment effect
    effect_idx = [name_to_idx[m] for m in effect_metrics if m in name_to_idx]

    D_base = len(base_metric_names)
    D = D_base + noise_dims

    # ---------- Session simulator ----------
    def _simulate_sessions(n: int, arm: str) -> np.ndarray:
        """
        Simulate n sessions for arm in {"A","B"}.
        Returns scaled session-level X in [0,1]^(D_base + noise_dims).
        """
        # Latent engagement drives correlation across metrics
        # (Beta chosen to produce many low-engagement, few high-engagement sessions)
        engagement = rng.beta(2.0, 5.0, size=n)  # in [0,1]

        # For arm B, only *affected metrics* use a shifted engagement signal.
        # This keeps some coordinates close to null while others move.
        if arm == "B" and engagement_shift != 0:
            engagement_eff = np.clip(engagement + engagement_shift, 0.0, 1.0)
        else:
            engagement_eff = engagement

        # Helper: pick engagement per metric (shifted only for affected metrics)
        def e_for_metric(metric_index: int) -> np.ndarray:
            if metric_index in effect_idx:
                return engagement_eff
            return engagement

        # --- Raw metrics on natural-ish scales ---
        # time_sec: lognormal noise around an engagement-dependent mean
        e_time = e_for_metric(name_to_idx["time"])
        base_time = 20.0 + 220.0 * e_time  # seconds
        time_noise = rng.lognormal(mean=0.0, sigma=0.35, size=n)  # mean ~ 1.06
        time_sec = base_time * time_noise
        time_sec = np.clip(time_sec, 0.0, CAPS["time_sec"])

        # clicks: Poisson with engagement-dependent rate
        e_clicks = e_for_metric(name_to_idx["clicks"])
        lam_clicks = 0.5 + 8.0 * e_clicks
        clicks = rng.poisson(lam=lam_clicks, size=n).astype(float)
        clicks = np.clip(clicks, 0.0, CAPS["clicks"])

        # pages: Poisson with engagement-dependent rate
        e_pages = e_for_metric(name_to_idx["pages"])
        lam_pages = 0.8 + 4.5 * e_pages
        pages = rng.poisson(lam=lam_pages, size=n).astype(float)
        pages = np.clip(pages, 0.0, CAPS["pages"])

        # scroll: Beta distribution pushed higher with engagement
        e_scroll = e_for_metric(name_to_idx["scroll"])
        a = 1.5 + 8.0 * e_scroll
        b = 2.5 + 3.0 * (1.0 - e_scroll)
        scroll = rng.beta(a, b)

        # add_to_cart: Bernoulli with engagement-dependent prob
        e_atc = e_for_metric(name_to_idx["add_to_cart"])
        p_atc = np.clip(0.03 + 0.40 * e_atc, 0.0, 1.0)
        add_to_cart = rng.binomial(1, p_atc, size=n).astype(float)

        # purchase: depends on engagement and add_to_cart
        e_buy = e_for_metric(name_to_idx["purchase"])
        p_buy = np.clip(0.005 + 0.22 * e_buy * add_to_cart, 0.0, 1.0)
        purchase = rng.binomial(1, p_buy, size=n).astype(float)

        # bounce: higher when engagement is low; we store non_bounce = 1 - bounce
        # Keep bounce unaffected unless you include "non_bounce" in effect_metrics.
        e_bounce = e_for_metric(name_to_idx["non_bounce"])
        p_bounce = np.clip(0.60 - 0.45 * e_bounce, 0.0, 1.0)
        bounce = rng.binomial(1, p_bounce, size=n).astype(float)
        non_bounce = 1.0 - bounce

        # return_7d: engagement-dependent probability
        e_ret = e_for_metric(name_to_idx["return_7d"])
        p_ret = np.clip(0.08 + 0.30 * e_ret, 0.0, 1.0)
        return_7d = rng.binomial(1, p_ret, size=n).astype(float)

        # --- Scale to [0,1] ---
        time = time_sec / CAPS["time_sec"]
        clicks_s = clicks / CAPS["clicks"]
        pages_s = pages / CAPS["pages"]

        X_base = np.column_stack([
            time,
            clicks_s,
            pages_s,
            scroll,
            add_to_cart,
            purchase,
            non_bounce,
            return_7d,
        ]).astype(np.float64)

        # Noise dimensions: i.i.d. Uniform(0,1), no treatment effect by construction
        if noise_dims > 0:
            X_noise = rng.uniform(0.0, 1.0, size=(n, noise_dims))
            X = np.concatenate([X_base, X_noise], axis=1)
        else:
            X = X_base

        # Safety: enforce bounds numerically
        X = np.clip(X, 0.0, 1.0)
        return X

    # ---------- Batched pairing ----------
    Z = np.zeros((T, D), dtype=np.float64)
    mean_A = np.zeros((T, D), dtype=np.float64)
    mean_B = np.zeros((T, D), dtype=np.float64)
    batch_sizes = np.zeros((T, 2), dtype=int)  # (nA, nB)

    for t in range(T):
        if variable_batch_sizes:
            # +/- 15% jitter, but never below min_batch
            jitterA = int(rng.integers(-int(0.15 * n_per_arm), int(0.15 * n_per_arm) + 1))
            jitterB = int(rng.integers(-int(0.15 * n_per_arm), int(0.15 * n_per_arm) + 1))
            nA = max(min_batch, n_per_arm + jitterA)
            nB = max(min_batch, n_per_arm + jitterB)
        else:
            nA = nB = n_per_arm

        XA = _simulate_sessions(nA, arm="A") # returns (nA, D)
        XB = _simulate_sessions(nB, arm="B")

        mA = XA.mean(axis=0)
        mB = XB.mean(axis=0)

        delta = mB - mA  # in [-1,1]^D
        z = 0.5 * (delta + 1.0)  # in [0,1]^D

        mean_A[t] = mA
        mean_B[t] = mB
        Z[t] = np.clip(z, 0.0, 1.0)
        batch_sizes[t] = (nA, nB)

    meta = {
        "metric_names": metric_names,
        "base_metric_names": base_metric_names,
        "noise_dims": noise_dims,
        "effect_metrics": list(effect_metrics),
        "effect_indices_base": effect_idx,
        "engagement_shift": engagement_shift,
        "caps": CAPS,
        "batch_sizes": batch_sizes,  # shape (T,2)
        "mean_A": mean_A,  # shape (T,D)
        "mean_B": mean_B,  # shape (T,D)
        "delta": mean_B - mean_A,  # shape (T,D)
        "Z_definition": "Z_t = 0.5 * ((mean_B - mean_A) + 1), null at 0.5",
    }
    return Z.astype(np.float64), meta


from typing import Optional, Sequence


def plot_metric_histograms(
        Z: np.ndarray,
        meta: dict,
        metrics: Optional[Sequence[str]] = None,
        bins: int = 30,
):
    """
    One histogram figure per metric for Z[:, d] across batches.
    Null reference is 0.5.
    """
    names = meta["metric_names"]
    name_to_idx = {n: i for i, n in enumerate(names)}

    if metrics is None:
        # default: show a few representative ones
        metrics = [n for n in ["time", "clicks", "scroll", "purchase", "noise_1"] if n in name_to_idx]

    for m in metrics:
        d = name_to_idx[m]
        plt.figure()
        plt.hist(Z[:, d], bins=bins, density=True)
        plt.axvline(0.5, linestyle="--")  # null reference
        plt.xlim(0, 1)
        plt.title(f"Histogram of Z_t for metric '{m}' (across batches)")
        plt.xlabel("Z_t value")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()


def plot_metric_timeseries(
        Z: np.ndarray,
        meta: dict,
        metrics: Optional[Sequence[str]] = None,
        max_T: Optional[int] = None,
):
    """
    One time-series figure per metric for Z[t, d] over batches t.
    Null reference is 0.5.
    """
    names = meta["metric_names"]
    name_to_idx = {n: i for i, n in enumerate(names)}

    if metrics is None:
        metrics = [n for n in ["time", "clicks", "scroll", "purchase", "noise_1"] if n in name_to_idx]

    T = Z.shape[0] if max_T is None else min(Z.shape[0], max_T)
    t = np.arange(T)

    for m in metrics:
        d = name_to_idx[m]
        plt.figure()
        plt.plot(t, Z[:T, d])
        plt.axhline(0.5, linestyle="--")  # null reference
        plt.title(f"Z_t time-series for metric '{m}'")
        plt.xlabel("Batch t")
        plt.ylabel("Z_t value")
        plt.tight_layout()
        plt.show()


def _build_ab_hparams(
        D: int,
        N: int,
        alpha: float = 0.05,
        batch_size: int = 1,
        verbose_progress: bool = False,
) -> dict:
    return {
        "alpha": alpha,
        "batch_size": batch_size,
        "data_type": f"uniform_{D}d",
        "N": N,
        "cut_to_simplex": False,
        "calculate_volume": True,
        "plot_grid_2d": False,
        "verbose_progress": verbose_progress,
        "cache_hedged": "EXACT",
        "grid_resolution": 100,
        "parabola_power": None,
        "parabola_adaptive": True,
        "parabola_grid_size": 2001,
        "parabola_C_touchpoint": "automatic",
        "extend_f_domain_for_parabola_fit": True,
        "conf_sphere_optimistic_rescale": False,
        "perform_volume_sanity_checks": False,
    }


def _geometric_mean_radius(volume: float, D: int) -> float:
    if volume <= 0.0:
        return 0.0
    return float(volume) ** (1.0 / float(D))


def _ensure_2d_bounds(lowers: np.ndarray, uppers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if lowers.ndim == 1:
        return lowers[:, None], uppers[:, None]
    return lowers, uppers


def _find_significant_dims(lowers_t: np.ndarray, uppers_t: np.ndarray, threshold: float = 0.5):
    above = np.where(lowers_t > threshold)[0]
    below = np.where(uppers_t < threshold)[0]
    return above, below


def _evaluate_ab_stopping(
        lowers: np.ndarray,
        uppers: np.ndarray,
        volumes: np.ndarray,
        r_tol: float,
        threshold: float = 0.5,
) -> dict:
    lowers, uppers = _ensure_2d_bounds(lowers, uppers)
    # lowers, e.g. (250, 10)
    T, D = lowers.shape
    if T > 1000:
        assert np.all(lowers[-1] > 0) and np.all(uppers[-1] < 1)

    for t in range(T):
        above, below = _find_significant_dims(lowers[t], uppers[t], threshold=threshold)
        r_t = _geometric_mean_radius(float(volumes[t]), D)

        if above.size > 0 or below.size > 0:
            return {
                "stop_t": t + 1,
                "decision": "significant",
                "above": above,
                "below": below,
                "r_t": r_t,
            }

        if r_t <= r_tol:
            return {
                "stop_t": t + 1,
                "decision": "no_significance",
                "above": np.array([], dtype=int),
                "below": np.array([], dtype=int),
                "r_t": r_t,
            }

    return {
        "stop_t": T,
        "decision": "no_decision",
        "above": np.array([], dtype=int),
        "below": np.array([], dtype=int),
        "r_t": _geometric_mean_radius(float(volumes[-1]), D),
    }


def _evaluate_ab_timepoint(
        lower_t: np.ndarray,
        upper_t: np.ndarray,
        volume_t: float,
        D: int,
        r_tol: float,
        threshold: float = 0.5,
) -> dict:
    """Evaluate the Table 3 stopping rule at one time point."""
    above, below = _find_significant_dims(lower_t, upper_t, threshold=threshold)
    r_t = _geometric_mean_radius(float(volume_t), D)
    if above.size > 0 or below.size > 0:
        decision = "significant"
    elif r_t <= r_tol:
        decision = "no_significance"
    else:
        decision = "continue"
    return {
        "decision": decision,
        "above": above,
        "below": below,
        "r_t": r_t,
    }


def run_ab_method_until_decision(
        Z: np.ndarray,
        method_type: str,
        r_tol: float,
        alpha: float = 0.05,
        threshold: float = 0.5,
        verbose_progress: bool = False,
        progress_every: int = 100,
        progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run one A/B method only until the Table 3 decision rule fires.

    This is the streaming/early-stopping counterpart of ``run_ab_experiment``.
    The three baseline implementations are updated one observation at a time.
    The hedged-capital implementations reuse their exact one-dimensional
    caches and evaluate only the current time point, instead of materializing
    the full confidence sequence.

    For the hedged methods, ``stop_t`` deliberately follows the indexing used
    by their existing ``run()`` implementations and hence by Table 3: output
    row ``t`` is labeled ``t + 1``. Keeping this convention makes the
    ``n_per_arm=400`` ablation directly reproducible against the paper.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] == 0:
        raise ValueError("Z must be a non-empty (T, D) array.")
    if np.any(Z < 0.0) or np.any(Z > 1.0):
        raise ValueError("All A/B stream observations must lie in [0, 1]^D.")

    T, D = Z.shape
    h = _build_ab_hparams(
        D=D,
        N=T,
        alpha=alpha,
        batch_size=1,
        verbose_progress=verbose_progress,
    )
    h["interval_types"] = [method_type]
    method, method_name, method_color = IntervalFactory.create_instance(method_type, h, Z)

    update_methods = {"CONF_SPHERE", "BANACH_SPHERE", "NORMALIZED_ELLIP"}
    is_bonferroni = isinstance(method, HedgedCapitalCIMultiDimensionalBonferroni)
    is_convex_sum = isinstance(method, HedgedCapitalConvexSum)
    if method_type not in update_methods and not is_bonferroni and not is_convex_sum:
        raise ValueError(f"Early stopping is not implemented for method {method_type!r}.")

    kd_calculators = None
    convex_target = None
    if is_bonferroni or is_convex_sum:
        kd_calculators = HedgedFactory.build_and_cache_Khedgeds(h, Z)
    if is_convex_sum:
        convex_target = 1.0 / alpha

    last_lower = np.zeros(D, dtype=float)
    last_upper = np.ones(D, dtype=float)
    last_volume = 1.0
    last_info = {
        "decision": "continue",
        "above": np.array([], dtype=int),
        "below": np.array([], dtype=int),
        "r_t": 1.0,
    }

    for t_zero in range(T):
        if method_type in update_methods:
            t_one = t_zero + 1
            method.update(t_one, Z[t_zero:t_one])
            lower_t = np.asarray(method.lowers[t_zero], dtype=float)
            upper_t = np.asarray(method.uppers[t_zero], dtype=float)
            volume_t = float(method.volumes[t_zero])
        elif is_bonferroni:
            target_per_dim = D / alpha
            lower_t = np.empty(D, dtype=float)
            upper_t = np.empty(D, dtype=float)
            for d, calculator in enumerate(kd_calculators):
                lower_t[d], upper_t[d] = calculator.find_confidence_interval_1d(
                    t_zero, target_per_dim
                )
            if h["cut_to_simplex"] and D > 1:
                from adaptive_sample.helper.hypercube_tools import HypercubeTools
                lower_t, upper_t = HypercubeTools.hypercube_and_simplex_intersection(
                    lower_t, upper_t, D
                )
            volume_t = VolumeCalculator.hypercube_volume(h, lower_t, upper_t)
        else:
            lower_t, upper_t, extra = method._bounds_at_t(
                t_zero, kd_calculators, convex_target
            )
            lower_t = np.asarray(lower_t, dtype=float)
            upper_t = np.asarray(upper_t, dtype=float)
            volume_t, _ = method.compute_volume(
                t_zero, kd_calculators, lower_t, upper_t, extra
            )
            volume_t = float(volume_t)

        info = _evaluate_ab_timepoint(
            lower_t=lower_t,
            upper_t=upper_t,
            volume_t=volume_t,
            D=D,
            r_tol=r_tol,
            threshold=threshold,
        )
        last_lower = lower_t
        last_upper = upper_t
        last_volume = volume_t
        last_info = info

        stop_t = t_zero + 1
        should_report_progress = (
            progress_callback is not None
            and (
                info["decision"] != "continue"
                or stop_t == T
                or (progress_every > 0 and stop_t % progress_every == 0)
            )
        )
        if should_report_progress:
            progress_callback(
                {
                    "method_type": method_type,
                    "method_name": method_name,
                    "processed_t": stop_t,
                    "max_t": T,
                    "decision": info["decision"],
                    "r_t": info["r_t"],
                }
            )

        if info["decision"] != "continue":
            break
    else:
        stop_t = T

    final_decision = last_info["decision"]
    if final_decision == "continue":
        final_decision = "no_decision"

    return {
        "method_type": method_type,
        "method_name": method_name,
        "method_color": method_color,
        "stop_t": int(stop_t),
        "decision": final_decision,
        "above": last_info["above"],
        "below": last_info["below"],
        "r_t": float(last_info["r_t"]),
        "final_volume": float(last_volume),
        "lower": last_lower,
        "upper": last_upper,
    }


def run_ab_experiment(
        Z: np.ndarray,
        meta: dict,
        method_types: Sequence[str],
        r_tol: float,
        alpha: float = 0.05,
        threshold: float = 0.5,
        verbose_progress: bool = False,
        short_T: int = 1000,
        long_T: Optional[int] = None,
        long_T_methods: Optional[Sequence[str]] = None,
) -> List[dict]:
    D = Z.shape[1]
    results = []

    for method_type in method_types:
        use_long = long_T_methods is not None and method_type in long_T_methods
        if use_long:
            target_T = long_T if long_T is not None else Z.shape[0]
        else:
            target_T = min(short_T, Z.shape[0])

        Z_used = Z[:target_T]
        N = Z_used.shape[0]

        h = _build_ab_hparams(
            D=D,
            N=N,
            alpha=alpha,
            batch_size=1,
            verbose_progress=verbose_progress,
        )
        h["interval_types"] = [method_type]

        method, method_name, method_color = IntervalFactory.create_instance(method_type, h, Z_used)
        ts, means, lowers, uppers, volumes, volume_stds = method.run()

        stop_info = _evaluate_ab_stopping(
            lowers=lowers,
            uppers=uppers,
            volumes=volumes,
            r_tol=r_tol,
            threshold=threshold,
        )

        results.append(
            {
                "method_type": method_type,
                "method_name": method_name,
                "method_color": method_color,
                "stop_t": stop_info["stop_t"],
                "decision": stop_info["decision"],
                "above": stop_info["above"],
                "below": stop_info["below"],
                "r_t": stop_info["r_t"],
                "final_volume": float(volumes[stop_info["stop_t"] - 1]),
            }
        )

    return results


def _format_dim_list(indices: np.ndarray, names: Sequence[str]) -> str:
    if indices.size == 0:
        return "-"
    return ", ".join([names[i] if i < len(names) else f"dim_{i}" for i in indices])


def report_ab_results(
        title: str,
        results: List[dict],
        metric_names: Sequence[str],
        r_tol: float,
) -> None:
    print("=" * 80)
    print(title)
    print(f"Geometric-mean radius tolerance: r_tol={r_tol:.4g}")
    print("-" * 80)
    name_width = max(len(r["method_name"]) for r in results)
    header = f"{'method':<{name_width}} | stop_t | decision        | r_t     | above | below"
    print(header)
    print("-" * len(header))

    for r in results:
        above_str = _format_dim_list(r["above"], metric_names)
        below_str = _format_dim_list(r["below"], metric_names)
        print(
            f"{r['method_name']:<{name_width}} | "
            f"{r['stop_t']:>6d} | "
            f"{r['decision']:<14} | "
            f"{r['r_t']:<7.4g} | "
            f"{above_str:<5} | "
            f"{below_str}"
        )

    print("=" * 80)


def run_ab_experiment_suite(
        Z: np.ndarray,
        meta: dict,
        method_types: Sequence[str],
        r_tol_strict: float = 0.02,
        alpha: float = 0.05,
        threshold: float = 0.5,
        verbose_progress: bool = False,
        short_T: int = 1000,
        long_T: Optional[int] = None,
        long_T_methods: Optional[Sequence[str]] = None,
        Z_no_effect: Optional[np.ndarray] = None,
        meta_no_effect: Optional[dict] = None,
) -> Dict[str, List[dict]]:

    results_effect = run_ab_experiment(
        Z=Z,
        meta=meta,
        method_types=method_types,
        r_tol=r_tol_strict,
        alpha=alpha,
        threshold=threshold,
        verbose_progress=verbose_progress,
        short_T=short_T,
        long_T=long_T,
        long_T_methods=long_T_methods,
    )

    report_ab_results(
        title="A/B experiment: strict tolerance (with effect)",
        results=results_effect,
        metric_names=meta["metric_names"],
        r_tol=r_tol_strict,
    )

    if Z_no_effect is not None:
        meta_alt = meta_no_effect if meta_no_effect is not None else meta
        results_no_effect = run_ab_experiment(
            Z=Z_no_effect,
            meta=meta_alt,
            method_types=method_types,
            r_tol=r_tol_strict,
            alpha=alpha,
            threshold=threshold,
            verbose_progress=verbose_progress,
            short_T=short_T,
            long_T=long_T,
            long_T_methods=long_T_methods,
        )
        report_ab_results(
            title="A/B experiment: strict tolerance (no effect)",
            results=results_no_effect,
            metric_names=meta_alt["metric_names"],
            r_tol=r_tol_strict,
        )
    else:
        results_no_effect = []

    return {
        "effect": results_effect,
        "no_effect": results_no_effect,
    }


# --- Example usage ---
if __name__ == "__main__":

    # 10_000 | 1_000 | 20 | 10
    T_sphere = 10_000
    T_other = 500

    Z, meta = get_batched_ab_stream(
        T=T_sphere, # 250 | 10_000
        n_per_arm=400, # batch size
        seed=1,
        noise_dims=2,
        effect_metrics=("time", "clicks", "scroll", "add_to_cart"),
        engagement_shift=0.06,
        variable_batch_sizes=True,
    )
    Z_null, meta_null = get_batched_ab_stream(
        T=T_sphere, # 250 | 10_000
        n_per_arm=400,
        seed=2,
        noise_dims=2,
        effect_metrics=(),
        engagement_shift=0.0,
        variable_batch_sizes=True,
    )
    run_ab_experiments = True
    run_plots = True

    if run_plots:
        print("Z shape:", Z.shape)
        print("Metrics:", meta["metric_names"])
        print("First row (Z_1):", Z[0])
        print("Mean(Z) over time (should be >0.5 on affected metrics):")
        print({name: float(Z.mean(axis=0)[i]) for i, name in enumerate(meta["metric_names"])})

        plot_metric_histograms(Z, meta)
        plot_metric_timeseries(Z, meta, max_T=100)

    if run_ab_experiments:
        method_types = [
            "CONF_SPHERE",
            "BANACH_SPHERE",
            "HEDGE_nd_BBX",
            # "HEDGE_nd_ELLIP",
            "HEDGE_nd_ELLIP_SAFE",
            # "HEDGE_nd_ELLIP_BBX",
            "HEDGE_nd_ELLIP_BBX_SAFE",
            "HEDGE_nd_BONF",
            # "HORSE_RACE_BOUNDED"
            # "HEDGE_nd_GRID",
            "NORMALIZED_ELLIP"

            # "CONF_SPHERE",
            # "BANACH_SPHERE",
            # "HEDGE_nd_BBX",
            # "HEDGE_nd_ELLIP",
            # "HEDGE_nd_BONF",
            # "HEDGE_nd_ELLIP_BBX",
        ]
        run_ab_experiment_suite(
            Z=Z,
            meta=meta,
            method_types=method_types,
            r_tol_strict=0.02,
            alpha=0.05,
            threshold=0.5,
            verbose_progress=True,
            short_T=T_other,
            long_T=T_sphere,
            long_T_methods=("CONF_SPHERE", "BANACH_SPHERE", "NORMALIZED_ELLIP"),
            Z_no_effect=Z_null,
            meta_no_effect=meta_null,
        )

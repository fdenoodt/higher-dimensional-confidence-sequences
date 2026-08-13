from typing import Dict, Optional, Sequence, Tuple
import numpy as np

all_metrics = [
    "time", "clicks", "pages", "scroll",
    "add_to_cart", "purchase", "non_bounce", "return_7d",
    "noise_1", "noise_2",
]

def get_batched_ab_stream(
    T: int = 200,
    n_per_arm: int = 500,
    seed: int = 0,
    noise_dims: int = 2,
    effect_metrics: Sequence[str] = ("time", "clicks", "scroll", "add_to_cart"),
    engagement_shift: float = 0.06,
    variable_batch_sizes: bool = True,
    min_batch: int = 100,
    # NEW: store one representative batch's raw session-level samples for plotting
    store_raw_batch: Optional[int] = 0,
    # NEW: estimate "true" (population) means via large Monte Carlo sample
    store_population_means: bool = True,
    population_n: int = 200_000,
) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(seed)

    CAPS = {"time_sec": 300.0, "clicks": 20.0, "pages": 10.0}

    base_metric_names = [
        "time", "clicks", "pages", "scroll",
        "add_to_cart", "purchase", "non_bounce", "return_7d",
    ]
    noise_metric_names = [f"noise_{j + 1}" for j in range(noise_dims)]
    metric_names = base_metric_names + noise_metric_names

    name_to_idx = {n: i for i, n in enumerate(base_metric_names)}
    effect_idx = [name_to_idx[m] for m in effect_metrics if m in name_to_idx]

    D_base = len(base_metric_names)
    D = D_base + noise_dims

    def _simulate_sessions(n: int, arm: str, return_latent: bool = False):
        """
        Returns:
            X: (n, D) in [0,1]
            If return_latent: also returns (E, E_eff), where E_eff is the shifted/clipped engagement
            used for affected metrics (equals E for arm A, or when engagement_shift=0).
        """
        E = rng.beta(2.0, 5.0, size=n)  # latent engagement in [0,1]

        if arm == "B" and engagement_shift != 0.0:
            E_eff = np.clip(E + engagement_shift, 0.0, 1.0)
        else:
            E_eff = E

        def e_for_metric(metric_index: int) -> np.ndarray:
            return E_eff if metric_index in effect_idx else E

        # time (scaled)
        e_time = e_for_metric(name_to_idx["time"])
        base_time = 20.0 + 220.0 * e_time
        time_noise = rng.lognormal(mean=0.0, sigma=0.35, size=n)
        time_sec = np.clip(base_time * time_noise, 0.0, CAPS["time_sec"])

        # clicks (scaled)
        e_clicks = e_for_metric(name_to_idx["clicks"])
        lam_clicks = 0.5 + 8.0 * e_clicks
        clicks = np.clip(rng.poisson(lam=lam_clicks, size=n).astype(float), 0.0, CAPS["clicks"])

        # pages (scaled)
        e_pages = e_for_metric(name_to_idx["pages"])
        lam_pages = 0.8 + 4.5 * e_pages
        pages = np.clip(rng.poisson(lam=lam_pages, size=n).astype(float), 0.0, CAPS["pages"])

        # scroll in [0,1]
        e_scroll = e_for_metric(name_to_idx["scroll"])
        a = 1.5 + 8.0 * e_scroll
        b = 2.5 + 3.0 * (1.0 - e_scroll)
        scroll = rng.beta(a, b)

        # add_to_cart
        e_atc = e_for_metric(name_to_idx["add_to_cart"])
        p_atc = np.clip(0.03 + 0.40 * e_atc, 0.0, 1.0)
        add_to_cart = rng.binomial(1, p_atc, size=n).astype(float)

        # purchase
        e_buy = e_for_metric(name_to_idx["purchase"])
        p_buy = np.clip(0.005 + 0.22 * e_buy * add_to_cart, 0.0, 1.0)
        purchase = rng.binomial(1, p_buy, size=n).astype(float)

        # non_bounce
        e_bounce = e_for_metric(name_to_idx["non_bounce"])
        p_bounce = np.clip(0.60 - 0.45 * e_bounce, 0.0, 1.0)
        bounce = rng.binomial(1, p_bounce, size=n).astype(float)
        non_bounce = 1.0 - bounce

        # return_7d
        e_ret = e_for_metric(name_to_idx["return_7d"])
        p_ret = np.clip(0.08 + 0.30 * e_ret, 0.0, 1.0)
        return_7d = rng.binomial(1, p_ret, size=n).astype(float)

        time = time_sec / CAPS["time_sec"]
        clicks_s = clicks / CAPS["clicks"]
        pages_s = pages / CAPS["pages"]

        X_base = np.column_stack([
            time, clicks_s, pages_s, scroll,
            add_to_cart, purchase, non_bounce, return_7d
        ]).astype(np.float64)

        if noise_dims > 0:
            X_noise = rng.uniform(0.0, 1.0, size=(n, noise_dims))
            X = np.concatenate([X_base, X_noise], axis=1)
        else:
            X = X_base

        X = np.clip(X, 0.0, 1.0)
        if return_latent:
            return X, E, E_eff
        return X

    # Batched pairing stream
    Z = np.zeros((T, D), dtype=np.float64)
    mean_A = np.zeros((T, D), dtype=np.float64)
    mean_B = np.zeros((T, D), dtype=np.float64)
    batch_sizes = np.zeros((T, 2), dtype=int)

    # optional raw batch storage
    raw = {}

    for t in range(T):
        if variable_batch_sizes:
            jitterA = int(rng.integers(-int(0.15 * n_per_arm), int(0.15 * n_per_arm) + 1))
            jitterB = int(rng.integers(-int(0.15 * n_per_arm), int(0.15 * n_per_arm) + 1))
            nA = max(min_batch, n_per_arm + jitterA)
            nB = max(min_batch, n_per_arm + jitterB)
        else:
            nA = nB = n_per_arm

        if store_raw_batch is not None and t == store_raw_batch:
            XA, EA, EA_eff = _simulate_sessions(nA, arm="A", return_latent=True)
            XB, EB, EB_eff = _simulate_sessions(nB, arm="B", return_latent=True)
            raw = {
                "raw_batch_index": t,
                "raw_XA": XA, "raw_XB": XB,
                "raw_EA": EA, "raw_EA_eff": EA_eff,
                "raw_EB": EB, "raw_EB_eff": EB_eff,
                "raw_nA": nA, "raw_nB": nB,
            }
        else:
            XA = _simulate_sessions(nA, arm="A")
            XB = _simulate_sessions(nB, arm="B")

        mA = XA.mean(axis=0)
        mB = XB.mean(axis=0)

        delta = mB - mA
        z = 0.5 * (delta + 1.0)

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
        "batch_sizes": batch_sizes,
        "mean_A": mean_A,
        "mean_B": mean_B,
        "delta": mean_B - mean_A,
        "Z_definition": "Z_t = 0.5 * ((mean_B - mean_A) + 1), null at 0.5",
        **raw,  # include raw batch data if stored
    }

    if store_population_means:
        XA_pop, EA_pop, EAeff_pop = _simulate_sessions(population_n, "A", return_latent=True)
        XB_pop, EB_pop, EBeff_pop = _simulate_sessions(population_n, "B", return_latent=True)
        meta.update({
            "population_n": population_n,
            "population_mean_A": XA_pop.mean(axis=0),
            "population_mean_B": XB_pop.mean(axis=0),
            "population_delta": XB_pop.mean(axis=0) - XA_pop.mean(axis=0),
            "population_mean_EA": float(EA_pop.mean()),
            "population_mean_EB": float(EB_pop.mean()),
            "population_mean_EB_eff": float(EBeff_pop.mean()),
        })

    return Z.astype(np.float64), meta




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Optional, Sequence

def plot_single_batch_session_distributions(
    meta: dict,
    metrics: Optional[Sequence[str]] = None,
    bins: int = 40,
    show_population_means: bool = True,
    include_engagement: bool = True,
    figsize=(12, 7),
):
    """
    Visualize session-level distributions *within one stored batch* (A vs B overlay).
    Requires get_batched_ab_stream(..., store_raw_batch=some_index).
    """

    required = ["raw_XA", "raw_XB"]
    for k in required:
        if k not in meta:
            raise ValueError(
                "No raw batch stored in meta. Call get_batched_ab_stream(..., store_raw_batch=0) first."
            )

    XA = meta["raw_XA"]
    XB = meta["raw_XB"]
    names = meta["metric_names"]
    name_to_idx = {n: i for i, n in enumerate(names)}
    effect_metrics = set(meta.get("effect_metrics", []))
    engagement_has_effect = meta.get("engagement_shift", 0.0) != 0.0

    if metrics is None:
        # A compact, informative default for a paper figure
        metrics = ["time", "clicks", "scroll", "add_to_cart", "purchase", "noise_1"]
        metrics = [m for m in metrics if m in name_to_idx]

    # layout: engagement + up to 5 metrics (2x3), or without engagement (2x3)
    panels_effect = []
    panels_no_effect = []

    if include_engagement:
        if engagement_has_effect:
            panels_effect.append(("engagement", None))
        else:
            panels_no_effect.append(("engagement", None))

    metrics_effect = [m for m in metrics if m in effect_metrics]
    metrics_no_effect = [m for m in metrics if m not in effect_metrics]
    for m in metrics_effect:
        panels_effect.append((m, name_to_idx[m]))
    for m in metrics_no_effect:
        panels_no_effect.append((m, name_to_idx[m]))

    panels = panels_effect + panels_no_effect

    n_panels = len(panels)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    # styling: muted, distinct colors
    palette = {
        "A": "#2F6CB3",          # muted blue
        "B": "#D9842B",          # muted orange
        "B_shifted": "#5A8F3A",  # muted green
    }

    # helper: plot overlay hist for continuous values in [0,1]
    def _plot_overlay_hist(ax, a, b, title, meanA=None, meanB=None, popA=None, popB=None):
        ax.hist(a, bins=bins, range=(0,1), density=True, alpha=0.40, label="A", color=palette["A"])
        ax.hist(b, bins=bins, range=(0,1), density=True, alpha=0.40, label="B", color=palette["B"])
        ax.set_xlim(0,1)
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        # population means (solid)
        if popA is not None:
            ax.axvline(popA, linestyle="-", color=palette["A"], linewidth=1.2)
        if popB is not None:
            ax.axvline(popB, linestyle="-", color=palette["B"], linewidth=1.2)

    def _title_with_effect(label: str, has_effect: bool) -> str:
        suffix = " (effect)" if has_effect else " (no effect)"
        return f"{label}{suffix}"

    # helper: plot binary metric as bars
    def _plot_binary(ax, a, b, title, popA=None, popB=None):
        pA = float(np.mean(a))
        pB = float(np.mean(b))
        ax.bar([0,1], [pA, pB], color=[palette["A"], palette["B"]])
        ax.set_xticks([0,1], ["A", "B"])
        ax.set_ylim(0,1)
        ax.set_title(title)
        ax.set_ylabel("P(value=1)")
        # optional population mean markers
        if popA is not None:
            ax.axhline(popA, linestyle="-", color=palette["A"], linewidth=1.2)
        if popB is not None:
            ax.axhline(popB, linestyle="-", color=palette["B"], linewidth=1.2)

    popA_vec = meta.get("population_mean_A", None) if show_population_means else None
    popB_vec = meta.get("population_mean_B", None) if show_population_means else None

    for i, (label, idx) in enumerate(panels):
        ax = axes[i]

        if label == "engagement":
            EA = meta.get("raw_EA", None)
            EB = meta.get("raw_EB", None)
            EB_eff = meta.get("raw_EB_eff", None)

            if EA is None or EB is None or EB_eff is None:
                ax.text(0.5, 0.5, "Engagement not stored", ha="center", va="center")
                ax.axis("off")
                continue

            # plot analytic PDFs for engagement
            try:
                from scipy.stats import beta as beta_dist
            except Exception:
                ax.text(0.5, 0.5, "scipy not available", ha="center", va="center")
                ax.axis("off")
                continue

            a_param, b_param = 2.0, 5.0
            x = np.linspace(0.0, 1.0, 400)
            pdf_base = beta_dist.pdf(x, a_param, b_param)

            shift = float(meta.get("engagement_shift", 0.0))
            pdf_shifted = np.zeros_like(x)
            mass_one = 0.0
            mass_zero = 0.0
            if shift >= 0.0:
                mask = x >= shift
                pdf_shifted[mask] = beta_dist.pdf(x[mask] - shift, a_param, b_param)
                if shift > 0.0:
                    mass_one = 1.0 - beta_dist.cdf(1.0 - shift, a_param, b_param)
            else:
                mask = x <= 1.0 + shift
                pdf_shifted[mask] = beta_dist.pdf(x[mask] - shift, a_param, b_param)
                mass_zero = beta_dist.cdf(-shift, a_param, b_param)

            ax.plot(x, pdf_base, color=palette["A"], linewidth=1.6)
            ax.plot(x, pdf_shifted, color=palette["B_shifted"], linewidth=1.6)

            ax.set_xlim(0,1)
            ax.set_title(_title_with_effect("latent engagement", engagement_has_effect))
            ax.set_xlabel("value")
            ax.set_ylabel("density")
            if show_population_means and "population_mean_EA" in meta and "population_mean_EB_eff" in meta:
                ax.axvline(meta["population_mean_EA"], linestyle="-", color=palette["A"], linewidth=1.2)
                ax.axvline(meta["population_mean_EB_eff"], linestyle="-", color=palette["B_shifted"], linewidth=1.2)

            y_max = max(float(np.max(pdf_base)), float(np.max(pdf_shifted)))
            if mass_one > 0.0:
                ax.vlines(1.0, 0.0, 0.12 * y_max, color=palette["B_shifted"], linewidth=2.0)
                #ax.text(0.98, 0.14 * y_max, f"mass@1={mass_one:.3f}", ha="right", va="bottom", fontsize=8)
            if mass_zero > 0.0:
                ax.vlines(0.0, 0.0, 0.12 * y_max, color=palette["B_shifted"], linewidth=2.0)
                #ax.text(0.02, 0.14 * y_max, f"mass@0={mass_zero:.3f}", ha="left", va="bottom", fontsize=8)
            continue

        # metric panel
        a = XA[:, idx]
        b = XB[:, idx]
        meanA = float(np.mean(a))
        meanB = float(np.mean(b))
        popA = float(popA_vec[idx]) if popA_vec is not None else None
        popB = float(popB_vec[idx]) if popB_vec is not None else None
        has_effect = label in effect_metrics

        # detect binary metrics (robustly)
        is_binary = np.all(np.isin(np.unique(a), [0.0, 1.0])) and np.all(np.isin(np.unique(b), [0.0, 1.0]))

        if is_binary:
            _plot_binary(ax, a, b, title=_title_with_effect(label, has_effect), popA=popA, popB=popB)
        else:
            _plot_overlay_hist(
                ax, a, b,
                title=_title_with_effect(label, has_effect),
                meanA=meanA,
                meanB=meanB,
                popA=popA,
                popB=popB,
            )

    # hide any unused axes
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    # shared legend outside the figure (bottom)
    legend_handles = [
        Patch(facecolor=palette["A"], alpha=0.40, label="A"),
        Patch(facecolor=palette["B"], alpha=0.40, label="B"),
    ]
    if include_engagement:
        legend_handles.append(Line2D([0], [0], color=palette["A"], linewidth=1.6, label="E (original)"))
        legend_handles.append(Line2D([0], [0], color=palette["B_shifted"], linewidth=1.6, label="E (shifted)"))
    legend_handles.extend([
        Line2D([0], [0], color=palette["A"], linestyle="-", linewidth=1.2, label="A mean (pop)"),
        Line2D([0], [0], color=palette["B"], linestyle="-", linewidth=1.2, label="B mean (pop)"),
    ])
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    t = meta.get("raw_batch_index", None)
    nA = meta.get("raw_nA", None)
    nB = meta.get("raw_nB", None)
    suptitle = "Session-level distributions within one batch"
    if t is not None:
        suptitle += f" (batch t={t}"
        if nA is not None and nB is not None:
            suptitle += f", nA={nA}, nB={nB}"
        suptitle += ")"
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    plt.show()




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


# --- Example usage ---
if __name__ == "__main__":
    # 10_000 | 1_000 | 20 | 10
    T_sphere = 10_000
    T_other = 500

    Z, meta = get_batched_ab_stream(
        T=T_sphere,  # 250 | 10_000
        n_per_arm=4_00,  # batch size
        seed=1,
        noise_dims=2,
        effect_metrics=("time", "clicks", "scroll", "add_to_cart"),
        engagement_shift=0.06,
        variable_batch_sizes=True,
    )

    plot_single_batch_session_distributions(
        meta,
        metrics = all_metrics,
        # metrics=["time", "clicks", "scroll", "add_to_cart", "purchase", "noise_1"],
        bins=40,
        include_engagement=True,
        show_population_means=True,
    )

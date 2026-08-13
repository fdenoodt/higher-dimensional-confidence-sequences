import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_sample.conf_sequences.interval_factory import IntervalFactory
from adaptive_sample.conf_sequences.model_comparison.adult_model_comparison_utils import (
    artifacts_dir,
    build_batched_stream,
    conditional_proportion_source_names,
    ensure_dir,
    metric_profile_names,
    write_json,
)


DEFAULT_METHODS = [
    "CONF_SPHERE",
    "BANACH_SPHERE",
    "HEDGE_nd_BBX",
    "HEDGE_nd_BONF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multivariate confidence-sequence model comparison on stored Adult predictions."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="adult_model_comparison",
        help="Artifact subdirectory name.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="CSV of stored predictions. Defaults to the training script output.",
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default="logreg",
        help="Baseline model name in the prediction file.",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="hist_gbdt",
        help="Challenger model name in the prediction file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of examples aggregated into one CS observation.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=11,
        help="Random permutation seed for forming the test stream.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Confidence-sequence alpha level.",
    )
    parser.add_argument(
        "--r-tol",
        type=float,
        default=0.02,
        help="Geometric-mean radius tolerance for a no-significance stop.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="IntervalFactory method types to run.",
    )
    parser.add_argument(
        "--decision-rule",
        choices=["any_positive", "all_positive", "target_positive"],
        default="all_positive",
        help="Deployment rule for the challenger model.",
    )
    parser.add_argument(
        "--target-metric",
        type=str,
        default=None,
        help="Metric name to target when using --decision-rule target_positive.",
    )
    parser.add_argument(
        "--metric-profile",
        choices=metric_profile_names(),
        default="strict_performance",
        help="Which vector-valued observation definition to use.",
    )
    parser.add_argument(
        "--conditional-proportion-source",
        choices=conditional_proportion_source_names(),
        default="empirical",
        help=(
            "For subgroup_accuracy_conditional, use empirical subgroup prevalences "
            "or estimate the needed prevalences as extra CS coordinates."
        ),
    )
    parser.add_argument(
        "--target-volume",
        type=float,
        default=None,
        help="Optional absolute target volume for an early no-decision stop.",
    )
    parser.add_argument(
        "--parabola-grid-size",
        type=int,
        default=2001,
        help="Approximation grid size used by ellipsoid-based hedged methods.",
    )
    return parser.parse_args()


def build_hparams(D: int, N: int, alpha: float, parabola_grid_size: int) -> Dict:
    return {
        "alpha": alpha,
        "batch_size": 1,
        "data_type": f"uniform_{D}d",
        "N": N,
        "cut_to_simplex": False,
        "calculate_volume": True,
        "plot_grid_2d": False,
        "verbose_progress": False,
        "cache_hedged": "EXACT",
        "grid_resolution": 100,
        "parabola_power": None,
        "parabola_adaptive": True,
        "parabola_grid_size": parabola_grid_size,
        "parabola_C_touchpoint": "automatic",
        "extend_f_domain_for_parabola_fit": True,
        "conf_sphere_optimistic_rescale": False,
        "perform_volume_sanity_checks": False,
    }


def geometric_mean_radius(volume: float, D: int) -> float:
    if volume <= 0.0:
        return 0.0
    return float(volume) ** (1.0 / float(D))


def evaluate_stopping(
    lowers: np.ndarray,
    uppers: np.ndarray,
    volumes: np.ndarray,
    r_tol: float,
    decision_rule: str,
    target_volume: float,
    target_index: Optional[int] = None,
) -> Dict:
    T, D = lowers.shape
    if decision_rule == "target_positive" and target_index is None:
        raise ValueError("target_index is required when decision_rule='target_positive'.")

    for t in range(T):
        above = np.where(lowers[t] > 0.5)[0]
        below = np.where(uppers[t] < 0.5)[0]
        r_t = geometric_mean_radius(float(volumes[t]), D)
        volume_t = float(volumes[t])
        if decision_rule == "all_positive":
            if above.size == D:
                return {
                    "stop_t": t + 1,
                    "decision": "significant_all_positive",
                    "above": above,
                    "below": below,
                    "r_t": r_t,
                }
            if below.size > 0:
                return {
                    "stop_t": t + 1,
                    "decision": "failed_all_positive",
                    "above": above,
                    "below": below,
                    "r_t": r_t,
                }
        elif decision_rule == "target_positive":
            assert target_index is not None
            if lowers[t, target_index] > 0.5:
                return {
                    "stop_t": t + 1,
                    "decision": "significant_target_positive",
                    "above": above,
                    "below": below,
                    "r_t": r_t,
                    "target_index": int(target_index),
                }
            if uppers[t, target_index] < 0.5:
                return {
                    "stop_t": t + 1,
                    "decision": "failed_target_positive",
                    "above": above,
                    "below": below,
                    "r_t": r_t,
                    "target_index": int(target_index),
                }
        elif above.size > 0 or below.size > 0:
            return {
                "stop_t": t + 1,
                "decision": "significant",
                "above": above,
                "below": below,
                "r_t": r_t,
            }

        if target_volume is not None and volume_t <= target_volume:
            return {
                "stop_t": t + 1,
                "decision": "no_decision_target_volume",
                "above": np.array([], dtype=int),
                "below": np.array([], dtype=int),
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
        "r_t": geometric_mean_radius(float(volumes[-1]), D),
    }


def format_dims(indices: np.ndarray, names: List[str]) -> str:
    if len(indices) == 0:
        return "-"
    return ", ".join(names[int(i)] for i in indices)


def metric_interpretation(metric_name: str) -> str:
    if metric_name.endswith("_conditional"):
        return "conditional subgroup gain"
    if metric_name.endswith("_weighted"):
        return "weighted subgroup gain"
    if metric_name.endswith("_proportion"):
        return "subgroup prevalence"
    if metric_name.endswith("accuracy") and metric_name != "overall_accuracy":
        return "weighted subgroup gain"
    return "population mean gain"


def metric_plain_language(metric_name: str, interpretation: str) -> str:
    if metric_name == "overall_accuracy":
        return "Among all examples, model B's accuracy minus model A's accuracy"
    if metric_name == "overall_brier":
        return "Among all examples, model B's Brier gain over model A"
    if interpretation == "subgroup prevalence":
        subgroup = metric_name.replace("_proportion", "").replace("_", " ")
        return f"Population proportion for {subgroup}"
    if interpretation == "conditional subgroup gain":
        subgroup = metric_name.replace("_accuracy_conditional", "").replace("_", " ")
        return f"Among {subgroup} examples, model B's accuracy minus model A's accuracy"
    if interpretation == "weighted subgroup gain":
        subgroup = metric_name.replace("_accuracy", "").replace("_", " ")
        return f"In the full population, the prevalence-weighted accuracy gain contributed by {subgroup} examples"
    return f"Mean B-minus-A gain for {metric_name.replace('_', ' ')}"


def interval_status(lower: float, upper: float) -> str:
    if lower > 0.0:
        return "significantly positive"
    if upper < 0.0:
        return "significantly negative"
    return "inconclusive"


def render_interval_lines(
    method_name: str,
    batch_idx: int,
    per_metric_bounds: pd.DataFrame,
    *,
    prefix: str,
    mean_col: str,
    lower_col: str,
    upper_col: str,
) -> List[str]:
    lines = [f"{method_name} {prefix} intervals (batch {batch_idx}):"]
    for row in per_metric_bounds.itertuples(index=False):
        description = metric_plain_language(row.metric, row.interpretation)
        lower = getattr(row, lower_col)
        upper = getattr(row, upper_col)
        mean = getattr(row, mean_col)
        if row.interpretation == "subgroup prevalence":
            lines.append(
                f"  - {row.metric}: [{lower:.6f}, {upper:.6f}] "
                f"with mean {mean:.6f}. {description}."
            )
            continue
        status = interval_status(lower, upper)
        lines.append(
            f"  - {row.metric}: [{lower:.6f}, {upper:.6f}] "
            f"with mean {mean:.6f}. {description}. Status: {status}."
        )
    return lines


def save_metric_summary(report_dir: Path, stream_meta: Dict) -> pd.DataFrame:
    metric_names = stream_meta.get("reported_metric_names", stream_meta["metric_names"])
    weighted_means = stream_meta.get("reported_metric_means", stream_meta["weighted_metric_means"])
    metric_scores = stream_meta.get("reported_metric_scores", {})
    model_a = stream_meta["model_a"]
    model_b = stream_meta["model_b"]
    conditional = stream_meta["conditional_accuracy_gains"]
    rows = []
    for metric_name in metric_names:
        scores = metric_scores.get(metric_name, {})
        rows.append(
            {
                "metric": metric_name,
                f"{model_a}_score": scores.get(model_a, np.nan),
                f"{model_b}_score": scores.get(model_b, np.nan),
                "mean_gain_b_minus_a": weighted_means[metric_name],
                "interpretation": metric_interpretation(metric_name),
            }
        )
    for key, value in conditional.items():
        if key in metric_names:
            continue
        scores = metric_scores.get(key, {})
        rows.append(
            {
                "metric": key,
                f"{model_a}_score": scores.get(model_a, np.nan),
                f"{model_b}_score": scores.get(model_b, np.nan),
                "mean_gain_b_minus_a": value,
                "interpretation": metric_interpretation(key),
            }
        )
    metric_df = pd.DataFrame(rows)
    metric_df.to_csv(report_dir / "metric_summary.csv", index=False)
    return metric_df


def _ratio_mean(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0.0)
    return np.clip(out, -1.0, 1.0)


def _ratio_interval(
    # joint accuracy
    numerator_lower: np.ndarray,
    numerator_upper: np.ndarray,
    # subgroup proportion (p(x=subgroup))
    denominator_lower: np.ndarray,
    denominator_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full_like(numerator_lower, -1.0, dtype=np.float64)
    upper = np.full_like(numerator_upper, 1.0, dtype=np.float64)
    valid = denominator_lower > 0.0
    if not np.any(valid):
        return lower, upper

    candidates = np.stack(
        [
            numerator_lower[valid] / denominator_lower[valid],
            numerator_lower[valid] / denominator_upper[valid],
            numerator_upper[valid] / denominator_lower[valid],
            numerator_upper[valid] / denominator_upper[valid],
        ],
        axis=0,
    )
    lower[valid] = np.clip(np.min(candidates, axis=0), -1.0, 1.0)
    upper[valid] = np.clip(np.max(candidates, axis=0), -1.0, 1.0)
    return lower, upper


def report_metric_arrays(
    stream_meta: Dict,
    means_gain: np.ndarray,
    lowers_gain: np.ndarray,
    uppers_gain: np.ndarray,
) -> tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    cs_metric_names = stream_meta["metric_names"]
    report_metric_names = stream_meta.get("reported_metric_names", cs_metric_names)
    if not (
        stream_meta.get("metric_profile") == "subgroup_accuracy_conditional"
        and stream_meta.get("conditional_proportion_source") == "cs"
    ):
        indices = [cs_metric_names.index(name) for name in report_metric_names]
        return (
            report_metric_names,
            means_gain[:, indices],
            lowers_gain[:, indices],
            uppers_gain[:, indices],
        )

    raw = {name: idx for idx, name in enumerate(cs_metric_names)}
    female_p_mean = means_gain[:, raw["female_proportion"]]
    female_p_lower = lowers_gain[:, raw["female_proportion"]]
    female_p_upper = uppers_gain[:, raw["female_proportion"]]
    male_p_mean = 1.0 - female_p_mean
    male_p_lower = 1.0 - female_p_upper
    male_p_upper = 1.0 - female_p_lower

    lt40_p_mean = means_gain[:, raw["age_lt40_proportion"]]
    lt40_p_lower = lowers_gain[:, raw["age_lt40_proportion"]]
    lt40_p_upper = uppers_gain[:, raw["age_lt40_proportion"]]
    ge40_p_mean = 1.0 - lt40_p_mean
    ge40_p_lower = 1.0 - lt40_p_upper
    ge40_p_upper = 1.0 - lt40_p_lower

    derived = {
        "overall_accuracy": (
            means_gain[:, raw["overall_accuracy"]],
            lowers_gain[:, raw["overall_accuracy"]],
            uppers_gain[:, raw["overall_accuracy"]],
        ),
        "overall_brier": (
            means_gain[:, raw["overall_brier"]],
            lowers_gain[:, raw["overall_brier"]],
            uppers_gain[:, raw["overall_brier"]],
        ),
    }
    ratio_specs = {
        "female_accuracy_conditional": ("female_accuracy_weighted", female_p_mean, female_p_lower, female_p_upper),
        "male_accuracy_conditional": ("male_accuracy_weighted", male_p_mean, male_p_lower, male_p_upper),
        "age_lt40_accuracy_conditional": ("age_lt40_accuracy_weighted", lt40_p_mean, lt40_p_lower, lt40_p_upper),
        "age_ge40_accuracy_conditional": ("age_ge40_accuracy_weighted", ge40_p_mean, ge40_p_lower, ge40_p_upper),
    }
    for metric_name, (numerator_name, p_mean, p_lower, p_upper) in ratio_specs.items():
        numerator_idx = raw[numerator_name]
        # here we normalize joint accuracy by the proportion to get conditional accuracy (gain).
        lower, upper = _ratio_interval(
            lowers_gain[:, numerator_idx], # e.g. joint accuracy gain for women
            uppers_gain[:, numerator_idx],
            p_lower, # e.g. p(female)
            p_upper,
        )
        derived[metric_name] = (
            _ratio_mean(means_gain[:, numerator_idx], p_mean),
            lower,
            upper,
        )

    return (
        report_metric_names,
        np.column_stack([derived[name][0] for name in report_metric_names]),
        np.column_stack([derived[name][1] for name in report_metric_names]),
        np.column_stack([derived[name][2] for name in report_metric_names]),
    )


def append_cs_proportion_bounds(
    per_metric_bounds: pd.DataFrame,
    stream_meta: Dict,
    means_gain: np.ndarray,
    lowers_gain: np.ndarray,
    uppers_gain: np.ndarray,
    stop_idx: int,
    final_idx: int,
) -> pd.DataFrame:
    if not (
        stream_meta.get("metric_profile") == "subgroup_accuracy_conditional"
        and stream_meta.get("conditional_proportion_source") == "cs"
    ):
        return per_metric_bounds

    raw = {name: idx for idx, name in enumerate(stream_meta["metric_names"])}
    female_idx = raw["female_proportion"]
    lt40_idx = raw["age_lt40_proportion"]
    proportion_specs = [
        (
            "female_proportion",
            means_gain[:, female_idx],
            lowers_gain[:, female_idx],
            uppers_gain[:, female_idx],
        ),
        (
            "male_proportion",
            1.0 - means_gain[:, female_idx],
            1.0 - uppers_gain[:, female_idx],
            1.0 - lowers_gain[:, female_idx],
        ),
        (
            "age_lt40_proportion",
            means_gain[:, lt40_idx],
            lowers_gain[:, lt40_idx],
            uppers_gain[:, lt40_idx],
        ),
        (
            "age_ge40_proportion",
            1.0 - means_gain[:, lt40_idx],
            1.0 - uppers_gain[:, lt40_idx],
            1.0 - lowers_gain[:, lt40_idx],
        ),
    ]
    rows = []
    for metric_name, means, lowers, uppers in proportion_specs:
        rows.append(
            {
                "metric": metric_name,
                "interpretation": metric_interpretation(metric_name),
                "mean_gain_at_stop": means[stop_idx],
                "lower_gain_at_stop": lowers[stop_idx],
                "upper_gain_at_stop": uppers[stop_idx],
                "mean_gain_final": means[final_idx],
                "lower_gain_final": lowers[final_idx],
                "upper_gain_final": uppers[final_idx],
                "significant_positive_at_stop": False,
                "significant_negative_at_stop": False,
            }
        )
    return pd.concat([per_metric_bounds, pd.DataFrame(rows)], ignore_index=True)


def plot_cumulative_means(batch_df: pd.DataFrame, plot_dir: Path) -> None:
    metric_names = [c for c in batch_df.columns if c not in {"batch_index", "start_row", "end_row_exclusive", "batch_size"}]
    t = np.arange(1, len(batch_df) + 1)
    n_metrics = len(metric_names)
    ncols = min(3, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), sharex=True)
    axes = axes.ravel()
    for idx, metric_name in enumerate(metric_names):
        cummean = batch_df[metric_name].expanding().mean()
        ax = axes[idx]
        ax.plot(t, cummean, color="#2F6CB3")
        ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_title(metric_name.replace("_", " "))
        ax.set_xlabel("Batch")
        ax.set_ylabel("Cumulative mean gain")
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(plot_dir / "cumulative_metric_means.png", dpi=200)
    plt.close(fig)


def plot_volumes(volume_curves: Dict[str, np.ndarray], plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for method_name, volumes in volume_curves.items():
        ax.plot(np.arange(1, len(volumes) + 1), volumes, label=method_name)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Confidence-region volume")
    ax.set_title("Volume shrinkage over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "volume_shrinkage.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact_root = ensure_dir(artifacts_dir(args.experiment_name))
    prediction_path = args.predictions_path or artifact_root / "predictions" / "adult_test_predictions.csv"
    if not prediction_path.exists() and args.predictions_path is None:
        prediction_path = artifacts_dir("adult_model_comparison") / "predictions" / "adult_test_predictions.csv"
    report_dir = ensure_dir(artifact_root / "reports")
    cs_dir = ensure_dir(artifact_root / "cs_runs")
    plot_dir = ensure_dir(artifact_root / "plots")

    prediction_df = pd.read_csv(prediction_path)
    Z, stream_bundle = build_batched_stream(
        prediction_df=prediction_df,
        model_a=args.model_a,
        model_b=args.model_b,
        batch_size=args.batch_size,
        shuffle_seed=args.shuffle_seed,
        metric_profile=args.metric_profile,
        conditional_proportion_source=args.conditional_proportion_source,
    )
    stream_meta = stream_bundle["stream_meta"]
    batch_df = stream_bundle["batch_df"]
    example_df = stream_bundle["example_df"]

    batch_df.to_csv(report_dir / "batched_stream.csv", index=False)
    example_df.to_csv(report_dir / "shuffled_prediction_stream.csv", index=False)
    np.savez(cs_dir / "adult_model_comparison_stream.npz", Z=Z)
    write_json(report_dir / "stream_metadata.json", stream_meta)

    D = Z.shape[1]
    N = Z.shape[0]
    h = build_hparams(D=D, N=N, alpha=args.alpha, parabola_grid_size=args.parabola_grid_size)

    summary_rows = []
    volume_curves = {}
    interval_blocks = []
    metric_names = stream_meta["metric_names"]
    metric_bounds = np.asarray([stream_meta["metric_bounds"][metric_name] for metric_name in metric_names], dtype=np.float64)
    if args.decision_rule == "target_positive":
        if args.target_metric is None:
            raise ValueError("--target-metric is required when --decision-rule target_positive.")
        if args.target_metric not in metric_names:
            if args.target_metric in stream_meta.get("reported_metric_names", []):
                raise ValueError(
                    f"Target metric '{args.target_metric}' is derived from multiple CS coordinates when "
                    "--conditional-proportion-source cs is used, so direct target stopping is not supported. "
                    f"Target one of the CS coordinates instead: {', '.join(metric_names)}"
                )
            raise ValueError(
                f"Unknown target metric '{args.target_metric}'. Available metrics: {', '.join(metric_names)}"
            )
        target_index = metric_names.index(args.target_metric)
    else:
        target_index = None

    for method_type in args.methods:
        print(f"Running {method_type}...")
        h_method = dict(h)
        h_method["interval_types"] = [method_type]
        method, method_name, _ = IntervalFactory.create_instance(method_type, h_method, Z)
        ts, means, lowers, uppers, volumes, volume_stds = method.run()
        stop = evaluate_stopping( # returns dict with keys: stop_t, decision, above, below, r_t
            lowers=np.asarray(lowers),
            uppers=np.asarray(uppers),
            volumes=np.asarray(volumes),
            r_tol=args.r_tol,
            decision_rule=args.decision_rule,
            target_volume=args.target_volume,
            target_index=target_index,
        )

        volume_curves[method_name] = np.asarray(volumes, dtype=np.float64)
        stop_idx = int(stop["stop_t"]) - 1
        final_idx = len(ts) - 1

        lowers_gain = (2.0 * np.asarray(lowers) - 1.0) * metric_bounds # (509, 6) where 509 is number of batches and 6 is number of metrics. Each entry is the lower bound on the gain for that metric at that batch, scaled to the original B-minus-A scale.
        uppers_gain = (2.0 * np.asarray(uppers) - 1.0) * metric_bounds
        means_gain = (2.0 * np.asarray(means) - 1.0) * metric_bounds
        report_metric_names, report_means_gain, report_lowers_gain, report_uppers_gain = report_metric_arrays(
            stream_meta=stream_meta,
            means_gain=means_gain,
            lowers_gain=lowers_gain,
            uppers_gain=uppers_gain,
        )

        per_metric_bounds = pd.DataFrame(
            {
                "metric": report_metric_names,
                "interpretation": [metric_interpretation(metric_name) for metric_name in report_metric_names],
                "mean_gain_at_stop": report_means_gain[stop_idx],
                "lower_gain_at_stop": report_lowers_gain[stop_idx],
                "upper_gain_at_stop": report_uppers_gain[stop_idx],
                "mean_gain_final": report_means_gain[final_idx],
                "lower_gain_final": report_lowers_gain[final_idx],
                "upper_gain_final": report_uppers_gain[final_idx],
                "significant_positive_at_stop": report_lowers_gain[stop_idx] > 0.0,
                "significant_negative_at_stop": report_uppers_gain[stop_idx] < 0.0,
            }
        )
        per_metric_bounds = append_cs_proportion_bounds(
            per_metric_bounds=per_metric_bounds,
            stream_meta=stream_meta,
            means_gain=means_gain,
            lowers_gain=lowers_gain,
            uppers_gain=uppers_gain,
            stop_idx=stop_idx,
            final_idx=final_idx,
        )
        safe_method_name = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        per_metric_bounds.to_csv(report_dir / f"bounds_{safe_method_name}.csv", index=False)
        interval_blocks.extend(
            [
                {
                    "method_name": method_name,
                    "batch_idx": int(stop["stop_t"]),
                    "lines": render_interval_lines(
                        method_name,
                        int(stop["stop_t"]),
                        per_metric_bounds,
                        prefix="stop-time",
                        mean_col="mean_gain_at_stop",
                        lower_col="lower_gain_at_stop",
                        upper_col="upper_gain_at_stop",
                    ),
                },
                {
                    "method_name": method_name,
                    "batch_idx": int(final_idx + 1),
                    "lines": render_interval_lines(
                        method_name,
                        int(final_idx + 1),
                        per_metric_bounds,
                        prefix="final",
                        mean_col="mean_gain_final",
                        lower_col="lower_gain_final",
                        upper_col="upper_gain_final",
                    ),
                },
            ]
        )
        np.savez(
            cs_dir / f"{safe_method_name}.npz",
            ts=np.asarray(ts),
            means=np.asarray(means),
            lowers=np.asarray(lowers),
            uppers=np.asarray(uppers),
            volumes=np.asarray(volumes),
            volume_stds=np.asarray(volume_stds),
        )

        summary_rows.append(
            {
                "method_type": method_type,
                "method_name": method_name,
                "stop_t": int(stop["stop_t"]),
                "decision": stop["decision"],
                "r_t": float(stop["r_t"]),
                "final_volume": float(volumes[final_idx]),
                "above_zero_metrics": format_dims(stop["above"], metric_names),
                "below_zero_metrics": format_dims(stop["below"], metric_names),
                "target_metric": args.target_metric if args.decision_rule == "target_positive" else "-",
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("final_volume")
    summary_df.to_csv(report_dir / "cs_summary.csv", index=False)

    metric_df = save_metric_summary(report_dir, stream_meta)
    plot_cumulative_means(batch_df=batch_df, plot_dir=plot_dir)
    plot_volumes(volume_curves=volume_curves, plot_dir=plot_dir)

    if args.metric_profile == "subgroup_accuracy":
        coordinate_description = "Subgroup coordinates are prevalence-weighted gains."
    elif (
        args.metric_profile == "subgroup_accuracy_conditional"
        and args.conditional_proportion_source == "cs"
    ):
        coordinate_description = (
            "CS coordinates contain weighted subgroup gains plus female and age_lt40 "
            "prevalence indicators; conditional subgroup intervals are reported as ratio transforms."
        )
    elif args.metric_profile == "subgroup_accuracy_conditional":
        coordinate_description = (
            "Subgroup coordinates target conditional subgroup gains via inverse-prevalence weighting."
        )
    else:
        coordinate_description = "All coordinates are bounded gains on the original B-minus-A scale."

    reported_names = stream_meta.get("reported_metric_names", metric_names)
    paper_lines = [
        f"Adult model-comparison experiment with model A={args.model_a} and model B={args.model_b}.",
        f"Test examples: {stream_meta['n_examples']}. Batch size: {args.batch_size}. Batches: {stream_meta['n_batches']}.",
        f"Decision rule: {args.decision_rule}.",
        f"Target metric: {args.target_metric}." if args.target_metric is not None else None,
        f"Metric profile: {args.metric_profile}. CS coordinates: {', '.join(metric_names)}.",
        f"Reported metrics: {', '.join(reported_names)}.",
        coordinate_description,
        "",
        "Empirical mean gains:",
    ]
    paper_lines = [line for line in paper_lines if line is not None]
    for _, row in metric_df.iterrows():
        paper_lines.append(
            f"  - {row['metric']}: {row['mean_gain_b_minus_a']:.6f} "
            f"({row['interpretation']}; {args.model_a}={row[f'{args.model_a}_score']:.6f}, "
            f"{args.model_b}={row[f'{args.model_b}_score']:.6f})"
        )
    paper_lines.append("")
    paper_lines.append("Confidence-sequence summary:")
    for _, row in summary_df.iterrows():
        paper_lines.append(
            f"  - {row['method_name']}: decision={row['decision']}, stop_t={int(row['stop_t'])}, "
            f"r_t={row['r_t']:.6f}, above_zero=[{row['above_zero_metrics']}], below_zero=[{row['below_zero_metrics']}]."
        )
    paper_lines.append("")
    paper_lines.append("Stop-time interval interpretation:")
    for block in interval_blocks:
        paper_lines.extend(block["lines"])
        paper_lines.append("")

    (report_dir / "paper_summary.txt").write_text("\n".join(paper_lines), encoding="utf-8")
    write_json(
        report_dir / "cs_run_metadata.json",
        {
            "alpha": args.alpha,
            "r_tol": args.r_tol,
            "methods": args.methods,
            "model_a": args.model_a,
            "model_b": args.model_b,
            "batch_size": args.batch_size,
            "shuffle_seed": args.shuffle_seed,
            "predictions_path": prediction_path,
            "decision_rule": args.decision_rule,
            "target_metric": args.target_metric,
            "metric_profile": args.metric_profile,
            "conditional_proportion_source": args.conditional_proportion_source,
            "target_volume": args.target_volume,
        },
    )

    print("\nMetric summary")
    print(metric_df.to_string(index=False))
    print("\nCS summary")
    print(summary_df.to_string(index=False))
    for block in interval_blocks:
        print("")
        for line in block["lines"]:
            print(line)
    print(f"\nSaved reports to {report_dir}")


if __name__ == "__main__":
    main()

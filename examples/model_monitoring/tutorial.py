from pathlib import Path
import csv
import sys
import zipfile

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_convex_sum import (  # noqa: E402
    HedgedCapitalConvexSumBoundingBox,
)
from adaptive_sample.conf_sequences.hedged_capital.hedged_factory import HedgedFactory  # noqa: E402


class StreamingBoundingBox(HedgedCapitalConvexSumBoundingBox):
    """Online interface for the bounding-box confidence sequence.

    The original class is optimized for a fixed data matrix and exposes `.run()`.
    This wrapper keeps the same statistical construction, but lets tutorial code
    append one vector observation at a time via `.observe(y_t)`.
    """

    def __init__(self, alpha, dimension):
        # The parent class expects a fixed-size data matrix.
        # In principle there shouldn't be a max, but this is due to how the parent class is implemented.
        max_observations = 100_000
        h = {
            "alpha": alpha,
            "batch_size": 1,
            "data_type": f"uniform_{dimension}d",
            "N": max_observations,
            "cut_to_simplex": False,
            "calculate_volume": False,
            "plot_grid_2d": False,
            "verbose_progress": False,
            "cache_hedged": "EXACT",
            "grid_resolution": 1000,
        }
        data = np.zeros((max_observations, dimension), dtype=float)
        super().__init__(h, data)
        self._n_seen = 0
        self._kd_calcs_cache = HedgedFactory.build_and_cache_Khedgeds(self.h, self.data)

    def observe(self, y_t, return_bounds=True):
        y_t = np.asarray(y_t, dtype=float)
        if y_t.shape != (self.D,):
            raise ValueError(f"Expected shape {(self.D,)}, got {y_t.shape}.")
        if np.any(y_t < 0.0) or np.any(y_t > 1.0):
            raise ValueError("The CS expects observations in [0, 1]^D.")
        if self._n_seen >= self.N:
            raise RuntimeError("Increase max_observations before adding more data.")

        self.data[self._n_seen] = y_t
        self._n_seen += 1

        mean_t = self.data[: self._n_seen].mean(axis=0)
        if not return_bounds:
            return mean_t, None

        lower_t, upper_t, _ = self._bounds_at_t(
            t=self._n_seen,
            Kd_calcs=self._kd_calcs_cache,
            C=1.0 / self.alpha,
        )
        lower_t = np.clip(lower_t, 0.0, 1.0)
        upper_t = np.clip(upper_t, 0.0, 1.0)
        return mean_t, (lower_t, upper_t)

    def current_bounds(self):
        if self._n_seen == 0:
            raise RuntimeError("No observations have been added yet.")
        lower_t, upper_t, _ = self._bounds_at_t(
            t=self._n_seen,
            Kd_calcs=self._kd_calcs_cache,
            C=1.0 / self.alpha,
        )
        return np.clip(lower_t, 0.0, 1.0), np.clip(upper_t, 0.0, 1.0)


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def load_adult_test_set(max_examples=3500, seed=7):
    adult_zip = REPO_ROOT / "adaptive_sample" / "conf_sequences" / "model_comparison" / "adult.zip"
    X, y = [], []

    with zipfile.ZipFile(adult_zip) as zf:
        rows = csv.DictReader(
            zf.open("adult.test").read().decode().splitlines()[1:],
            fieldnames=ADULT_COLUMNS,
            skipinitialspace=True,
        )
        for row in rows:
            row = {name: value.strip() for name, value in row.items()}
            if "?" in row.values():
                continue
            x_t = dict(row)
            for name in ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]:
                x_t[name] = float(x_t[name])
            X.append(x_t)
            y.append(int(row["income"].strip().rstrip(".") == ">50K"))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    if max_examples is not None:
        order = order[:max_examples]

    X = [X[i] for i in order]
    y = np.asarray([y[i] for i in order], dtype=int)
    return X, y


def model_a_score(x):
    return (
            -6.8
            + 0.045 * x["age"]
            + 0.38 * x["education_num"]
            + 0.018 * x["hours_per_week"]
            + 0.85 * (x["capital_gain"] > 0)
            + 0.45 * ("Married" in x["marital_status"])
            - 0.45 * (x["sex"] == "Female")
    )


def model_b_score(x):
    return (
            -8.6
            + 0.052 * x["age"]
            + 0.43 * x["education_num"]
            + 0.021 * x["hours_per_week"]
            + 1.15 * (x["capital_gain"] > 0)
            - 0.35 * (x["capital_loss"] > 0)
            + 0.65 * ("Married" in x["marital_status"])
            + 0.25 * (x["sex"] == "Female")
    )


def model_a_predict(x):
    return int(sigmoid(model_a_score(x)) >= 0.5)


def model_b_predict(x):
    return int(sigmoid(model_b_score(x)) >= 0.5)


X, y = load_adult_test_set()

REPORTED_METRICS = [
    "Global accuracy",
    "Accuracy on women",
    "Accuracy on men",
    "Accuracy age > 40",
    "Accuracy age <= 40",
]


def observation(x_t, z_t, model_predict):
    pred_t = model_predict(x_t)
    correct = float(pred_t == z_t)
    female = float(x_t["sex"] == "Female")
    older = float(x_t["age"] > 40)

    return np.array(
        [
            correct,
            correct * female,
            correct * (1.0 - female),
            correct * older,
            correct * (1.0 - older),
            female,
            older,
        ],
        dtype=float,
    )


def conditional_interval(num_lower, num_upper, den_lower, den_upper, clip_low, clip_high):
    den_lower = max(float(den_lower), 1e-12)
    den_upper = max(float(den_upper), 1e-12)
    candidates = np.array(
        [
            num_lower / den_lower,
            num_lower / den_upper,
            num_upper / den_lower,
            num_upper / den_upper,
        ]
    )
    return (
        float(np.clip(candidates.min(), clip_low, clip_high)),
        float(np.clip(candidates.max(), clip_low, clip_high)),
    )


def report_accuracy_intervals(mean, lower, upper):
    female_p = mean[5]
    older_p = mean[6]

    estimates = np.array(
        [
            mean[0],
            mean[1] / max(female_p, 1e-12),
            mean[2] / max(1.0 - female_p, 1e-12),
            mean[3] / max(older_p, 1e-12),
            mean[4] / max(1.0 - older_p, 1e-12),
        ]
    )

    intervals = [
        (lower[0], upper[0]),
        conditional_interval(lower[1], upper[1], lower[5], upper[5], 0.0, 1.0),
        conditional_interval(lower[2], upper[2], 1.0 - upper[5], 1.0 - lower[5], 0.0, 1.0),
        conditional_interval(lower[3], upper[3], lower[6], upper[6], 0.0, 1.0),
        conditional_interval(lower[4], upper[4], 1.0 - upper[6], 1.0 - lower[6], 0.0, 1.0),
    ]
    return estimates, np.asarray(intervals)


def plot_intervals(metric_names, estimates, intervals, title, xlabel, path, reference=None):
    y = np.arange(len(metric_names))[::-1]
    colors = ["#2563EB", "#DB2777", "#0891B2", "#D97706", "#7C3AED"]
    x_min = float(min(intervals[:, 0].min(), reference if reference is not None else intervals[:, 0].min()))
    x_max = float(max(intervals[:, 1].max(), reference if reference is not None else intervals[:, 1].max()))
    pad = max(0.025, 0.08 * (x_max - x_min))
    plot_min = x_min - pad if reference is not None else max(0.0, x_min - pad)
    plot_max = x_max + pad if reference is not None else min(1.0, x_max + pad)

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")

    for row in y:
        ax.axhspan(row - 0.36, row + 0.36, color="white", alpha=0.72, zorder=0)

    if reference is not None:
        ax.axvspan(reference, intervals[:, 1].max(), color="#ECFDF5", alpha=0.75, zorder=0)
        ax.axvline(reference, color="#111827", linestyle="--", linewidth=1.5, zorder=1)

    for i, (name, estimate, bounds) in enumerate(zip(metric_names, estimates, intervals)):
        row = y[i]
        lower, upper = bounds
        color = colors[i % len(colors)]
        ax.hlines(row, lower, upper, color=color, linewidth=4.0, alpha=0.92, zorder=2)
        ax.plot([lower, lower], [row - 0.10, row + 0.10], color=color, linewidth=2.0, zorder=2)
        ax.plot([upper, upper], [row - 0.10, row + 0.10], color=color, linewidth=2.0, zorder=2)
        ax.scatter(estimate, row, s=92, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        label_on_left = estimate > plot_min + 0.78 * (plot_max - plot_min)
        label_offset = -0.012 if label_on_left else 0.012
        ax.text(
            estimate + label_offset * (plot_max - plot_min),
            row + 0.18,
            f"{estimate:.1%}",
            color="#334155",
            fontsize=9,
            ha="right" if label_on_left else "left",
            va="bottom",
        )

    ax.set_xlim(plot_min, plot_max)

    ax.set_yticks(y)
    ax.set_yticklabels(metric_names, fontsize=11, color="#111827")
    ax.set_xlabel(xlabel, fontsize=11, color="#111827")
    ax.set_title(title, fontsize=14, color="#111827", pad=14)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="x", color="#CBD5E1", linewidth=1.0, alpha=0.55)
    ax.tick_params(axis="x", colors="#334155")
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


alpha = 0.05
asset_dir = SCRIPT_DIR / "assets"
asset_dir.mkdir(exist_ok=True)

accuracy_cs = StreamingBoundingBox(
    alpha=alpha,
    dimension=7,
)

latest_mean = None
latest_box = None
for x_t, z_t in zip(X, y):
    cs_observation = observation(x_t, z_t, model_b_predict)
    latest_mean, latest_box = accuracy_cs.observe(cs_observation, return_bounds=False)

# Compute the box at the final time. During live monitoring you can call
# `accuracy_cs.observe(y_t)` without `return_bounds=False` to get every box.
latest_box = accuracy_cs.current_bounds()

accuracy_estimates, accuracy_intervals = report_accuracy_intervals(
    latest_mean,
    latest_box[0],
    latest_box[1],
)

plot_intervals(
    REPORTED_METRICS,
    accuracy_estimates,
    accuracy_intervals,
    title="Model B Accuracy",
    xlabel="Accuracy",
    path=asset_dir / "tutorial_accuracy_intervals.png",
)


def gain_observation(x_t, z_t, model_a_predict, model_b_predict):
    correct_a = float(model_a_predict(x_t) == z_t)
    correct_b = float(model_b_predict(x_t) == z_t)
    gain = correct_b - correct_a
    female = float(x_t["sex"] == "Female")
    older = float(x_t["age"] > 40)

    raw_gain_vector = np.array(
        [
            gain,
            gain * female,
            gain * (1.0 - female),
            gain * older,
            gain * (1.0 - older),
        ],
        dtype=float,
    )
    scaled_gain_vector = 0.5 * (raw_gain_vector + 1.0)

    return np.concatenate([scaled_gain_vector, np.array([female, older])])


def report_gain_intervals(mean_scaled, lower_scaled, upper_scaled):
    mean = mean_scaled.copy()
    lower = lower_scaled.copy()
    upper = upper_scaled.copy()

    mean[:5] = 2.0 * mean[:5] - 1.0
    lower[:5] = 2.0 * lower[:5] - 1.0
    upper[:5] = 2.0 * upper[:5] - 1.0

    female_p = mean[5]
    older_p = mean[6]
    estimates = np.array(
        [
            mean[0],
            mean[1] / max(female_p, 1e-12),
            mean[2] / max(1.0 - female_p, 1e-12),
            mean[3] / max(older_p, 1e-12),
            mean[4] / max(1.0 - older_p, 1e-12),
        ]
    )

    intervals = [
        (lower[0], upper[0]),
        conditional_interval(lower[1], upper[1], lower[5], upper[5], -1.0, 1.0),
        conditional_interval(lower[2], upper[2], 1.0 - upper[5], 1.0 - lower[5], -1.0, 1.0),
        conditional_interval(lower[3], upper[3], lower[6], upper[6], -1.0, 1.0),
        conditional_interval(lower[4], upper[4], 1.0 - upper[6], 1.0 - lower[6], -1.0, 1.0),
    ]
    return estimates, np.asarray(intervals)


gain_cs = StreamingBoundingBox(
    alpha=alpha,
    dimension=7,
)

latest_gain_mean = None
latest_gain_box = None
for x_t, z_t in zip(X, y):
    cs_observation = gain_observation(x_t, z_t, model_a_predict, model_b_predict)
    latest_gain_mean, latest_gain_box = gain_cs.observe(cs_observation, return_bounds=False)

latest_gain_box = gain_cs.current_bounds()

gain_estimates, gain_intervals = report_gain_intervals(
    latest_gain_mean,
    latest_gain_box[0],
    latest_gain_box[1],
)

plot_intervals(
    REPORTED_METRICS,
    gain_estimates,
    gain_intervals,
    title="Accuracy Gain: Model B vs. Model A",
    xlabel="Accuracy gain",
    path=asset_dir / "tutorial_gain_intervals.png",
    reference=0.0,
)

print("Saved figures:")
print(f"  {asset_dir / 'tutorial_accuracy_intervals.png'}")
print(f"  {asset_dir / 'tutorial_accuracy_intervals.svg'}")
print(f"  {asset_dir / 'tutorial_gain_intervals.png'}")
print(f"  {asset_dir / 'tutorial_gain_intervals.svg'}")
print(f"Number of observations made: {accuracy_cs._n_seen}")

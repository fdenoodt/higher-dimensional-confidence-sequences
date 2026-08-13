import json
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

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

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

TARGET_COLUMN = "income"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def model_comparison_dir() -> Path:
    return Path(__file__).resolve().parent


def adult_zip_path() -> Path:
    return model_comparison_dir() / "adult.zip"


def artifacts_dir(experiment_name: str = "adult_model_comparison") -> Path:
    return model_comparison_dir() / "artifacts" / experiment_name


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_adult_from_zip(zip_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as zf:
        train_df = pd.read_csv(
            zf.open("adult.data"),
            header=None,
            names=ADULT_COLUMNS,
            skipinitialspace=True,
            na_values="?",
        )
        test_df = pd.read_csv(
            zf.open("adult.test"),
            header=None,
            names=ADULT_COLUMNS,
            skiprows=1,
            skipinitialspace=True,
            na_values="?",
        )

    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(str).str.strip()
    test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].astype(str).str.strip().str.rstrip(".")

    return train_df, test_df


def prepare_adult_frames(zip_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = load_adult_from_zip(zip_path)

    for df in (train_df, test_df):
        df["label"] = (df[TARGET_COLUMN] == ">50K").astype(int)
        df["age_group"] = np.where(df["age"] < 40, "lt40", "ge40")
        df["race_group"] = np.where(df["race"] == "White", "white", "non_white")

    return train_df, test_df


def write_json(path: Path, payload: Dict) -> None:
    serializable = _to_serializable(payload)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def clipped_log_loss_binary(y_true: np.ndarray, prob_pos: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(prob_pos, eps, 1.0 - eps)
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def brier_loss_binary(y_true: np.ndarray, prob_pos: np.ndarray) -> np.ndarray:
    return (prob_pos - y_true) ** 2


def add_prediction_columns(df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    #########################
    # THE IMPORTANT PART: this is where we add all the columns to the dataframe that we will then batch and stream.
    # We compute these columns once here for efficiency, rather than recomputing them for each batch.
    #########################
    out = df.copy()

    y_true = out["label"].to_numpy(dtype=np.float64)
    prob_a = out[f"prob_{model_a}"].to_numpy(dtype=np.float64)
    prob_b = out[f"prob_{model_b}"].to_numpy(dtype=np.float64)
    pred_a = out[f"pred_{model_a}"].to_numpy(dtype=np.int64)
    pred_b = out[f"pred_{model_b}"].to_numpy(dtype=np.int64)

    correct_a = (pred_a == y_true).astype(np.float64)
    correct_b = (pred_b == y_true).astype(np.float64)

    out[f"correct_{model_a}"] = correct_a
    out[f"correct_{model_b}"] = correct_b
    out["acc_gain_b_minus_a"] = correct_b - correct_a
    out["brier_gain_b_minus_a"] = (
            brier_loss_binary(y_true, prob_a) - brier_loss_binary(y_true, prob_b)
    )
    out["logloss_gain_b_minus_a"] = (
            clipped_log_loss_binary(y_true, prob_a) - clipped_log_loss_binary(y_true, prob_b)
    )

    out["female_acc_gain"] = np.where(out["sex"] == "Female", out["acc_gain_b_minus_a"], 0.0)
    out["male_acc_gain"] = np.where(out["sex"] == "Male", out["acc_gain_b_minus_a"], 0.0)
    out["lt40_acc_gain"] = np.where(out["age_group"] == "lt40", out["acc_gain_b_minus_a"], 0.0)
    out["ge40_acc_gain"] = np.where(out["age_group"] == "ge40", out["acc_gain_b_minus_a"], 0.0)

    return out


def metric_profile_names() -> List[str]:
    return ["subgroup_accuracy", "subgroup_accuracy_conditional", "strict_performance"]


def conditional_proportion_source_names() -> List[str]:
    return ["empirical", "cs"]


def stream_metric_columns(metric_profile: str, conditional_proportion_source: str = "empirical") -> List[str]:
    if metric_profile == "subgroup_accuracy":
        return [
            "acc_gain_b_minus_a",
            "brier_gain_b_minus_a",
            "female_acc_gain",
            "male_acc_gain",
            "lt40_acc_gain",
            "ge40_acc_gain",
        ]
    if metric_profile == "subgroup_accuracy_conditional":
        if conditional_proportion_source == "cs":
            return [
                "acc_gain_b_minus_a", # p(model_b correct) - p(model_a correct)
                "brier_gain_b_minus_a",
                "female_acc_gain", # p(model_b correct, sex=female) - p(model_a correct, sex=female)
                "male_acc_gain",
                "lt40_acc_gain",
                "ge40_acc_gain",
                "female_indicator", # for p(x=female)
                "lt40_indicator", # for p(x=lt40)
            ]
        return [
            "acc_gain_b_minus_a",
            "brier_gain_b_minus_a",
            "female_acc_gain_conditional",
            "male_acc_gain_conditional",
            "lt40_acc_gain_conditional",
            "ge40_acc_gain_conditional",
        ]
    if metric_profile == "strict_performance":
        return [
            "acc_gain_b_minus_a",
            "brier_gain_b_minus_a",
            "true_class_conf_gain_b_minus_a",
        ]
    raise ValueError(f"Unknown metric profile: {metric_profile}")


def stream_metric_names(metric_profile: str, conditional_proportion_source: str = "empirical") -> List[str]:
    if metric_profile == "subgroup_accuracy":
        return [
            "overall_accuracy",
            "overall_brier",
            "female_accuracy",
            "male_accuracy",
            "age_lt40_accuracy",
            "age_ge40_accuracy",
        ]
    if metric_profile == "subgroup_accuracy_conditional":
        if conditional_proportion_source == "cs":
            return [
                "overall_accuracy",
                "overall_brier",
                "female_accuracy_weighted",
                "male_accuracy_weighted",
                "age_lt40_accuracy_weighted",
                "age_ge40_accuracy_weighted",
                "female_proportion",
                "age_lt40_proportion",
            ]
        return [
            "overall_accuracy",
            "overall_brier",
            "female_accuracy_conditional",
            "male_accuracy_conditional",
            "age_lt40_accuracy_conditional",
            "age_ge40_accuracy_conditional",
        ]
    if metric_profile == "strict_performance":
        return [
            "overall_accuracy",
            "overall_brier",
            "true_class_confidence",
        ]
    raise ValueError(f"Unknown metric profile: {metric_profile}")


def reported_metric_names(metric_profile: str) -> List[str]:
    if metric_profile == "subgroup_accuracy_conditional":
        return [
            "overall_accuracy",
            "overall_brier",
            "female_accuracy_conditional",
            "male_accuracy_conditional",
            "age_lt40_accuracy_conditional",
            "age_ge40_accuracy_conditional",
        ]
    return stream_metric_names(metric_profile)


def build_batched_stream(
        prediction_df: pd.DataFrame,
        model_a: str,
        model_b: str,
        batch_size: int,
        shuffle_seed: int,
        metric_profile: str,
        conditional_proportion_source: str = "empirical",
) -> Tuple[np.ndarray, Dict]:
    if conditional_proportion_source not in conditional_proportion_source_names():
        raise ValueError(
            "Unknown conditional_proportion_source "
            f"'{conditional_proportion_source}'. Expected one of: "
            f"{', '.join(conditional_proportion_source_names())}"
        )

    # this will add all the e.g., "acc_gain_b_minus_a", "brier_gain_b_minus_a", ... columns to the dataframe, which we will then batch and stream
    enriched = add_prediction_columns(prediction_df, model_a=model_a, model_b=model_b)

    enriched["female_indicator"] = (enriched["sex"] == "Female").astype(np.float64)
    enriched["lt40_indicator"] = (enriched["age_group"] == "lt40").astype(np.float64)

    subgroup_props = {
        "female": float(enriched["female_indicator"].mean()),
        "male": float(1.0 - enriched["female_indicator"].mean()),
        "age_lt40": float(enriched["lt40_indicator"].mean()),
        "age_ge40": float(1.0 - enriched["lt40_indicator"].mean()),
    }

    enriched["female_acc_gain_conditional"] = np.where(
        enriched["sex"] == "Female",
        enriched["acc_gain_b_minus_a"] / subgroup_props["female"],
        0.0,
    )
    enriched["male_acc_gain_conditional"] = np.where(
        enriched["sex"] == "Male",
        enriched["acc_gain_b_minus_a"] / subgroup_props["male"],
        0.0,
    )
    enriched["lt40_acc_gain_conditional"] = np.where(
        enriched["age_group"] == "lt40",
        enriched["acc_gain_b_minus_a"] / subgroup_props["age_lt40"],
        0.0,
    )
    enriched["ge40_acc_gain_conditional"] = np.where(
        enriched["age_group"] == "ge40",
        enriched["acc_gain_b_minus_a"] / subgroup_props["age_ge40"],
        0.0,
    )

    y_true = enriched["label"].to_numpy(dtype=np.float64)
    prob_a = enriched[f"prob_{model_a}"].to_numpy(dtype=np.float64)
    prob_b = enriched[f"prob_{model_b}"].to_numpy(dtype=np.float64)
    enriched["true_class_conf_gain_b_minus_a"] = np.where(y_true == 1.0, prob_b - prob_a, prob_a - prob_b)
    enriched[f"true_class_conf_{model_a}"] = np.where(y_true == 1.0, prob_a, 1.0 - prob_a)
    enriched[f"true_class_conf_{model_b}"] = np.where(y_true == 1.0, prob_b, 1.0 - prob_b)
    enriched[f"brier_loss_{model_a}"] = brier_loss_binary(y_true, prob_a)
    enriched[f"brier_loss_{model_b}"] = brier_loss_binary(y_true, prob_b)

    metric_columns = stream_metric_columns(
        metric_profile,
        conditional_proportion_source=conditional_proportion_source,
    )
    metric_names = stream_metric_names(
        metric_profile,
        conditional_proportion_source=conditional_proportion_source,
    )
    report_metric_names = reported_metric_names(metric_profile)
    metric_bounds = []
    for column in metric_columns:
        if conditional_proportion_source == "empirical" and column == "female_acc_gain_conditional":
            metric_bounds.append(1.0 / subgroup_props["female"])
        elif conditional_proportion_source == "empirical" and column == "male_acc_gain_conditional":
            metric_bounds.append(1.0 / subgroup_props["male"])
        elif conditional_proportion_source == "empirical" and column == "lt40_acc_gain_conditional":
            metric_bounds.append(1.0 / subgroup_props["age_lt40"])
        elif conditional_proportion_source == "empirical" and column == "ge40_acc_gain_conditional":
            metric_bounds.append(1.0 / subgroup_props["age_ge40"])
        else:
            metric_bounds.append(1.0)
    metric_bounds = np.asarray(metric_bounds, dtype=np.float64)

    rng = np.random.default_rng(shuffle_seed)
    perm = rng.permutation(len(enriched))
    shuffled = enriched.iloc[perm].reset_index(drop=True)

    batches = []
    batch_rows = []
    for start in range(0, len(shuffled), batch_size):
        batch = shuffled.iloc[start: start + batch_size]
        x_t = batch[metric_columns].mean(axis=0).to_numpy(dtype=np.float64)

        batches.append(0.5 * (x_t / metric_bounds + 1.0))  # normalize coordinatewise to [0, 1]

        batch_rows.append(
            {
                "batch_index": len(batch_rows),
                "start_row": int(start),
                "end_row_exclusive": int(start + len(batch)),
                "batch_size": int(len(batch)),
                **{name: float(val) for name, val in zip(metric_names, x_t)},
            }
        )

    conditional_gains = {
        "female_accuracy_conditional": float(
            shuffled.loc[shuffled["sex"] == "Female", "acc_gain_b_minus_a"].mean()
        ),
        "male_accuracy_conditional": float(
            shuffled.loc[shuffled["sex"] == "Male", "acc_gain_b_minus_a"].mean()
        ),
        "age_lt40_accuracy_conditional": float(
            shuffled.loc[shuffled["age_group"] == "lt40", "acc_gain_b_minus_a"].mean()
        ),
        "age_ge40_accuracy_conditional": float(
            shuffled.loc[shuffled["age_group"] == "ge40", "acc_gain_b_minus_a"].mean()
        ),
    }
    reported_metric_means = {
        name: float(shuffled[col].mean()) for name, col in zip(metric_names, metric_columns)
    }
    if metric_profile == "subgroup_accuracy_conditional":
        reported_metric_means = {
            "overall_accuracy": float(shuffled["acc_gain_b_minus_a"].mean()),
            "overall_brier": float(shuffled["brier_gain_b_minus_a"].mean()),
            **conditional_gains,
        }

    reported_metric_scores = {
        "overall_accuracy": {
            model_a: float(shuffled[f"correct_{model_a}"].mean()),
            model_b: float(shuffled[f"correct_{model_b}"].mean()),
        },
        "overall_brier": {
            model_a: float(shuffled[f"brier_loss_{model_a}"].mean()),
            model_b: float(shuffled[f"brier_loss_{model_b}"].mean()),
        },
        "true_class_confidence": {
            model_a: float(shuffled[f"true_class_conf_{model_a}"].mean()),
            model_b: float(shuffled[f"true_class_conf_{model_b}"].mean()),
        },
        "female_accuracy": {
            model_a: float(np.where(shuffled["sex"] == "Female", shuffled[f"correct_{model_a}"], 0.0).mean()),
            model_b: float(np.where(shuffled["sex"] == "Female", shuffled[f"correct_{model_b}"], 0.0).mean()),
        },
        "male_accuracy": {
            model_a: float(np.where(shuffled["sex"] == "Male", shuffled[f"correct_{model_a}"], 0.0).mean()),
            model_b: float(np.where(shuffled["sex"] == "Male", shuffled[f"correct_{model_b}"], 0.0).mean()),
        },
        "age_lt40_accuracy": {
            model_a: float(np.where(shuffled["age_group"] == "lt40", shuffled[f"correct_{model_a}"], 0.0).mean()),
            model_b: float(np.where(shuffled["age_group"] == "lt40", shuffled[f"correct_{model_b}"], 0.0).mean()),
        },
        "age_ge40_accuracy": {
            model_a: float(np.where(shuffled["age_group"] == "ge40", shuffled[f"correct_{model_a}"], 0.0).mean()),
            model_b: float(np.where(shuffled["age_group"] == "ge40", shuffled[f"correct_{model_b}"], 0.0).mean()),
        },
        "female_accuracy_conditional": {
            model_a: float(shuffled.loc[shuffled["sex"] == "Female", f"correct_{model_a}"].mean()),
            model_b: float(shuffled.loc[shuffled["sex"] == "Female", f"correct_{model_b}"].mean()),
        },
        "male_accuracy_conditional": {
            model_a: float(shuffled.loc[shuffled["sex"] == "Male", f"correct_{model_a}"].mean()),
            model_b: float(shuffled.loc[shuffled["sex"] == "Male", f"correct_{model_b}"].mean()),
        },
        "age_lt40_accuracy_conditional": {
            model_a: float(shuffled.loc[shuffled["age_group"] == "lt40", f"correct_{model_a}"].mean()),
            model_b: float(shuffled.loc[shuffled["age_group"] == "lt40", f"correct_{model_b}"].mean()),
        },
        "age_ge40_accuracy_conditional": {
            model_a: float(shuffled.loc[shuffled["age_group"] == "ge40", f"correct_{model_a}"].mean()),
            model_b: float(shuffled.loc[shuffled["age_group"] == "ge40", f"correct_{model_b}"].mean()),
        },
    }

    meta = {
        "metric_names": metric_names,
        "reported_metric_names": report_metric_names,
        "metric_columns": metric_columns,
        "metric_bounds": {name: float(bound) for name, bound in zip(metric_names, metric_bounds)},
        "conditional_proportion_source": conditional_proportion_source,
        "model_a": model_a,
        "model_b": model_b,
        "metric_profile": metric_profile,
        "batch_size": batch_size,
        "shuffle_seed": shuffle_seed,
        "n_examples": int(len(shuffled)),
        "n_batches": int(len(batch_rows)),
        "subgroup_proportions": subgroup_props,
        "conditional_accuracy_gains": conditional_gains,
        "reported_metric_means": reported_metric_means,
        "reported_metric_scores": reported_metric_scores,
        "weighted_metric_means": {
            name: float(shuffled[col].mean()) for name, col in zip(metric_names, metric_columns)
        },
        "Z_definition": "Z_t[d] = 0.5 * (X_t[d] / B_d + 1), where X_t[d] is the batch mean gain and B_d is that coordinate's absolute bound.",
    }

    batch_df = pd.DataFrame(batch_rows)
    batches = np.asarray(batches, dtype=np.float64)
    return (
        batches, # Z stream that gets fed to CS
        {"stream_meta": meta, # summary of how stream was built
         "batch_df": batch_df, # per batch mean gains x_t
         "example_df": shuffled # per example gains
         }
    )


def summarize_model_metrics(pred_df: pd.DataFrame, model_names: Sequence[str]) -> pd.DataFrame:
    y_true = pred_df["label"].to_numpy(dtype=np.float64)
    rows = []
    for model_name in model_names:
        prob = pred_df[f"prob_{model_name}"].to_numpy(dtype=np.float64)
        pred = pred_df[f"pred_{model_name}"].to_numpy(dtype=np.int64)
        rows.append(
            {
                "model": model_name,
                "accuracy": float((pred == y_true).mean()),
                "brier": float(brier_loss_binary(y_true, prob).mean()),
                "log_loss": float(clipped_log_loss_binary(y_true, prob).mean()),
            }
        )
    return pd.DataFrame(rows)

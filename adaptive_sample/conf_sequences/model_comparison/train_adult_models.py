import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from adaptive_sample.conf_sequences.model_comparison.adult_model_comparison_utils import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    adult_zip_path,
    artifacts_dir,
    ensure_dir,
    prepare_adult_frames,
    summarize_model_metrics,
    write_json,
)


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def build_histgb_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(random_state: int) -> dict:
    return {
        "logreg": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gbdt": Pipeline(
            steps=[
                ("preprocessor", build_histgb_preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=300,
                        min_samples_leaf=20,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "weak_tree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=4,
                        min_samples_leaf=50,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Adult-income comparison models and store predictions.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=adult_zip_path(),
        help="Path to adult.zip.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="adult_model_comparison",
        help="Artifact subdirectory name.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=7,
        help="Random seed used for model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df, test_df = prepare_adult_frames(args.zip_path)

    X_train = train_df.drop(columns=[TARGET_COLUMN, "label", "age_group", "race_group"])
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    X_test = test_df.drop(columns=[TARGET_COLUMN, "label", "age_group", "race_group"])

    artifact_root = ensure_dir(artifacts_dir(args.experiment_name))
    model_dir = ensure_dir(artifact_root / "models")
    prediction_dir = ensure_dir(artifact_root / "predictions")
    report_dir = ensure_dir(artifact_root / "reports")

    models = build_models(random_state=args.random_state)
    metrics_rows = []
    prediction_df = test_df.copy()
    prediction_df.insert(0, "row_id", np.arange(len(prediction_df), dtype=int))

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, model_dir / f"{model_name}.joblib")

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        prediction_df[f"prob_{model_name}"] = prob
        prediction_df[f"pred_{model_name}"] = pred

    metrics_df = summarize_model_metrics(prediction_df, model_names=list(models.keys()))
    metrics_rows.extend(metrics_df.to_dict(orient="records"))

    prediction_path = prediction_dir / "adult_test_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)
    metrics_df.to_csv(report_dir / "test_model_metrics.csv", index=False)

    payload = {
        "zip_path": args.zip_path,
        "random_state": args.random_state,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "models": {
            "logreg": {
                "type": "LogisticRegression",
                "solver": "lbfgs",
                "max_iter": 2000,
            },
            "hist_gbdt": {
                "type": "HistGradientBoostingClassifier",
                "max_depth": 6,
                "learning_rate": 0.05,
                "max_iter": 300,
                "min_samples_leaf": 20,
            },
            "weak_tree": {
                "type": "DecisionTreeClassifier",
                "max_depth": 4,
                "min_samples_leaf": 50,
            },
        },
        "test_metrics": metrics_rows,
        "artifacts": {
            "prediction_csv": prediction_path,
            "metrics_csv": report_dir / "test_model_metrics.csv",
            "model_dir": model_dir,
        },
    }
    write_json(report_dir / "training_summary.json", payload)

    print("\nTest metrics")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved predictions to {prediction_path}")


if __name__ == "__main__":
    main()

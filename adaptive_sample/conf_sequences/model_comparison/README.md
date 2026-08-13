# Adult Model Comparison Experiments

This folder contains a real-data multivariate confidence-sequence application on the UCI Adult dataset.

## Files

- `train_adult_models.py`
  Trains the comparison models, saves them, and stores test-set predictions.
- `run_adult_model_comparison_cs.py`
  Builds a bounded multivariate stream from the stored predictions and runs the requested confidence-sequence methods.
- `adult_model_comparison_utils.py`
  Shared dataset loading, metric construction, batching, and reporting utilities.

The following commands run the experiments from the paper. The subgroup ratios are obtained directly from the CS.

```text
python -u -m adaptive_sample.conf_sequences.model_comparison.run_adult_model_comparison_cs --experiment-name adult_model_comparison_target_accuracy_conditional1 --model-a logreg --model-b hist_gbdt --metric-profile subgroup_accuracy_conditional --decision-rule target_positive --target-metric overall_accuracy --batch-size 32 --r-tol 0.005 --conditional-proportion-source cs --methods CONF_SPHERE
```

```text
python -u -m adaptive_sample.conf_sequences.model_comparison.run_adult_model_comparison_cs --experiment-name adult_model_comparison_target_accuracy_conditional2 --model-a logreg --model-b hist_gbdt --metric-profile subgroup_accuracy_conditional --decision-rule target_positive --target-metric overall_accuracy --batch-size 32 --r-tol 0.005 --conditional-proportion-source cs --methods BANACH_SPHERE
```

```text
python -u -m adaptive_sample.conf_sequences.model_comparison.run_adult_model_comparison_cs --experiment-name adult_model_comparison_target_accuracy_conditional3 --model-a logreg --model-b hist_gbdt --metric-profile subgroup_accuracy_conditional --decision-rule target_positive --target-metric overall_accuracy --batch-size 32 --r-tol 0.005 --conditional-proportion-source cs --methods HEDGE_nd_BBX
```

```text
python -u -m adaptive_sample.conf_sequences.model_comparison.run_adult_model_comparison_cs --experiment-name adult_model_comparison_target_accuracy_conditional4 --model-a logreg --model-b hist_gbdt --metric-profile subgroup_accuracy_conditional --decision-rule target_positive --target-metric overall_accuracy --batch-size 32 --r-tol 0.005 --conditional-proportion-source cs --methods HEDGE_nd_ELLIP
```

```text
python -u -m adaptive_sample.conf_sequences.model_comparison.run_adult_model_comparison_cs --experiment-name adult_model_comparison_target_accuracy_conditional5 --model-a logreg --model-b hist_gbdt --metric-profile subgroup_accuracy_conditional --decision-rule target_positive --target-metric overall_accuracy --batch-size 32 --r-tol 0.005 --conditional-proportion-source cs --methods HEDGE_nd_ELLIP_SAFE
```

```text
python -u -m adaptive_sample.conf_sequences.model_comparison.run_adult_model_comparison_cs --experiment-name adult_model_comparison_target_accuracy_conditional6 --model-a logreg --model-b hist_gbdt --metric-profile subgroup_accuracy_conditional --decision-rule target_positive --target-metric overall_accuracy --batch-size 32 --r-tol 0.005 --conditional-proportion-source cs --methods HEDGE_nd_ELLIP_BBX
```

```text
python -u -m adaptive_sample.conf_sequences.model_comparison.run_adult_model_comparison_cs --experiment-name adult_model_comparison_target_accuracy_conditional7 --model-a logreg --model-b hist_gbdt --metric-profile subgroup_accuracy_conditional --decision-rule target_positive --target-metric overall_accuracy --batch-size 32 --r-tol 0.005 --conditional-proportion-source cs --methods HEDGE_nd_ELLIP_BBX_SAFE
```

```text
python -u -m adaptive_sample.conf_sequences.model_comparison.run_adult_model_comparison_cs --experiment-name adult_model_comparison_target_accuracy_conditional8 --model-a logreg --model-b hist_gbdt --metric-profile subgroup_accuracy_conditional --decision-rule target_positive --target-metric overall_accuracy --batch-size 32 --r-tol 0.005 --conditional-proportion-source cs --methods HEDGE_nd_BONF
```

```text
python -u -m adaptive_sample.conf_sequences.model_comparison.run_adult_model_comparison_cs --experiment-name adult_model_comparison_target_accuracy_conditional9 --model-a logreg --model-b hist_gbdt --metric-profile subgroup_accuracy_conditional --decision-rule target_positive --target-metric overall_accuracy --batch-size 32 --r-tol 0.005 --conditional-proportion-source cs --methods NORMALIZED_ELLIP
```

Generated models, predictions, figures, and JSON summaries are written under the ignored `artifacts/` directory.

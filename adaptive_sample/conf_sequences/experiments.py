from typing import List, Optional
import json
import os
import re
from datetime import datetime
import numpy as np
from joblib import Parallel, delayed

from shared.decorators import Key
from adaptive_sample.conf_sequences import BaseConfidenceSequence, MultiDimensionalConfidenceSequence
from adaptive_sample.conf_sequences.hedged_capital import HedgedCapitalConvexSum
from adaptive_sample.conf_sequences.hedged_capital.hedged_factory import HedgedFactory
from adaptive_sample.conf_sequences.interval_factory import IntervalFactory
from adaptive_sample.plotting import Plotter
from adaptive_sample.shared_code import DataDistributionChecker, StaticRunTracker
from tqdm import tqdm  # Ensure this import is at the top of the file


class Experiments:
    @staticmethod
    def _json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (set, tuple)):
            return list(obj)
        return str(obj)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")

    @staticmethod
    def _create_experiment_run_dir(h: dict, experiment_name: str):
        log_root = h['log_path']
        try:
            key_name = Key.name(h)
        except Exception:
            key_name = "run"
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(log_root, "experiment_data", experiment_name, key_name, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir, run_id, key_name

    @staticmethod
    def _write_json(path: str, payload: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=Experiments._json_safe)

    @staticmethod
    def _save_method_npz(run_dir: str, method_idx: int, method_name: str, **arrays) -> str:
        safe_name = Experiments._sanitize_name(method_name)
        filename = f"method_{method_idx:02d}_{safe_name}.npz"
        path = os.path.join(run_dir, filename)
        np.savez(path, **arrays)
        return filename

    @staticmethod
    def _reference_volume_lines(h: dict, data, t_last: int) -> List[dict]:
        ref_lines = []
        for interval_type in ["HEDGE_nd_GRID", "HEDGE_nd_BONF"]:
            try:
                method, method_name, method_color = IntervalFactory.create_instance(interval_type, h, data)
            except Exception as exc:
                print(f"Warning: failed to build {interval_type} reference line: {exc}")
                continue
            method: MultiDimensionalConfidenceSequence = method
            vol, _ = method.compute_volume_at(t=t_last)
            ref_lines.append({
                "y": vol,
                "label": f"{method_name} ref",
                "color": method_color,
                "linestyle": "--",
            })
        return ref_lines
    @staticmethod
    def l1_error_experiment(h: dict, N, alpha, batch_size, data_type, stop_early_threshold):
        raise NotImplementedError("This function is deprecated.")
        """
        Perform 1,000 repeats. Compute how many ground true means are within the CI.
        Report both the error for an individual confidence interval and over the confidence sequence.
        """
        nb_runs = h['perform_l1_err_experiment_nb_runs']  # 1_000

        interval_types = h['interval_types']  # e.g. ["T_CI", "T_CI_BONF", "HEDGE_CI", "HEDGE_CI_SUM"]
        methods_str = [IntervalFactory.create_instance(method, h)[1] for method in interval_types]
        methods_funcs: List[BaseConfidenceSequence] = [IntervalFactory.create_instance(method, h)[0] for method in
                                                       interval_types]

        method_errors_individual = {method: 0 for method in methods_str}
        method_errors_global = {method: 0 for method in methods_str}
        method_errors_early_stopping = {method: 0 for method in methods_str}

        for run_idx in range(nb_runs):
            if run_idx % 100 == 0:
                print(f"Run {run_idx + 1}/{nb_runs}")
            data, true_mean = DataDistributionChecker.get_data(data_type, N)  # or "fixed" for constant data

            # Compute confidence intervals
            results = []
            for method_name, method_func in zip(methods_str, methods_funcs):
                method_data = method_func(data)
                results.append({"name": method_name, "data": method_data})
                # results[i]["data"] is a tuple (ts, means, lowers, uppers) where ts is the time points,

            for result in results:
                method_name = result["name"]
                ts, means, lowers, uppers = result["data"]

                # Convert lowers and uppers to NumPy arrays for comparison
                lowers = np.array(lowers)
                uppers = np.array(uppers)

                # cap between 0 and 1 if data_type is not "gaussian"
                if data_type != "gaussian":
                    lowers = np.clip(lowers, 0.0, 1.0)
                    uppers = np.clip(uppers, 0.0, 1.0)

                # Check if the true mean is within the confidence interval at any time step
                contains_true_mean = np.logical_and(lowers <= true_mean, uppers >= true_mean)
                # "true" means "good", so inside the interval

                # Count as an error if the true mean is not within the confidence interval
                if not np.all(contains_true_mean):
                    method_errors_global[method_name] += 1

                # average individual error for this run
                method_errors_individual[method_name] += 1 - np.mean(contains_true_mean)

                # take the first confidence that is narrow enough and check if it contains the true mean
                narrow_idx = np.where(np.abs(uppers - lowers) < stop_early_threshold)[0]
                if len(narrow_idx) > 0:  # there is at least one narrow confidence interval
                    first_narrow_idx = narrow_idx[0]
                else:
                    first_narrow_idx = -1  # no narrow confidence interval found so take the last one
                # Check if the true mean is within the first narrow confidence interval,
                # if not we count it as an error
                if not (lowers[first_narrow_idx] <= true_mean <= uppers[first_narrow_idx]):
                    method_errors_early_stopping[method_name] += 1

        for method_name in method_errors_global:
            method_errors_global[method_name] = (method_errors_global[method_name] / nb_runs) * 100
            method_errors_individual[method_name] = (method_errors_individual[method_name] / nb_runs) * 100
            method_errors_early_stopping[method_name] = (method_errors_early_stopping[method_name] / nb_runs) * 100

        Plotter.plot_l1_errors(h, method_errors_individual, method_errors_global,
                               method_errors_early_stopping,
                               target_percentage=alpha,
                               N=N, batch_size=batch_size,
                               data_type=data_type)

    @staticmethod
    def coverage_experiment(h, N, alpha, batch_size, data_type, stop_early_threshold):
        """
        Perform coverage experiment to analyze the distribution of distances between
        the empirical mean and the true mean after the stopping criterion is achieved.
        """
        nb_runs = h['perform_coverage_experiment_nb_runs']  # e.g. 5_000
        StaticRunTracker.reset_run_number()

        interval_types = h['interval_types']
        data, true_mean = DataDistributionChecker.get_data(data_type, N)

        methods_str = [IntervalFactory.create_instance(method, h, data)[1] for method in interval_types]
        distances_per_method = {method_name: [] for method_name in methods_str}
        avg_intervals_per_method = {method_name: [] for method_name in methods_str}  # Track intervals before narrow idx

        def process_run(run_idx):
            StaticRunTracker.increment_run_number()

            try:
                results = []
                for interval_type in interval_types:
                    method_func, method_name, _ = IntervalFactory.create_instance(interval_type, h, data)
                    results.append({"name": method_name, "data": method_func.run()})
            except AssertionError as e:
                print(f"Warning: Assertion error in run {run_idx}, skipping this run.")
                return None

            run_distances = {}
            intervals_count = {}
            for result in results:
                method_name = result["name"]
                ts, means, lowers, uppers, _, _ = result["data"]

                lowers = np.array(lowers)
                uppers = np.array(uppers)

                narrow_idx = np.where(np.abs(uppers - lowers) < stop_early_threshold)[0]
                first_narrow_idx = narrow_idx[0] if len(narrow_idx) > 0 else -1

                empirical_mean = means[first_narrow_idx]
                distance = np.abs(empirical_mean - true_mean)
                run_distances[method_name] = distance

                # Count intervals before the first narrow index
                intervals_count[method_name] = first_narrow_idx + 1 if first_narrow_idx != -1 else len(ts)

            return run_distances, intervals_count

        results = Parallel(n_jobs=-1)(delayed(process_run)(run_idx) for run_idx in range(nb_runs))

        for run_result in results:
            if run_result is not None:
                run_distances, intervals_count = run_result
                for method_name, distance in run_distances.items():
                    distances_per_method[method_name].append(distance)
                for method_name, count in intervals_count.items():
                    avg_intervals_per_method[method_name].append(count)

        for method_name in distances_per_method:
            distances_per_method[method_name] = np.array(distances_per_method[method_name])
            avg_intervals_per_method[method_name] = np.mean(avg_intervals_per_method[method_name])  # Compute average

        run_dir, run_id, key_name = Experiments._create_experiment_run_dir(h, "coverage_experiment")
        method_entries = []
        for idx, method_name in enumerate(distances_per_method.keys()):
            filename = Experiments._save_method_npz(
                run_dir,
                idx,
                method_name,
                distances=distances_per_method[method_name],
                avg_intervals=np.array(avg_intervals_per_method[method_name], dtype=float),
            )
            method_entries.append({"name": method_name, "file": filename})

        quantiles = [1 - float(h['alpha'])]
        indicators = [0.02, 0.05, 0.1, 0.2]
        title = (f"Distribution of D(mu, early_stop_mean) - Target {h['stop_early_threshold']}, "
                 f"alpha={alpha}, {data_type}, N={N}, batch_size={batch_size}")
        Experiments._write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "experiment": "coverage_experiment",
                "run_id": run_id,
                "key_name": key_name,
                "h": h,
                "quantiles": quantiles,
                "indicators": indicators,
                "title": title,
                "xlabel": "Distance",
                "ylabel": "Frequency",
            },
        )
        Experiments._write_json(os.path.join(run_dir, "index.json"), {"methods": method_entries})

        # Plot the distribution of distances for all methods
        Plotter.plot_multiple_distributions_with_indicators(
            h,
            distances_per_method,
            quantiles=quantiles,
            indicators=indicators,
            avg_intervals_per_method=avg_intervals_per_method,  # Pass the averages
            title=title,
            xlabel="Distance",
            ylabel="Frequency"
        )

    @staticmethod
    def individual_run_experiment(h, N, alpha, batch_size, data_type, stop_early_threshold):
        run_dir, run_id, key_name = Experiments._create_experiment_run_dir(h, "individual_run_experiment")
        run_entries = []
        data, true_mean = DataDistributionChecker.get_data(data_type, N)  # or "fixed" for constant data
        if DataDistributionChecker.dim(h) < 5:
            Plotter.plot_distribution(h, data, true_mean, data_type=data_type)

        for i in range(h['num_intervals']):
            data, true_mean = DataDistributionChecker.get_data(data_type, N)  # or "fixed" for constant data

            # interval_types = ["T_CI", "T_CI_BONF", "HEDGE_CI", "HEDGE_CI_SUM"]
            interval_types = h['interval_types']
            results = []
            for interval_type in interval_types:
                method_func, method_name, method_color = IntervalFactory.create_instance(interval_type, h, data)
                ts, means, lowers, uppers, _, _ = method_func.run()
                results.append({
                    "name": method_name,
                    "color": method_color,
                    "data": (ts, means, lowers, uppers),
                    "alpha": alpha
                })

            run_subdir = os.path.join(run_dir, f"run_{i:03d}")
            os.makedirs(run_subdir, exist_ok=True)
            np.savez(os.path.join(run_subdir, "run_data.npz"), data=data, true_mean=true_mean)

            method_entries = []
            for method_idx, result in enumerate(results):
                ts, means, lowers, uppers = result["data"]
                filename = Experiments._save_method_npz(
                    run_subdir,
                    method_idx,
                    result["name"],
                    ts=np.asarray(ts),
                    means=np.asarray(means),
                    lowers=np.asarray(lowers),
                    uppers=np.asarray(uppers),
                    alpha=np.array(result["alpha"], dtype=float),
                    color=np.array(result["color"], dtype=str),
                )
                method_entries.append({"name": result["name"], "color": result["color"], "file": filename})

            Experiments._write_json(
                os.path.join(run_subdir, "index.json"),
                {"run_index": i, "methods": method_entries},
            )
            run_entries.append({"run_dir": f"run_{i:03d}", "run_index": i})

            # results[i]["data"] is a tuple (ts, means, lowers, uppers) where ts is the time points,
            for log_scale in [False, True]:
                Plotter.plot_conf_intervs(h, results,
                                          true_mean=true_mean,
                                          fill=False, batch_size=batch_size, data_type=data_type,
                                          stop_early_threshold=stop_early_threshold,
                                          log_x_axis=log_scale,
                                          alpha=alpha)

            if h['plot_interval_widths']:
                Plotter.plot_conf_interv_widths(h, results, batch_size=batch_size)

        Experiments._write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "experiment": "individual_run_experiment",
                "run_id": run_id,
                "key_name": key_name,
                "h": h,
                "num_intervals": h['num_intervals'],
            },
        )
        Experiments._write_json(os.path.join(run_dir, "index.json"), {"runs": run_entries})
    @staticmethod
    def put_bonf_first_if_in_list(lst):
        """If any element contains 'bonf' (case-insensitive), put it first."""
        bonf_items = [x for x in lst if "bonf" in x.lower()]
        if bonf_items:
            lst = bonf_items + [x for x in lst if x not in bonf_items]
        return lst

    @staticmethod
    def volume_vs_time_experiment(h: dict):
        D = DataDistributionChecker.dim(h)
        if D == 1:
            print("Dimensionality is 1, skipping dimensionality experiment.")
            return

        h = h.copy()
        D = h['volume_vs_time_experiment_D']
        time_points = h['volume_vs_time_experiment_time_points']
        h['calculate_volume'] = True
        h['N'] = max(time_points)

        # set the dim based on `h[volume_vs_time_experiment_D]`
        h['data_type'] = h['data_type'].rsplit('_', 1)[0] + f"_{h['volume_vs_time_experiment_D']}d"

        data, true_mean = DataDistributionChecker.get_data(h['data_type'], h['N'])  # or "fixed" for constant data
        interval_types = h['interval_types']  # e.g. ["T_CI", "T_CI_BONF", "HEDGE_CI", "HEDGE_CI_SUM"]

        interval_types = Experiments.put_bonf_first_if_in_list(interval_types)

        names = []
        colors = []
        volumes = []
        volume_stds = []

        run_dir, run_id, key_name = Experiments._create_experiment_run_dir(h, "volume_vs_time_experiment")
        method_entries = []
        for method_idx, interval_type in enumerate(interval_types):
            method, method_name, method_color = IntervalFactory.create_instance(interval_type, h, data)
            method: MultiDimensionalConfidenceSequence = method  # type: ignore
            # ts, _, _, _, volume_list, volume_std_list = method_func.run()

            ts = time_points
            volume_list = []
            volume_std_list = []


            for t in tqdm(time_points, desc=f"Computing volumes for {method_name}", unit="time point"):
                vol, vol_std = method.compute_volume_at(t=(t - 1))  # t-1 because t is 1-based index
                volume_list.append(vol)
                volume_std_list.append(vol_std)

            names.append(method_name)
            colors.append(method_color)
            volumes.append(volume_list)
            volume_stds.append(volume_std_list)
            print(f"\t{method_name}: final volume {volume_list[-1]}")

            filename = Experiments._save_method_npz(
                run_dir,
                method_idx,
                method_name,
                volumes=np.asarray(volume_list),
                volume_stds=np.asarray(volume_std_list),
                color=np.array(method_color, dtype=str),
            )
            method_entries.append({"name": method_name, "color": method_color, "file": filename})

        Experiments._write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "experiment": "volume_vs_time_experiment",
                "run_id": run_id,
                "key_name": key_name,
                "h": h,
                "N": h['N'],
                "D": D,
                "time_points": time_points,
            },
        )
        Experiments._write_json(os.path.join(run_dir, "index.json"), {"methods": method_entries})

        for y_log_scale in [False, True]:
            Plotter.plot_volumes_over_time(
                h, names, colors, volumes, volume_stds, h['N'], D, y_log_scale, time_points=time_points
            )

    @staticmethod
    def coverage_vs_time_experiment(h: dict):
        D = DataDistributionChecker.dim(h)
        if D == 1:
            print("Dimensionality is 1, skipping coverage-vs-time experiment.")
            return

        h = h.copy()
        D = h['coverage_vs_time_experiment_D']
        time_points = h['coverage_vs_time_experiment_time_points']
        repeats = h['coverage_vs_time_experiment_repeats']

        h['N'] = max(time_points)
        h['calculate_volume'] = False
        h['data_type'] = h['data_type'].rsplit('_', 1)[0] + f"_{D}d"

        interval_types = Experiments.put_bonf_first_if_in_list(h['interval_types'])
        num_methods = len(interval_types)
        num_times = len(time_points)

        coverage_counts = np.zeros((num_methods, num_times), dtype=float)
        names = [None] * num_methods
        colors = [None] * num_methods

        for run_idx in tqdm(range(repeats), desc="Coverage vs time runs"):
            data, true_mean = DataDistributionChecker.get_data(h['data_type'], h['N'])
            for method_idx, interval_type in enumerate(interval_types):
                method, method_name, method_color = IntervalFactory.create_instance(interval_type, h, data)
                if run_idx == 0:
                    names[method_idx] = method_name
                    colors[method_idx] = method_color

                if not hasattr(method, "is_member_at"):
                    raise ValueError(f"{method_name} does not implement is_member_at().")

                for t_idx, t in enumerate(time_points):
                    if method.is_member_at(true_mean, t - 1):
                        coverage_counts[method_idx, t_idx] += 1.0

        coverages = coverage_counts / float(repeats)

        run_dir, run_id, key_name = Experiments._create_experiment_run_dir(h, "coverage_vs_time_experiment")
        method_entries = []
        for method_idx, method_name in enumerate(names):
            filename = Experiments._save_method_npz(
                run_dir,
                method_idx,
                method_name,
                coverage=np.asarray(coverages[method_idx]),
                time_points=np.asarray(time_points, dtype=int),
                color=np.array(colors[method_idx], dtype=str),
            )
            method_entries.append({"name": method_name, "color": colors[method_idx], "file": filename})

        Experiments._write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "experiment": "coverage_vs_time_experiment",
                "run_id": run_id,
                "key_name": key_name,
                "h": h,
                "N": h['N'],
                "D": D,
                "time_points": time_points,
                "repeats": repeats,
            },
        )
        Experiments._write_json(os.path.join(run_dir, "index.json"), {"methods": method_entries})

        Plotter.plot_coverage_over_time(
            h, names, colors, coverages, h['N'], D, time_points=time_points, alpha=h['alpha']
        )

    @staticmethod
    def volume_vs_dim_experiment(h: dict, baseline_idx: int = 0, skip_stds: bool = False):
        """
        Plots V(d) and ratio V(d)/V_baseline(d) across dimensions.

        Args:
            h: config dict (modified copy inside).
            baseline_idx: index of method used as denominator for ratios (default first method).
            skip_stds: if True, skip computing/plotting std bands for ratios (and ignore final_stds).
        Returns:
            names, colors, Ds, final_volumes, final_stds, ratio_matrix, ratio_stds
        """
        D_check = DataDistributionChecker.dim(h)
        if D_check == 1:
            print("Dimensionality is 1, skipping dimensionality experiment.")
            return None

        D_MAX = h['volume_vs_dim_experiment_max_dim']  # inclusive
        N = h['volume_vs_dim_experiment_N']

        # local copy so we don't mutate caller's dict
        h = h.copy()
        h['calculate_volume'] = True
        h['N'] = N

        num_methods = len(h['interval_types'])
        Ds = range(2, D_MAX + 1)
        final_volumes = np.zeros((num_methods, len(Ds)))
        final_stds = np.zeros_like(final_volumes)

        names = [None] * num_methods
        colors = [None] * num_methods
        interval_types = h['interval_types']
        interval_types = Experiments.put_bonf_first_if_in_list(interval_types)

        run_dir, run_id, key_name = Experiments._create_experiment_run_dir(h, "volume_vs_dim_experiment")
        method_entries = []

        for idx_d, d in enumerate(Ds):
            print(f"Dimensionality experiment for d={d}")
            # assume h['data_type'] = XXXX_{d}d. Overwrite the dimension part. e.g., 'highly_correlated_2d' -> '..._3d'
            prev_dim_part = h['data_type'].split('_')[-1]  # e.g., '2d'
            assert prev_dim_part.endswith('d'), "data_type should end with dimension part like '2d'"

            # replace dimension part
            h['data_type'] = h['data_type'][:-len(prev_dim_part)] + f"{d}d"  # e.g., 'highly_correlated_' + '2d'

            data, true_mean = DataDistributionChecker.get_data(h['data_type'], h['N'])

            for method_idx, interval_type in enumerate(interval_types):
                method, method_name, method_color = IntervalFactory.create_instance(interval_type, h, data)
                method: MultiDimensionalConfidenceSequence = method

                if idx_d == 0:
                    names[method_idx] = method_name
                    colors[method_idx] = method_color

                t_last = h['N'] // h['batch_size'] - 1
                vol, std = method.compute_volume_at(t=t_last)

                final_volumes[method_idx, idx_d] = vol
                final_stds[method_idx, idx_d] = std

                print(f"\t{method_name}: final volume {vol:.6g} ± {std:.6g}")

        for method_idx, method_name in enumerate(names):
            filename = Experiments._save_method_npz(
                run_dir,
                method_idx,
                method_name,
                Ds=np.asarray(list(Ds)),
                final_volumes=final_volumes[method_idx],
                final_stds=final_stds[method_idx],
                color=np.array(colors[method_idx], dtype=str),
            )
            method_entries.append({"name": method_name, "color": colors[method_idx], "file": filename})

        Experiments._write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "experiment": "volume_vs_dim_experiment",
                "run_id": run_id,
                "key_name": key_name,
                "h": h,
                "baseline_idx": baseline_idx,
                "skip_stds": skip_stds,
                "D_MAX": D_MAX,
                "N": N,
            },
        )
        Experiments._write_json(os.path.join(run_dir, "index.json"), {"methods": method_entries})

        # --- Plot absolute volumes (log and linear) ---
        Plotter.plot_volumes_over_dims(h, names, colors, final_volumes, final_stds, Ds, y_log_scale=True)
        Plotter.plot_volumes_over_dims(h, names, colors, final_volumes, final_stds, Ds, y_log_scale=False)

        # --- Compute ratios relative to baseline ---
        M, K = final_volumes.shape
        if baseline_idx < 0 or baseline_idx >= M:
            raise IndexError("baseline_idx out of range of methods")

        # protect against zeros in baseline
        eps = 1e-12
        baseline = final_volumes[baseline_idx, :].astype(float)
        baseline_safe = np.where(baseline <= 0, eps, baseline)

        ratio_matrix = final_volumes / baseline_safe[np.newaxis, :]  # broadcast, shape (M, K)

        # --- compute ratio stds (delta-method) unless skip_stds True ---
        if skip_stds:
            ratio_stds = np.zeros_like(ratio_matrix)
        else:
            A = final_volumes
            sA = final_stds
            B = baseline_safe[np.newaxis, :]
            sB = final_stds[baseline_idx, :][np.newaxis, :]

            # delta-method variance approx: Var(A/B) ≈ (sA^2 / B^2) + (A^2 * sB^2 / B^4)
            ratio_var = (sA ** 2) / (B ** 2) + (A ** 2) * (sB ** 2) / (B ** 4)
            # numerical safety: force non-negative
            ratio_var = np.maximum(ratio_var, 0.0)
            ratio_stds = np.sqrt(ratio_var)

            # If baseline had eps replacements, scale down the uncertainty where B was replaced
            # (optional heuristic — we keep it simple and let the large uncertainty show if baseline was zero)

        # --- Plot ratios (linear & possibly log) ---
        # choose ylabel including baseline name
        baseline_name = names[baseline_idx] if names[baseline_idx] is not None else f"method_{baseline_idx}"
        ylabel = f"Volume / {baseline_name}"

        # Plot linear (ratios often >0 and easier to read linearly)
        Plotter.plot_volume_ratios_over_dims(h, names, colors, ratio_matrix, ratio_stds, Ds, y_log_scale=False,
                                             ylabel=ylabel)

        # Optionally plot log scale if user wants (comment/uncomment as needed)
        # Plotter.plot_volume_ratios_over_dims(h, names, colors, ratio_matrix, ratio_stds, Ds, y_log_scale=True,
        #                                     ylabel=ylabel)

        return names, colors, Ds, final_volumes, final_stds, ratio_matrix, ratio_stds

    @staticmethod
    def volume_vs_p_experiment(h: dict, baseline_idx: int = 0, skip_stds: bool = False):
        """
        Plots volume vs parabola power and ratio to baseline method.
        """
        D_check = DataDistributionChecker.dim(h)
        if D_check == 1:
            print("Dimensionality is 1, skipping volume-vs-p experiment.")
            return None

        Ps = list(h['volume_vs_p_experiment_Ps'])
        if len(Ps) == 0:
            raise ValueError("No parabola power values provided")

        N = h['volume_vs_p_experiment_N']

        h = h.copy()
        h['calculate_volume'] = True
        h['N'] = N
        h['parabola_adaptive'] = False

        interval_types = Experiments.put_bonf_first_if_in_list(h['interval_types'])
        num_methods = len(interval_types)

        final_volumes = np.zeros((num_methods, len(Ps)))
        final_stds = np.zeros_like(final_volumes)
        names = [None] * num_methods
        colors = [None] * num_methods

        data, _ = DataDistributionChecker.get_data(h['data_type'], h['N'])
        t_last = h['N'] // h['batch_size'] - 1
        ref_lines = Experiments._reference_volume_lines(h, data, t_last)

        run_dir, run_id, key_name = Experiments._create_experiment_run_dir(h, "volume_vs_p_experiment")
        method_entries = []

        for idx_p, p in enumerate(Ps):
            try:
                p_int = int(p)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid parabola power value: {p}") from exc

            print(f"Parabola power experiment for p={p_int}")
            h['parabola_power'] = p_int

            for method_idx, interval_type in enumerate(interval_types):
                method, method_name, method_color = IntervalFactory.create_instance(interval_type, h, data)
                method: MultiDimensionalConfidenceSequence = method

                if idx_p == 0:
                    names[method_idx] = method_name
                    colors[method_idx] = method_color

                vol, std = method.compute_volume_at(t=t_last)

                final_volumes[method_idx, idx_p] = vol
                final_stds[method_idx, idx_p] = std

                print(f"\t{method_name}: final volume {vol:.6g} +/- {std:.6g}")

        for method_idx, method_name in enumerate(names):
            filename = Experiments._save_method_npz(
                run_dir,
                method_idx,
                method_name,
                Ps=np.asarray(Ps),
                final_volumes=final_volumes[method_idx],
                final_stds=final_stds[method_idx],
                color=np.array(colors[method_idx], dtype=str),
            )
            method_entries.append({"name": method_name, "color": colors[method_idx], "file": filename})

        Experiments._write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "experiment": "volume_vs_p_experiment",
                "run_id": run_id,
                "key_name": key_name,
                "h": h,
                "baseline_idx": baseline_idx,
                "skip_stds": skip_stds,
                "Ps": list(Ps),
                "N": N,
                "ref_lines": ref_lines,
            },
        )
        Experiments._write_json(os.path.join(run_dir, "index.json"), {"methods": method_entries})

        xlabel = "Parabola power (p)"
        Plotter.plot_volumes_over_dims(h, names, colors, final_volumes, final_stds, Ps, y_log_scale=True,
                                       xlabel=xlabel, ref_lines=ref_lines)
        Plotter.plot_volumes_over_dims(h, names, colors, final_volumes, final_stds, Ps, y_log_scale=False,
                                       xlabel=xlabel, ref_lines=ref_lines)

        M, _ = final_volumes.shape
        if baseline_idx < 0 or baseline_idx >= M:
            raise IndexError("baseline_idx out of range of methods")

        eps = 1e-12
        baseline = final_volumes[baseline_idx, :].astype(float)
        baseline_safe = np.where(baseline <= 0, eps, baseline)
        ratio_matrix = final_volumes / baseline_safe[np.newaxis, :]

        if skip_stds:
            ratio_stds = np.zeros_like(ratio_matrix)
        else:
            A = final_volumes
            sA = final_stds
            B = baseline_safe[np.newaxis, :]
            sB = final_stds[baseline_idx, :][np.newaxis, :]
            ratio_var = (sA ** 2) / (B ** 2) + (A ** 2) * (sB ** 2) / (B ** 4)
            ratio_var = np.maximum(ratio_var, 0.0)
            ratio_stds = np.sqrt(ratio_var)

        baseline_name = names[baseline_idx] if names[baseline_idx] is not None else f"method_{baseline_idx}"
        ylabel = f"Volume / {baseline_name}"
        Plotter.plot_volume_ratios_over_dims(h, names, colors, ratio_matrix, ratio_stds, Ps, y_log_scale=False,
                                             ylabel=ylabel, xlabel=xlabel)

        return names, colors, Ps, final_volumes, final_stds, ratio_matrix, ratio_stds

    @staticmethod
    def volume_vs_C_experiment(h: dict, baseline_idx: int = 0, skip_stds: bool = False):
        """
        Plots volume vs parabola touchpoint C and ratio to baseline method.
        """
        D_check = DataDistributionChecker.dim(h)
        if D_check == 1:
            print("Dimensionality is 1, skipping volume-vs-C experiment.")
            return None

        # Cs = list(h['volume_vs_C_experiment_Ps'])
        #  #div = (D ** (1.0 / power)) -> #touchpointC = C / div
        power = h['volume_vs_C_experiment_fixed_parabola_pow']
        C_touchpnt_marker = (1/h['alpha']) / (D_check ** (1.0 / power))

        min_C = (1/h['alpha']) / (D_check * (D_check ** (1.0 / power)))
        max_C = ((1.0 / h['alpha']) * D_check)
        Cs = np.linspace(min_C, max_C, 100)

        if len(Cs) == 0:
            print("No C values provided; skipping volume-vs-C experiment.")
            return None

        # if h.get('parabola_power', None) is None:
        #     raise ValueError("parabola_power must be set for volume-vs-C experiment.")

        N = h['volume_vs_C_experiment_N']

        h = h.copy()
        h['calculate_volume'] = True
        h['N'] = N
        h['parabola_adaptive'] = False
        h['parabola_power'] = power

        interval_types = Experiments.put_bonf_first_if_in_list(h['interval_types'])
        num_methods = len(interval_types)

        final_volumes = np.zeros((num_methods, len(Cs)))
        final_stds = np.zeros_like(final_volumes)
        names = [None] * num_methods
        colors = [None] * num_methods

        data, _ = DataDistributionChecker.get_data(h['data_type'], h['N'])
        t_last = h['N'] // h['batch_size'] - 1
        ref_lines = Experiments._reference_volume_lines(h, data, t_last)

        run_dir, run_id, key_name = Experiments._create_experiment_run_dir(h, "volume_vs_C_experiment")
        method_entries = []

        for idx_c, C in enumerate(Cs):
            try:
                c_val = float(C)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid C value: {C}") from exc

            print(f"Touchpoint C experiment for C={c_val}")
            h['parabola_C_touchpoint'] = c_val

            for method_idx, interval_type in enumerate(interval_types):
                method, method_name, method_color = IntervalFactory.create_instance(interval_type, h, data)
                method: MultiDimensionalConfidenceSequence = method

                if idx_c == 0:
                    names[method_idx] = method_name
                    colors[method_idx] = method_color

                vol, std = method.compute_volume_at(t=t_last)

                final_volumes[method_idx, idx_c] = vol
                final_stds[method_idx, idx_c] = std

                print(f"\t{method_name}: final volume {vol:.6g} +/- {std:.6g}")

        for method_idx, method_name in enumerate(names):
            filename = Experiments._save_method_npz(
                run_dir,
                method_idx,
                method_name,
                Cs=np.asarray(Cs, dtype=float),
                final_volumes=final_volumes[method_idx],
                final_stds=final_stds[method_idx],
                color=np.array(colors[method_idx], dtype=str),
            )
            method_entries.append({"name": method_name, "color": colors[method_idx], "file": filename})

        xlabel = "Touchpoint C"
        x_marker = {"x": C_touchpnt_marker, "label": "C_touchpnt_marker", "color": "black", "linestyle": ":"}

        Experiments._write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "experiment": "volume_vs_C_experiment",
                "run_id": run_id,
                "key_name": key_name,
                "h": h,
                "baseline_idx": baseline_idx,
                "skip_stds": skip_stds,
                "Cs": list(Cs),
                "N": N,
                "ref_lines": ref_lines,
                "x_marker": x_marker,
            },
        )
        Experiments._write_json(os.path.join(run_dir, "index.json"), {"methods": method_entries})


        Plotter.plot_volumes_over_dims(h, names, colors, final_volumes, final_stds, Cs, y_log_scale=True,
                                       xlabel=xlabel, ref_lines=ref_lines, x_marker=x_marker)
        Plotter.plot_volumes_over_dims(h, names, colors, final_volumes, final_stds, Cs, y_log_scale=False,
                                       xlabel=xlabel, ref_lines=ref_lines, x_marker=x_marker)

        M, _ = final_volumes.shape
        if baseline_idx < 0 or baseline_idx >= M:
            raise IndexError("baseline_idx out of range of methods")

        eps = 1e-12
        baseline = final_volumes[baseline_idx, :].astype(float)
        baseline_safe = np.where(baseline <= 0, eps, baseline)
        ratio_matrix = final_volumes / baseline_safe[np.newaxis, :]

        if skip_stds:
            ratio_stds = np.zeros_like(ratio_matrix)
        else:
            A = final_volumes
            sA = final_stds
            B = baseline_safe[np.newaxis, :]
            sB = final_stds[baseline_idx, :][np.newaxis, :]
            ratio_var = (sA ** 2) / (B ** 2) + (A ** 2) * (sB ** 2) / (B ** 4)
            ratio_var = np.maximum(ratio_var, 0.0)
            ratio_stds = np.sqrt(ratio_var)

        baseline_name = names[baseline_idx] if names[baseline_idx] is not None else f"method_{baseline_idx}"
        ylabel = f"Volume / {baseline_name}"
        Plotter.plot_volume_ratios_over_dims(h, names, colors, ratio_matrix, ratio_stds, Cs, y_log_scale=False,
                                             ylabel=ylabel, xlabel=xlabel)

        return names, colors, Cs, final_volumes, final_stds, ratio_matrix, ratio_stds

    @staticmethod
    def volume_vs_parabola_gr_sz_experiment(h: dict, baseline_idx: int = 0, skip_stds: bool = False):
        """
        Plots volume vs parabola grid size and ratio to baseline method.
        """
        D_check = DataDistributionChecker.dim(h)
        if D_check == 1:
            print("Dimensionality is 1, skipping volume-vs-grid-size experiment.")
            return None

        grid_sizes = list(h['volume_vs_parabola_gr_sz_experiment_grid_sizes'])
        if len(grid_sizes) == 0:
            print("No grid sizes provided; skipping volume-vs-grid-size experiment.")
            return None

        N = h['volume_vs_parabola_gr_sz_experiment_N']

        h = h.copy()
        h['calculate_volume'] = True
        h['N'] = N

        interval_types = Experiments.put_bonf_first_if_in_list(h['interval_types'])
        num_methods = len(interval_types)

        final_volumes = np.zeros((num_methods, len(grid_sizes)))
        final_stds = np.zeros_like(final_volumes)
        names = [None] * num_methods
        colors = [None] * num_methods

        data, _ = DataDistributionChecker.get_data(h['data_type'], h['N'])
        t_last = h['N'] // h['batch_size'] - 1
        ref_lines = Experiments._reference_volume_lines(h, data, t_last)

        run_dir, run_id, key_name = Experiments._create_experiment_run_dir(h, "volume_vs_parabola_gr_sz_experiment")
        method_entries = []

        for idx_g, grid_size in enumerate(grid_sizes):
            try:
                grid_size_int = int(grid_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid grid size value: {grid_size}") from exc

            print(f"Parabola grid-size experiment for grid_size={grid_size_int}")
            h['parabola_grid_size'] = grid_size_int

            for method_idx, interval_type in enumerate(interval_types):
                method, method_name, method_color = IntervalFactory.create_instance(interval_type, h, data)
                method: MultiDimensionalConfidenceSequence = method

                if idx_g == 0:
                    names[method_idx] = method_name
                    colors[method_idx] = method_color

                vol, std = method.compute_volume_at(t=t_last)

                final_volumes[method_idx, idx_g] = vol
                final_stds[method_idx, idx_g] = std

                print(f"\t{method_name}: final volume {vol:.6g} +/- {std:.6g}")

        for method_idx, method_name in enumerate(names):
            filename = Experiments._save_method_npz(
                run_dir,
                method_idx,
                method_name,
                grid_sizes=np.asarray(grid_sizes, dtype=int),
                final_volumes=final_volumes[method_idx],
                final_stds=final_stds[method_idx],
                color=np.array(colors[method_idx], dtype=str),
            )
            method_entries.append({"name": method_name, "color": colors[method_idx], "file": filename})

        Experiments._write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "experiment": "volume_vs_parabola_gr_sz_experiment",
                "run_id": run_id,
                "key_name": key_name,
                "h": h,
                "baseline_idx": baseline_idx,
                "skip_stds": skip_stds,
                "grid_sizes": list(grid_sizes),
                "N": N,
                "ref_lines": ref_lines,
            },
        )
        Experiments._write_json(os.path.join(run_dir, "index.json"), {"methods": method_entries})

        xlabel = "Parabola grid size"
        Plotter.plot_volumes_over_dims(h, names, colors, final_volumes, final_stds, grid_sizes, y_log_scale=True,
                                       xlabel=xlabel, ref_lines=ref_lines)
        Plotter.plot_volumes_over_dims(h, names, colors, final_volumes, final_stds, grid_sizes, y_log_scale=False,
                                       xlabel=xlabel, ref_lines=ref_lines)

        M, _ = final_volumes.shape
        if baseline_idx < 0 or baseline_idx >= M:
            raise IndexError("baseline_idx out of range of methods")

        eps = 1e-12
        baseline = final_volumes[baseline_idx, :].astype(float)
        baseline_safe = np.where(baseline <= 0, eps, baseline)
        ratio_matrix = final_volumes / baseline_safe[np.newaxis, :]

        if skip_stds:
            ratio_stds = np.zeros_like(ratio_matrix)
        else:
            A = final_volumes
            sA = final_stds
            B = baseline_safe[np.newaxis, :]
            sB = final_stds[baseline_idx, :][np.newaxis, :]
            ratio_var = (sA ** 2) / (B ** 2) + (A ** 2) * (sB ** 2) / (B ** 4)
            ratio_var = np.maximum(ratio_var, 0.0)
            ratio_stds = np.sqrt(ratio_var)

        baseline_name = names[baseline_idx] if names[baseline_idx] is not None else f"method_{baseline_idx}"
        ylabel = f"Volume / {baseline_name}"
        Plotter.plot_volume_ratios_over_dims(h, names, colors, ratio_matrix, ratio_stds, grid_sizes,
                                             y_log_scale=False, ylabel=ylabel, xlabel=xlabel)

        return names, colors, grid_sizes, final_volumes, final_stds, ratio_matrix, ratio_stds


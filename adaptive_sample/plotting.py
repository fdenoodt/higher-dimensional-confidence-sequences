from typing import Tuple, Optional, Union, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import gaussian_kde

from shared.decorators import Key
from adaptive_sample.shared_code import DataDistributionChecker, DistributionType
import os

try:
    import tikzplotlib
except ImportError:
    print("tikzplotlib not installed. Install it using `pip install tikzplotlib`.")


class Plotter:
    @staticmethod
    def _results_single_dim(true_mean, results, d):
        # results is list of methods, and res[method] is dict{"data": [ts, means, lowers, uppers], "color": color, "name": name}
        # define results[method] as {"data": [ts, means[d], lowers[d], uppers[d]], "color": color, "name": name}
        true_mean_d = true_mean[d]
        try:
            results_d = [{"data": [result["data"][0],  # time steps
                                   result["data"][1][:, d],  # means, take all time steps but only one dimension
                                   result["data"][2][:, d],  # lower confidence intervals
                                   result["data"][3][:, d]],  # upper confidence intervals
                          "color": result["color"],
                          "name": result["name"]} for result in results]
        except IndexError:
            raise IndexError(f"Dimension {d} is out of bounds for true_mean with shape {np.shape(true_mean)}. "
                             f"Available dimensions: {list(range(len(true_mean)))}")
        except Exception:
            raise ValueError(f"Unexpected error while processing results for dimension {d}. ")
        return results_d, true_mean_d

    @staticmethod
    def _maybe_show(h: dict, fig: Optional[plt.Figure] = None) -> None:
        if h.get('plot_experiments', True):
            plt.show()
            return
        if fig is not None:
            plt.close(fig)
        else:
            plt.close('all')

    @staticmethod
    def plot_conf_intervs(h, results, true_mean: float, fill, batch_size, data_type, stop_early_threshold: float,
                          alpha: float, log_x_axis: bool = False) -> Tuple[plt.Figure, str]:
        if isinstance(true_mean, (int, float)):
            fig, ax = plt.subplots(figsize=(10, 6))
            Plotter._plot_conf_intervs_scalar_data(results, true_mean, fill, batch_size, data_type,
                                                   stop_early_threshold, alpha, log_x_axis, ax=ax)
            fig.suptitle(f'Confidence Intervals for {data_type} Data (batch size={batch_size}, alpha={alpha})',
                         fontsize=16)
            fig.tight_layout()
            path = Plotter._save_plot(h, f"conf_intervs_log_x_axis={log_x_axis}", "")
            Plotter._maybe_show(h, fig)
            return fig, path
        elif isinstance(true_mean, (list, np.ndarray)):
            if len(true_mean) == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                Plotter._plot_conf_intervs_scalar_data(results, true_mean[0], fill, batch_size, data_type,
                                                       stop_early_threshold, alpha, log_x_axis, ax=ax)
                fig.suptitle(f'Confidence Intervals for {data_type} Data (batch size={batch_size}, alpha={alpha})',
                             fontsize=16)
                fig.tight_layout()
                path = Plotter._save_plot(h, f"conf_intervs", "")
                Plotter._maybe_show(h, fig)
                return fig, path
            else:
                # Create a grid layout for multiple dimensions
                n_dims = len(true_mean)
                n_cols = 2
                n_rows = (n_dims + n_cols - 1) // n_cols  # Calculate rows based on number of dimensions
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
                axes = axes.flatten()  # Flatten the grid for easy indexing

                for d in range(n_dims):
                    results_d, true_mean_d = Plotter._results_single_dim(true_mean, results, d)
                    Plotter._plot_conf_intervs_scalar_data(results_d, true_mean_d, fill, batch_size, data_type,
                                                           stop_early_threshold, alpha, log_x_axis, ax=axes[d])
                    axes[d].set_title(f'Dimension {d + 1}', fontsize=12)

                # Remove unused subplots
                for i in range(n_dims, len(axes)):
                    fig.delaxes(axes[i])

                # Add a distinct title for the entire figure
                fig.suptitle(f'Confidence Intervals for {data_type} Data (batch size={batch_size}, alpha={alpha})',
                             fontsize=16)
                fig.tight_layout()
                # save the plot
                path = Plotter._save_plot(h, f"conf_intervs", "")
                Plotter._maybe_show(h, fig)
                return fig, path

        else:
            raise ValueError(
                f"Unexpected type for true_mean: {type(true_mean)}. Expected int, float, list, or np.ndarray.")

    @staticmethod
    def _plot_conf_intervs_scalar_data(results, true_mean: float, fill, batch_size, data_type,
                                       stop_early_threshold: float, alpha: float, log_x_axis: bool = False, ax=None):
        # If no Axes object is provided, create one
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the true mean
        if true_mean is not None:
            ax.axhline(y=true_mean, color='red', label='True Mean')

        for result in results:
            ts, means, lowers, uppers = result["data"]
            widths = np.array(uppers) - np.array(lowers)

            # Find the first time the width is smaller or equal to stop_early_threshold
            narrow_idx = np.where(widths <= stop_early_threshold)[0]
            if narrow_idx.size > 0:
                first_narrow_t = ts[narrow_idx[0]]
                first_narrow_mean = means[narrow_idx[0]]

                # Put the point both on the upper and lower CI
                ax.scatter(first_narrow_t, lowers[narrow_idx[0]], color=result["color"], marker='o', zorder=5)
                ax.scatter(first_narrow_t, uppers[narrow_idx[0]], color=result["color"], marker='o', zorder=5)
                # Put text on the upper CI line and under the lower CI line
                ax.text(first_narrow_t, uppers[narrow_idx[0]], f'{first_narrow_t}', color=result["color"],
                        fontsize=9, ha='center', va='bottom')
                ax.text(first_narrow_t, lowers[narrow_idx[0]] - 0.015, f'{first_narrow_t}', color=result["color"],
                        fontsize=9, ha='center', va='top')  # The -0.015 is to lower the text a bit

            if fill:
                ax.fill_between(ts, lowers, uppers, color=result["color"], alpha=result["alpha"], label=result["name"])
            else:
                ax.plot(ts, lowers, color=result["color"], alpha=0.5, label=f'{result["name"]}')
                ax.plot(ts, uppers, color=result["color"], alpha=0.5)

        ts, means, lowers, uppers = results[0]["data"]
        ax.plot(ts, means, color='black', alpha=0.7, label='Empirical Mean')  # Plot empirical mean

        ax.set_xlabel('Number of Samples $t$')
        ax.set_ylabel('Mean ± CI')
        if data_type == "uniform" or data_type.startswith("fixed") or data_type.startswith(
                "beta_") or data_type.startswith("dirichlet_"):
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.1, 0.1))  # Grid lines on y axis every 0.1
        elif data_type == "gaussian":
            ax.set_ylim(-3, 3)

        if log_x_axis:
            ax.set_xscale('log')

        ax.grid(True)
        ax.legend()
        return ax

    @staticmethod
    def plot_conf_interv_widths(h, results, batch_size) -> Tuple[plt.Figure, str]:
        ts, means, lowers, uppers = results[0]["data"]  # t means
        num_timesteps = len(ts)

        # Case 1: Scalar or 1D
        if isinstance(means, (int, float)) or len(np.shape(means)) == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            Plotter._plot_conf_interv_widths_scalar_data(results, batch_size, ax=ax)
            fig.suptitle(f'Widths of Confidence Intervals (batch size={batch_size})', fontsize=16)
            fig.tight_layout()
            path = Plotter._save_plot(h, f"conf_interv_widths", "")
            Plotter._maybe_show(h, fig)
            return fig, path

        # Case 2: Multidimensional
        elif isinstance(means, (list, np.ndarray)):
            n_dims = means.shape[1] if isinstance(means, np.ndarray) else len(means[0])
            n_cols = 2
            n_rows = (n_dims + n_cols - 1) // n_cols  # Calculate rows based on number of dimensions
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
            axes = axes.flatten()

            for d in range(n_dims):
                results_d, _ = Plotter._results_single_dim(np.zeros(n_dims), results, d)
                Plotter._plot_conf_interv_widths_scalar_data(results_d, batch_size, ax=axes[d])
                axes[d].set_title(f'Dimension {d + 1}', fontsize=12)

            # Remove unused subplots
            for i in range(n_dims, len(axes)):
                fig.delaxes(axes[i])

            fig.suptitle(f'Widths of Confidence Intervals (batch size={batch_size})', fontsize=16)
            fig.tight_layout()
            path = Plotter._save_plot(h, f"conf_interv_widths", "")
            Plotter._maybe_show(h, fig)
            return fig, path

        else:
            raise ValueError(f"Unexpected type for means: {type(means)}. Expected int, float, list, or np.ndarray.")

    @staticmethod
    def _plot_conf_interv_widths_scalar_data(results, batch_size, ax=None):
        # Second plot: CI widths
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for result in results:
            ts, _, lowers, uppers = result["data"]
            widths = np.array(uppers) - np.array(lowers)
            # ax.plot(ts, widths, label=f'{result["name"]} Width', color=result["color"])
            ax.scatter(ts, widths, label=f'{result["name"]} Width', color=result["color"], alpha=0.7)

        ax.set_xlabel('Number of Samples $t$')
        ax.set_ylabel('CI Width (log scale)')
        ax.set_yscale('log')
        ax.grid(True, which="both", linewidth=0.5)
        # ax.legend(loc='upper right')

        return ax

    @staticmethod
    def plot_l1_errors(h, error_percentages_per_case, error_percentages_global, error_percentages_early_stopping,
                       target_percentage: float, N, batch_size, data_type) -> Tuple[plt.Figure, str]:
        """
        Plot the average L1 errors for individual confidence intervals and global sequences.
        """
        assert target_percentage > 0 and target_percentage <= 1, "Target percentage must be between 0 and 100."

        fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes explicitly
        methods = list(error_percentages_global.keys())
        global_percentages = list(error_percentages_global.values())
        per_case_percentages = [error_percentages_per_case[method] for method in methods]

        x = np.arange(len(methods))  # the label locations
        width = 0.25  # the width of the bars

        ax.bar(x - width, per_case_percentages, width, label="Per Case", color="blue")
        ax.bar(x, global_percentages, width, label="Global", color="orange")
        early_stopping_percentages = [error_percentages_early_stopping[method] for method in methods]
        ax.bar(x + width, early_stopping_percentages, width, label="Early Stopping", color="green")

        # Add value labels on top of the bars
        for i, v in enumerate(per_case_percentages):
            ax.text(i - width, v + 2, f"{v:.1f}%", ha='center', va='bottom')
        for i, v in enumerate(global_percentages):
            ax.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom')
        for i, v in enumerate(early_stopping_percentages):
            ax.text(i + width, v + 2, f"{v:.1f}%", ha='center', va='bottom')

        ax.set_xlabel("Methods")
        ax.set_ylabel("Error Percentage (%)")
        ax.set_title(
            f"Average L1 Errors (Target: {target_percentage * 100}%, N={N}, batch_size={batch_size}, Data Type: {data_type})")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        path = Plotter._save_plot(h, f"l1_errors", "")
        Plotter._maybe_show(h, fig)
        return fig, path

    @staticmethod
    def _plot_dirichlet_distribution(h, data, true_mean, alphas) -> Tuple[plt.Figure, str]:
        """
        Visualize samples from a Dirichlet distribution.

        Parameters:
        - data: np.ndarray, shape (n_samples, n_dimensions)
        - true_mean: np.ndarray, shape (n_dimensions,)
        - alphas: np.ndarray, shape (n_dimensions,)
        """
        dim = data.shape[1]
        n_samples = data.shape[0]

        if dim == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(data.flatten(), bins=30, color='skyblue', edgecolor='black')
            ax.axvline(true_mean[0], color='red', label='True Mean')
            ax.set_title(f"1D Dirichlet Samples (α={alphas[0]:.2f})")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.legend()
            path = Plotter._save_plot(h, f"samples", "")
            Plotter._maybe_show(h, fig)
            return fig, path

        elif dim == 2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(data[:, 0], data[:, 1], alpha=0.6, label='Samples')
            ax.scatter(*true_mean, color='red', label='True Mean', s=100)
            ax.set_title(f"2D Dirichlet Samples (α={alphas})")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.legend()
            ax.axis("equal")
            # Set limits for better visualization
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True)
            path = Plotter._save_plot(h, f"samples", "")
            Plotter._maybe_show(h, fig)
            return fig, path

        elif dim == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*true_mean, color='red', s=100, label='True Mean')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.3)

            # Draw the triplex triangle
            triangle_vertices = np.array([
                [0, 0, 1],  # Vertex 1
                [0, 1, 0],  # Vertex 2
                [1, 0, 0]  # Vertex 3
            ])
            triangle = Poly3DCollection([triangle_vertices], alpha=0.1, color='gray', edgecolor='k')
            ax.add_collection3d(triangle)

            ax.set_title(f"3D Dirichlet Samples (α={alphas})")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("x3")
            ax.view_init(elev=20, azim=30)  # Rotate the figure (elevation and azimuth angles)

            # set limits for better visualization
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)

            ax.legend()
            path = Plotter._save_plot(h, f"samples", "")
            Plotter._maybe_show(h, fig)
            return fig, path

        else:
            import pandas as pd
            fig = plt.figure(figsize=(10, 8))
            df = pd.DataFrame(data, columns=[f"x{i}" for i in range(dim)])
            sns.pairplot(df, plot_kws={'alpha': 0.4, 's': 15})
            plt.suptitle("Dirichlet Samples (Pairplot of 2D projections)", y=1.02)
            path = Plotter._save_plot(h, f"samples", "")
            Plotter._maybe_show(h, fig)
            return fig, path

    @staticmethod
    def plot_distribution(h, data, true_mean, data_type: str):
        """
        Plot the distribution of the data with the true mean.
        """
        distribution_type, args, _ = DataDistributionChecker.check_distribution(data_type)
        if distribution_type == DistributionType.DIRICHLET:
            resp = Plotter._plot_dirichlet_distribution(h, data, true_mean, args)
            return resp
        # for higher dim, take one dimension
        elif DataDistributionChecker.dim({'data_type': data_type}) > 1:
            data = np.array(data)[:, 0]  # For fixed_nd, we take the first dimension for plotting
            true_mean = true_mean[0]

        # Define the figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram for other data types
        ax.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Data Distribution')

        if data_type in ["uniform", "fixed"] or data_type.startswith("beta_"):
            ax.set_xlim(0, 1)

        # Add the true mean line
        ax.axvline(true_mean, color='red', linewidth=2, label='True Mean')

        # Set labels, title, and legend
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of Data (True Mean: {true_mean}, Data Type: {data_type})')
        ax.legend()
        ax.grid(True)

        # Adjust layout
        fig.tight_layout()

        # Save the plot
        path = Plotter._save_plot(h, f"samples", "")

        # Show the plot
        Plotter._maybe_show(h, fig)

        return fig, path

    @staticmethod
    def plot_2d_function(h, Z, grid_pts, target: float = 20):
        # clip data to [-\inf, 200] for visualization
        Z = np.clip(Z, -np.inf, 200)

        # Extract unique x and y values from grid_pts
        x_values = sorted(set(pt[0] for pt in grid_pts))
        y_values = sorted(set(pt[1] for pt in grid_pts))

        # Create the meshgrid using the actual x and y values
        X, Y = np.meshgrid(x_values, y_values)

        fig = plt.figure(figsize=(12, 6))

        # First graph: 3D surface with contour
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis')
        contour_levels = [target]  # Define the level to highlight
        ax1.contour(X, Y, Z, levels=contour_levels, colors='red', linewidths=2.5, offset=20)
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')
        ax1.set_zlabel('Z-axis')
        ax1.view_init(elev=30, azim=210)

        # Second graph: Contour only
        ax2 = fig.add_subplot(122)
        ax2.contour(X, Y, Z, levels=contour_levels, colors='red', linewidths=2.5)
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')

        plt.tight_layout()
        path = Plotter._save_plot(h, "2d_function", "")
        Plotter._maybe_show(h, fig)
        return fig, path

    @staticmethod
    def plot_multiple_distributions_with_indicators(h: dict, distances_per_method, quantiles, indicators,
                                                    avg_intervals_per_method, title, xlabel, ylabel):
        """
        Plot the distribution of distances for multiple methods using KDE with quantile-based indicators,
        and show cumulative percentages at specific x ticks.

        Parameters:
        - distances_per_method: Dict of method names to distances (List or np.ndarray).
        - quantiles: List of float values in [0, 1] (e.g., [0.95] for 95% mass to the left).
        - indicators: List of float values in [0, 1] to indicate specific distances (e.g., [0.02, 0.05, 0.1, 0.2]).
        - avg_intervals_per_method: Dict of method names to average number of intervals before narrow idx.
        - title: Title of the plot.
        - xlabel: Label for the x-axis.
        - ylabel: Label for the y-axis.
        """

        fig = plt.figure(figsize=(12, 8))
        palette = sns.color_palette(n_colors=len(distances_per_method))

        for i, ((method_name, distances), color) in enumerate(zip(distances_per_method.items(), palette)):
            distances = np.asarray(distances)

            kde = gaussian_kde(distances)
            x_vals = np.linspace(0, 1, 1000)
            y_vals = kde(x_vals)

            # plt.fill_between(x_vals, y_vals, alpha=0.3, color=color)
            plt.plot(x_vals, y_vals,
                     label=f"{method_name} (Avg. Intervals: {avg_intervals_per_method[method_name]:.1f})", color=color,
                     alpha=0.8)

            for q in quantiles:
                q_value = np.quantile(distances, q)
                plt.axvline(q_value, color=color, alpha=0.8, label=f'{method_name}: {int(q * 100)}% @ {q_value:.3f}')

            for x in indicators:
                percent = 100 * np.mean(distances <= x)
                y_at_x = kde(x)[0]
                plt.plot(x, y_at_x, marker='o', color=color, markersize=4)
                plt.text(x, y_at_x, f'{percent:.1f}%', color='black', ha='center', fontsize=8)

        # generally [0, 1] but [0, 0.5] for small distances
        if np.max(distances) <= 0.3:
            plt.xlim(0, 0.25)
            plt.xticks(np.arange(0, 0.26, 0.01))
        else:
            plt.xlim(0, 1)
            plt.xticks(np.arange(0, 1.1, 0.1))

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        path = Plotter._save_plot(h, "dist_distr", "")
        Plotter._maybe_show(h, fig)
        return fig, path

    @staticmethod
    def plot_1d_function(
            f: Union[Callable[[float], float], List[Callable[[float], float]]],
            x_min: float,
            x_max: float,
            x_coords: List[float],
            title: str,
            ylim: Union[None, float] = None,
            legend: Union[str, List[str]] = None,
            # labels: Union[None, List[str]] = None
    ):
        x_values = np.linspace(x_min, x_max, 10_000)

        if not isinstance(f, list):
            f = [f]

        plt.figure(figsize=(30, 18))

        for i, func in enumerate(f):
            y_values = [func(x) for x in x_values]
            label = legend[i] if isinstance(legend, list) else (legend if legend is not None else f"f{i}")
            plt.plot(x_values, y_values, label=label)

            # Find and mark the minimum of the function
            min_idx = np.argmin(y_values)
            plt.scatter(x_values[min_idx], y_values[min_idx], color='red',
                        label=f'Min of f{i} at x={x_values[min_idx]:.2f}')

        # scatter the coordinates
        for x in x_coords:
            y_coord = func(x)
            plt.scatter(x, y_coord, color='green', marker='x', s=100, label=f'Coord at x={x:.2f}')

        if ylim is not None:
            plt.ylim(0, ylim)

        plt.title(title)

        # Add more grid lines
        plt.grid(which='major', linestyle='-', linewidth=0.8)
        plt.grid(which='minor', linestyle='--', linewidth=0.5)
        plt.minorticks_on()

        plt.legend()
        plt.show()

    @staticmethod
    def plot_2d_levelset(
            funcs: Union[Callable[[float, float], float], List[Callable[[float, float], float]]],
            x_min: float,
            x_max: float,
            y_min: float,
            y_max: float,
            z_max: float,
            labels: List[str] = None,
            colors: List[str] = None
    ):
        """
        Plot the region {(x, y) | f(x, y) < z_max} as a filled contour for one or multiple functions.
        """
        # Ensure funcs is a list
        if not isinstance(funcs, (list, tuple)):
            funcs = [funcs]

        # Default labels/colors
        if labels is None:
            labels = [f"f{i}" for i in range(len(funcs))]
        if colors is None:
            default_colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'violet']
            colors = default_colors[:len(funcs)]

        # Build grid
        x = np.linspace(x_min, x_max, 300)
        y = np.linspace(y_min, y_max, 300)
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, f in enumerate(funcs):
            Z = np.vectorize(f)(X, Y)
            region = Z < z_max

            # Filled contour for the region where f(x, y) < z_max
            ax.contourf(X, Y, region, levels=[0.5, 1],
                        colors=[colors[i]], alpha=0.5)

            # Contour line at f(x, y) = z_max
            ax.contour(X, Y, Z, levels=[z_max],
                       colors='black', linewidths=1.2)

        # Turn off autoscaling margins, enforce exact limits & equal aspect
        ax.margins(0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.set_aspect('equal', adjustable='box')

        # Labels and grid
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Regions where f(x, y) < {z_max}")
        ax.grid(True)

        plt.show()

    @staticmethod
    def plot_coordinates(coordinates: List[Tuple[float, float]], title):
        """
        Plot a list of coordinates on a 2D scatter plot.

        Parameters:
        - coordinates: List of tuples (x, y) representing the coordinates to plot.
        - title: Title of the plot.
        """
        x_values, y_values = zip(*coordinates)
        # max 0, 1
        x_values = np.clip(x_values, 0.0 + 1e-6, 1.0 - 1e-6)  # Avoid 0 to prevent overlap with lower limit
        y_values = np.clip(y_values, 0.0 + 1e-6, 1.0 - 1e-6)  # Avoid 0 to prevent overlap with lower limit
        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values, color='blue', alpha=0.6)
        plt.title(title)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.axis('equal')

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.margins(0)
        plt.show()

    @staticmethod
    def plot_volumes_over_time(h, names, colors, volumes, volume_stds, N, D, log_y_scale: bool,
                               time_points=None):
        """
        Plot the evolution of confidence interval volumes over time (up to N observations).
        """

        fig, ax = plt.subplots(figsize=(10, 6))

        for name, color, volume_list, std_list in zip(names, colors, volumes, volume_stds):
            if time_points is None:
                timesteps = np.arange(1, len(volume_list) + 1)
            else:
                timesteps = np.asarray(time_points, dtype=int)
            ax.plot(timesteps, volume_list, label=name, color=color)

            if np.any(np.array(std_list) > 0):
                # Calculate upper and lower bounds for 95% confidence interval
                upper_bound = np.array(volume_list) + 1.96 * np.array(std_list)
                lower_bound = np.array(volume_list) - 1.96 * np.array(std_list)

                # Plot the bounds as a transparent fill
                ax.fill_between(timesteps, lower_bound, upper_bound, color=color, alpha=0.2)

        ax.set_xlabel("Time (observations)")
        ax.set_ylabel("Volume of confidence interval")
        ax.set_title(
            f"Confidence Interval Volumes over Time (N={N}, dim={D}, data_type={h['data_type']}, "
            f"{'[log]' if log_y_scale else ''})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        if log_y_scale:
            plt.yscale('log')

        plt.tight_layout()
        path = Plotter._save_plot(h, f"vol_over_time_log={log_y_scale}", "")
        Plotter._maybe_show(h, fig)

        return fig, path

    @staticmethod
    def plot_coverage_over_time(h, names, colors, coverages, N, D, time_points, alpha: float):
        """
        Plot per-time-step coverage over time (fraction of repeats covering the true mean).
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        timesteps = np.asarray(time_points, dtype=int)
        for name, color, coverage_list in zip(names, colors, coverages):
            ax.plot(timesteps, coverage_list, label=name, color=color)

        target = 1.0 - float(alpha)
        ax.axhline(target, color="black", linestyle="--", linewidth=1.2,
                   label=f"1 - alpha = {target:.2f}")

        ax.set_xlabel("Time (observations)")
        ax.set_ylabel("Coverage")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(
            f"Per-time-step Coverage (N={N}, dim={D}, data_type={h['data_type']})"
        )
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        plt.tight_layout()
        path = Plotter._save_plot(h, "coverage_over_time", "")
        Plotter._maybe_show(h, fig)
        return fig, path

    @staticmethod
    def plot_volume_ratios_over_time(h, names, colors, volume_ratios, N, D, y_log_scale):
        """
        Plot the evolution of confidence interval volume ratios over time (up to N observations).
        """

        fig, ax = plt.subplots(figsize=(10, 6))

        for name, color, ratio_list in zip(names, colors, volume_ratios):
            timesteps = np.arange(1, len(ratio_list) + 1)
            ax.plot(timesteps, ratio_list, label=name, color=color)

        ax.set_xlabel("Time (observations)")
        ax.set_ylabel("Volume Ratio of confidence interval")
        ax.set_title(
            f"Confidence Interval Volume Ratios over Time (N={N}, dim={D}, data_type={h['data_type']}, "
            f"{'[log]' if y_log_scale else ''})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        if y_log_scale:
            plt.yscale('log')

        plt.tight_layout()
        path = Plotter._save_plot(h, f"vol_ratio_over_time_log={y_log_scale}", "")
        Plotter._maybe_show(h, fig)

        return fig, path

    @staticmethod
    def plot_volumes_over_dims(h, names, colors, final_volumes, final_stds, Ds, y_log_scale: bool = False,
                               xlabel: str = "Dimension (d)", title: Optional[str] = None,
                               ref_lines: Optional[List[dict]] = None,
                               x_marker: Optional[dict] = None):
        dims = np.array(list(Ds))
        final_volumes = np.asarray(final_volumes, dtype=float)
        final_stds = np.asarray(final_stds, dtype=float)

        M = len(names)
        if final_volumes.shape != (M, len(dims)):
            raise ValueError(
                f"final_volumes shape {final_volumes.shape} does not match (num_methods, len(Ds)) = {(M, len(dims))}")

        fig, ax = plt.subplots(figsize=(10, 6))

        for name, color, vols, stds in zip(names, colors, final_volumes, final_stds):
            # main line
            ax.plot(dims, vols, label=name, marker='o', linestyle='-', color=color)

            # if there's any non-zero stds, plot 95% CI band
            if np.any(stds > 0):
                upper = vols + 1.96 * stds
                lower = vols - 1.96 * stds
                # keep bounds non-negative
                lower = np.maximum(lower, 0.0)
                ax.fill_between(dims, lower, upper, color=color, alpha=0.2, linewidth=0.0)

        if ref_lines:
            for ref in ref_lines:
                y = ref.get("y", None)
                if y is None:
                    continue
                ax.axhline(
                    y=y,
                    color=ref.get("color", "gray"),
                    linestyle=ref.get("linestyle", "--"),
                    linewidth=1.5,
                    label=ref.get("label", None),
                )

        if x_marker:
            x_val = x_marker.get("x", None)
            if x_val is not None:
                ax.axvline(
                    x=x_val,
                    color=x_marker.get("color", "black"),
                    linestyle=x_marker.get("linestyle", ":"),
                    linewidth=1.5,
                    label=x_marker.get("label", None),
                )

        ax.set_xticks(dims)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Volume of confidence interval")
        if title is None:
            title = (f"CI Volume vs {xlabel} (N={h.get('N', '')}, data_type={h.get('data_type', '')}) "
                     f"{'[log y]' if y_log_scale else ''}")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        if y_log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        path = Plotter._save_plot(h, f"volumes_over_dims_log={y_log_scale}", "")
        Plotter._maybe_show(h, fig)
        return fig, path

    @staticmethod
    def plot_volume_ratios_over_dims(h, names, colors, ratio_matrix, ratio_stds, Ds, y_log_scale: bool = False,
                                     ylabel="Volume ratio", xlabel: str = "Dimension (d)",
                                     title: Optional[str] = None):
        """
        Plot volume ratios over dimensions with 95% CI bands.

        Args:
            ratio_matrix: array-like shape (M, len(Ds)) with ratio values (e.g. method_volume / baseline_volume).
            ratio_stds: array-like same shape with std deviations for the ratios (or zeros).
            Ds: iterable of integer dimensions.
            ylabel: y-axis label (default "Volume ratio").
        Returns:
            (fig, path)
        """
        dims = np.array(list(Ds))
        ratio_matrix = np.asarray(ratio_matrix, dtype=float)
        ratio_stds = np.asarray(ratio_stds, dtype=float)

        M = len(names)
        if ratio_matrix.shape != (M, len(dims)):
            raise ValueError("ratio_matrix shape mismatch")

        fig, ax = plt.subplots(figsize=(10, 6))

        for name, color, ratios, stds in zip(names, colors, ratio_matrix, ratio_stds):
            ax.plot(dims, ratios, label=name, marker='o', linestyle='-', color=color)
            if np.any(stds > 0):
                upper = ratios + 1.96 * stds
                lower = ratios - 1.96 * stds
                lower = np.maximum(lower, 0.0)
                ax.fill_between(dims, lower, upper, color=color, alpha=0.2, linewidth=0.0)

        ax.set_xticks(dims)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is None:
            title = f"{ylabel} vs {xlabel} (N={h.get('N', '')}, data_type={h.get('data_type', '')})"
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        if y_log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        path = Plotter._save_plot(h, "volume_ratios_over_dims", "")
        Plotter._maybe_show(h, fig)
        return fig, path

    @staticmethod
    def _save_plot(h: dict, dir, filename) -> Optional[str]:
        if h['save_plots_locally'] is False:
            return None

        # Create the directory if it does not exist
        dir_path = os.path.join(h['log_path'], dir)
        os.makedirs(dir_path, exist_ok=True)  # Ensure the directory exists

        key_name = Key.name(h)
        filename = f"{filename}_{key_name}"

        # Save the plot as a PDF
        plt.savefig(os.path.join(dir_path, f"{filename}.pdf"), bbox_inches='tight')
        try:
            tikzplotlib.save(os.path.join(dir_path, f"{filename}.tex"))
        except Exception as e:
            print(f"Failed to save TikZ file: {e}")

        return os.path.join(dir_path, f"{filename}")  # without extension

    @staticmethod
    def clear_figs():
        """
        Clear all figures in matplotlib.
        """
        plt.close('all')

    class GridPlotter:
        # supports both simplex and full grid, but only 2d
        @staticmethod
        def _project_simplex_to_2d(bary):
            """Project 3-component barycentric coords (a,b,c) onto a 2D triangle."""
            v0 = np.array([0.0, 0.0])
            v1 = np.array([1.0, 0.0])
            v2 = np.array([0.5, np.sqrt(3) / 2.0])
            return bary @ np.vstack([v0, v1, v2])

        @staticmethod
        def _convex_hull(points):
            """Monotone chain convex hull. Input (n,2). Returns hull (m,2) closed (first==last)."""
            pts = np.asarray(points, dtype=float)
            if pts.shape[0] <= 1:
                return pts.copy()
            # remove (near) duplicates by rounding then unique
            pts = np.unique(np.round(pts, 12), axis=0)
            if pts.shape[0] <= 1:
                return np.vstack([pts[0], pts[0]])
            # sort lexicographically
            pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

            lower = []
            for p in pts:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(tuple(p))
            upper = []
            for p in pts[::-1]:
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(tuple(p))

            hull = np.array(lower[:-1] + upper[:-1], dtype=float)
            if hull.shape[0] > 0:
                hull = np.vstack([hull, hull[0]])
            return hull

        @staticmethod
        def display_2d_confidence_region(
                # a: Optional[float], b: Optional[float], a_prime: Optional[float],
                # b_prime: Optional[float], pow: Optional[int],
                f_tilde: Optional[Callable[[float, float], float]], C: Optional[float],
                low, up, title: str, grid=None, in_cs=None,
                domain_start: float = 0.0, domain_end: float = 1.0):
            """
            Plots (x-a)^2 / b^2 + (y-a')^2 / b'^2 = 1 along with rectangle boundaries
            and optionally the confidence region, zooming in on the specified domain.
            """

            plt.figure(figsize=(12, 12))

            # Plot the rectangle boundaries
            if len(low) > 0:
                for t in range(low.shape[0]):
                    lower_x, lower_y = low[t]
                    upper_x, upper_y = up[t]
                    plt.gca().add_patch(plt.Rectangle(
                        (lower_x, lower_y),  # Bottom-left corner
                        upper_x - lower_x,  # Width
                        upper_y - lower_y,  # Height
                        edgecolor='skyblue', facecolor='none', linewidth=1.5, label="Boundary" if t == 0 else None
                    ))

            # Plot the confidence region if grid and in_cs are provided
            if grid is not None and in_cs is not None:
                grid = np.asarray(grid)
                if grid.ndim != 2:
                    raise ValueError("grid must be a 2D array")

                # Project grid to 2D if necessary
                if grid.shape[1] == 3 and np.allclose(grid.sum(axis=1), 1.0, atol=1e-9):
                    pts2d = Plotter.GridPlotter._project_simplex_to_2d(grid)
                elif grid.shape[1] == 2:
                    pts2d = grid.astype(float)
                else:
                    raise ValueError("grid shape must be (n,2) or (n,3) barycentric")

                mask = np.asarray(in_cs, dtype=bool)
                if mask.shape[0] != pts2d.shape[0]:
                    raise ValueError("in_cs must have the same length as grid")

                pts_in = pts2d[mask]
                pts_out = pts2d[~mask]

                if pts_in.size:
                    plt.scatter(pts_in[:, 0], pts_in[:, 1], s=1, alpha=0.9, color='tab:green', label="Inside CS")

                    # Plot the convex hull of inside points
                    hull = Plotter.GridPlotter._convex_hull(pts_in)
                    if hull.size:
                        plt.plot(hull[:, 0], hull[:, 1], linewidth=1.5, color='tab:green', label="CS Boundary")

            if f_tilde is not None:
                assert C is not None, "C must be provided if f_tilde is given if f_tilde is given."
                # plt.contour(X, Y, contour, levels=[C])
                resolution = 1_000

                plt.contour(
                    np.linspace(domain_start, domain_end, resolution),
                    np.linspace(domain_start, domain_end, resolution),
                    np.vectorize(f_tilde)(*np.meshgrid(
                        np.linspace(domain_start, domain_end, resolution),
                        np.linspace(domain_start, domain_end, resolution),
                    )),
                    levels=[C],  # draw only f(x) = C level
                    colors="orange",
                    linewidths=2,
                )

            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.title(title)
            plt.grid(True)
            plt.axis("equal")
            plt.legend()
            plt.xlim(domain_start, domain_end)
            plt.ylim(domain_start, domain_end)
            plt.show()

        @staticmethod
        def _project_simplex_to_3d(bary: np.ndarray) -> np.ndarray:
            bary = np.asarray(bary, dtype=float)
            if bary.ndim != 2 or bary.shape[1] != 4:
                raise ValueError("bary must be shape (n,4) to project to 3D")
            v = np.array([[0., 0., 0.],
                          [1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]])
            return bary.dot(v)

        @staticmethod
        def _draw_cuboid(ax, lower, upper, linewidth=1.0, label=None):
            lower = np.asarray(lower)
            upper = np.asarray(upper)
            x0, y0, z0 = lower
            x1, y1, z1 = upper
            corners = np.array([
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ])
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            for i, j in edges:
                xs, ys, zs = zip(corners[i], corners[j])
                ax.plot(xs, ys, zs, linewidth=linewidth,
                        color='blue', label=label if (label is not None and i == 0 and j == 1) else None)

        # ---------- fast 3D display ----------
        @staticmethod
        def display_3d_confidence_region(ellipse_params: Optional[np.ndarray],
                                         low, up,
                                         title: str,
                                         grid=None, in_cs=None,
                                         *,
                                         max_points: int = 2500,
                                         ellipsoid_resolution: tuple = (80, 40),  # u_steps, v_steps
                                         draw_hull: bool = True,
                                         rasterize: bool = True,
                                         domain_start: float = 0.0,
                                         domain_end: float = 1.0):
            """
            Faster 3D plotting.

            Args:
                ellipse_params: (3,2) array like [[cx, rx],[cy, ry],[cz, rz]] or None
                low, up: list/array of axis-aligned boxes (m,3)
                grid: (n,3) coordinates or (n,4) barycentric
                in_cs: boolean mask length n
                max_points: max number of points to scatter in total (split between in/out)
                ellipsoid_resolution: (u_steps, v_steps) for surface sampling (kept low)
                draw_hull: whether to compute & draw the convex hull (disabled automatically if too many pts)
                rasterize: use rasterization on large scatter to speed vector renderers
            """
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(111, projection='3d')

            # -------- ellipsoid (fast, low-res) --------
            if ellipse_params is not None:
                ep = np.asarray(ellipse_params)
                if ep.shape == (3, 2):
                    center = np.array([ep[0, 0], ep[1, 0], ep[2, 0]])
                    radii = np.array([ep[0, 1], ep[1, 1], ep[2, 1]])
                    # sample param grid but limit total points
                    u_steps, v_steps = ellipsoid_resolution
                    u = np.linspace(0, 2 * np.pi, u_steps)
                    v = np.linspace(0, np.pi, v_steps)
                    uu, vv = np.meshgrid(u, v)
                    xs = (center[0] + radii[0] * np.cos(uu) * np.sin(vv)).ravel()
                    ys = (center[1] + radii[1] * np.sin(uu) * np.sin(vv)).ravel()
                    zs = (center[2] + radii[2] * np.cos(vv)).ravel()
                    # downsample ellipsoid surface points if too many
                    surf_count = xs.size
                    if surf_count > 1500:
                        sel = np.random.choice(surf_count, size=1500, replace=False)
                        xs, ys, zs = xs[sel], ys[sel], zs[sel]
                    # scatter representation is faster than a dense surface
                    ax.scatter(xs, ys, zs, s=6, alpha=0.25, marker='o', label='Ellipsoid', rasterized=rasterize)

            # -------- cuboid boundaries --------
            if len(low) > 0:
                for t in range(np.atleast_2d(low).shape[0]):
                    lower = np.atleast_2d(low)[t]
                    upper = np.atleast_2d(up)[t]
                    Plotter.GridPlotter._draw_cuboid(ax, lower, upper, linewidth=1.5,
                                                     label="Boundary" if t == 0 else None)

            # -------- grid points (downsample before plotting) --------
            if grid is not None and in_cs is not None:
                grid = np.asarray(grid)
                if grid.ndim != 2:
                    raise ValueError("grid must be a 2D array")
                if grid.shape[1] == 4 and np.allclose(grid.sum(axis=1), 1.0, atol=1e-9):
                    pts3d = Plotter.GridPlotter._project_simplex_to_3d(grid)
                elif grid.shape[1] == 3:
                    pts3d = grid.astype(float)
                else:
                    raise ValueError("grid shape must be (n,3) or (n,4) barycentric for 3D plotting")

                mask = np.asarray(in_cs, dtype=bool)
                if mask.shape[0] != pts3d.shape[0]:
                    raise ValueError("in_cs must have the same length as grid")

                pts_in = pts3d[mask]
                pts_out = pts3d[~mask]

                # split budget between inside & outside points proportionally
                n_in = pts_in.shape[0]
                n_out = pts_out.shape[0]
                total = n_in + n_out
                if total == 0:
                    pass
                else:
                    # compute sample sizes
                    if total > max_points:
                        frac_in = n_in / total if total > 0 else 0.5
                        sel_in = int(np.round(max_points * frac_in))
                        sel_out = max_points - sel_in
                        sel_in = min(sel_in, n_in)
                        sel_out = min(sel_out, n_out)
                    else:
                        sel_in, sel_out = n_in, n_out

                    # sample indices (or take all)
                    def sample_rows(arr, k):
                        if arr.shape[0] <= k:
                            return arr
                        idx = np.random.choice(arr.shape[0], size=k, replace=False)
                        return arr[idx]

                    pts_in_plot = sample_rows(pts_in, sel_in) if n_in > 0 else np.empty((0, 3))
                    # pts_out_plot = sample_rows(pts_out, sel_out) if n_out > 0 else np.empty((0, 3))

                    # plot outside faint, inside solid; use rasterize for large sets
                    # if pts_out_plot.size:
                    #     ax.scatter(pts_out_plot[:, 0], pts_out_plot[:, 1], pts_out_plot[:, 2],
                    #                s=6, alpha=0.18, marker='o', label="Outside CS",
                    #                rasterized=(rasterize and pts_out_plot.shape[0] > 500))
                    if pts_in_plot.size:
                        ax.scatter(pts_in_plot[:, 0], pts_in_plot[:, 1], pts_in_plot[:, 2],
                                   s=20, alpha=0.9, marker='o', label="Inside CS",
                                   rasterized=(rasterize and pts_in_plot.shape[0] > 500))

                        # try hull if requested and not too many points
                        if draw_hull and pts_in.shape[0] > 3 and pts_in.shape[0] <= 4000:
                            try:
                                from scipy.spatial import ConvexHull
                                hull = ConvexHull(pts_in)
                                faces = [pts_in[simplex] for simplex in hull.simplices]
                                poly = Poly3DCollection(faces, alpha=0.22)
                                poly.set_edgecolor('tab:blue')
                                poly.set_linewidth(0.5)
                                ax.add_collection3d(poly)
                            except Exception:
                                # if SciPy missing or hull fails, skip quietly
                                pass

            # -------- final touches --------
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.set_zlabel("Dimension 3")
            ax.set_title(title)
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
            # ax.set_xlim(0, 1)
            # ax.set_ylim(0, 1)
            # ax.set_zlim(0, 1)
            ax.set_xlim(domain_start, domain_end)
            ax.set_ylim(domain_start, domain_end)
            ax.set_zlim(domain_start, domain_end)

            # attempt to make aspect ratio more equal (heuristic)
            try:
                max_range = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).ptp().max()
                Xb = 0.5 * (ax.get_xlim3d()[0] + ax.get_xlim3d()[1])
                Yb = 0.5 * (ax.get_ylim3d()[0] + ax.get_ylim3d()[1])
                Zb = 0.5 * (ax.get_zlim3d()[0] + ax.get_zlim3d()[1])
                ax.set_xlim3d(Xb - max_range / 2, Xb + max_range / 2)
                ax.set_ylim3d(Yb - max_range / 2, Yb + max_range / 2)
                ax.set_zlim3d(Zb - max_range / 2, Zb + max_range / 2)
            except Exception:
                pass

            plt.show()


if __name__ == "__main__":
    # Example usage
    plotter = Plotter()

    Plotter.plot_coordinates([(-1, 0.2), (0.3, 0.4), (1, 0.6)], "Example Coordinates Plot")

# Some tests
import numpy as np

from adaptive_sample.conf_sequences.conf_sphere.banach_conf_sphere import BanachSphere
from adaptive_sample.conf_sequences.conf_sphere.chugg_conf_sphere import EmpiricalBernsteinConfSphere
from adaptive_sample.shared_code import DataDistributionChecker


def sample_mixture_1(N, d, seed=None):
    # # mixture 2 is just uniform in [0,1]^d
    # rng = np.random.RandomState(seed)
    # return rng.rand(N, d)

    data, _true_mean = DataDistributionChecker.get_data(f'fixed_{d}d', N)
    return data


def main(ConfSphereClass, optimistic_rescale, D, N, SKIP_FIRST_N=0):
    h = {
        'alpha': 0.05,
        'batch_size': 1,
        'data_type': f'fixed_{D}d',
        'N': N,
        'cut_to_simplex': False,
        'plot_grid_2d': False,
        'calculate_volume': True,
        'verbose_progress': True,
        'D': D,
        'conf_sphere_optimistic_rescale': optimistic_rescale
    }

    # Initialize random number generator
    rng_global = np.random.RandomState(41)

    # Generate data
    X = sample_mixture_1(N=h['N'], d=h['D'], seed=rng_global.randint(1_000_000))

    # Initialize confidence sphere
    cs = ConfSphereClass(h, X) # BanachSphere(h, X) or EmpiricalBernsteinConfSphere(h, X)
    # ts, means, lowers, uppers, volumes, _ = cs.run()
    import time  # Add this at the top of the file if not already imported

    start_time = time.time()
    ts, means, lowers, uppers, volumes, _ = cs.run()
    end_time = time.time()
    print(f"Execution time for cs.run(): {end_time - start_time:.4f} seconds")

    # Define dist_fns and results for plotting
    dist_fns = {'Mixture 1': sample_mixture_1}
    results = {
        'Mixture 1': {
            'Empirical-Bernstein': {
                'volumes': np.zeros((1, len(ts))),
                'lowers': np.zeros((1, len(ts), h['D'])),
                'uppers': np.zeros((1, len(ts), h['D']))
            }
        }
    }

    # Populate results
    results['Mixture 1']['Empirical-Bernstein']['volumes'][0, :] = volumes
    results['Mixture 1']['Empirical-Bernstein']['lowers'][0, :, :] = lowers
    results['Mixture 1']['Empirical-Bernstein']['uppers'][0, :, :] = uppers

    # Define ns_plot
    ns_plot = np.arange(1, len(ts) + 1)

    # Plot widths
    ns_plot_skipped = ns_plot[SKIP_FIRST_N:]

    # Plot volumes
    plt.figure(figsize=(7, 4))
    for dist_name in dist_fns.keys():
        mean_vol = results[dist_name]['Empirical-Bernstein']['volumes'].mean(axis=0)[SKIP_FIRST_N:]
        std_vol = results[dist_name]['Empirical-Bernstein']['volumes'].std(axis=0)[SKIP_FIRST_N:]

        plt.plot(ns_plot_skipped, mean_vol, label=f'Volume - {dist_name}')
        plt.fill_between(ns_plot_skipped, mean_vol - std_vol, mean_vol + std_vol, alpha=0.15)

    plt.xlabel('n')
    plt.ylabel('Volume')
    plt.title('Volumes of Confidence Sets')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.show()

    # Plot bounds (lowers and uppers)
    plt.figure(figsize=(7, 4))
    for dist_name in dist_fns.keys():
        mean_lowers = results[dist_name]['Empirical-Bernstein']['lowers'].mean(axis=0)[SKIP_FIRST_N:, :]
        mean_uppers = results[dist_name]['Empirical-Bernstein']['uppers'].mean(axis=0)[SKIP_FIRST_N:, :]

        for dim in range(h['D']):
            plt.plot(ns_plot_skipped, mean_lowers[:, dim], label=f'Lower (dim {dim + 1}) - {dist_name}')
            plt.plot(ns_plot_skipped, mean_uppers[:, dim], label=f'Upper (dim {dim + 1}) - {dist_name}')

    plt.xlabel('n')
    plt.ylabel('Bounds')
    plt.title('Lowers and Uppers of Confidence Sets')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.show()

    # Plot width (upper - lower) for each dimension
    plt.figure(figsize=(7, 4))
    for dist_name in dist_fns.keys():
        mean_lowers = results[dist_name]['Empirical-Bernstein']['lowers'].mean(axis=0)[SKIP_FIRST_N:, :]
        mean_uppers = results[dist_name]['Empirical-Bernstein']['uppers'].mean(axis=0)[SKIP_FIRST_N:, :]

        for dim in range(h['D']):
            widths = mean_uppers[:, dim] - mean_lowers[:, dim]
            plt.plot(ns_plot_skipped, widths, label=f'Width (dim {dim + 1}) - {dist_name}')

    plt.xlabel('n')
    plt.ylabel('Width')
    plt.title('Widths of Confidence Sets')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.show()


def compute_volume_at_matches_run_test(ConfSphereClass: type, optimistic_rescale: bool, n_checks=200,
                                       atol_bounds=1e-12):
    h = {
        'alpha': 0.05,
        'batch_size': 1,
        'data_type': 'fixed_2d',
        'N': 10_000,
        'cut_to_simplex': False,
        'plot_grid_2d': False,
        'calculate_volume': True,
        'verbose_progress': True,
        'D': 2,
        'conf_sphere_optimistic_rescale': optimistic_rescale,
    }

    # Initialize random number generator
    rng_global = np.random.RandomState(41)

    # Generate data
    X = sample_mixture_1(N=h['N'], d=h['D'], seed=rng_global.randint(1_000_000))

    # Initialize confidence sphere
    cs = ConfSphereClass(h, X)
    ts, means_run, lowers_run, uppers_run, volumes_run, volume_stds_run = cs.run()

    T = len(ts)
    if T == 0:
        raise ValueError("No time steps to check (len(ts) == 0)")

    # choose indices to check: up to n_checks uniformly across [0, T-1]
    if n_checks >= T:
        indices = list(range(T))
    else:
        indices = np.unique(np.round(np.linspace(0, T - 1, n_checks)).astype(int)).tolist()

    scale = cs.rescale_factor
    shift = cs.box_center

    cs = ConfSphereClass(h, X)
    for t in indices:
        num_obs = (t + 1) * cs.batch_size

        # Rebuild scaled center & radius and map -> original
        mu_hat_scaled, r_scaled = cs._rebuild_state_up_to(num_obs)
        mu_hat_orig = (mu_hat_scaled * scale) + shift
        r_orig = float(r_scaled * scale)

        # compute lower/upper from rebuilt state (original coords)
        lower_rebuilt = np.maximum(cs.domain_low, mu_hat_orig - r_orig)
        upper_rebuilt = np.minimum(cs.domain_high, mu_hat_orig + r_orig)

        # compare to run-time stored bounds
        lower_run = lowers_run[t]
        upper_run = uppers_run[t]

        if not np.allclose(lower_run, lower_rebuilt, atol=atol_bounds, rtol=0.0):
            diff = np.abs(lower_run - lower_rebuilt)
            idx = int(np.argmax(diff))
            raise AssertionError(
                f"Lower bound mismatch at t={t}: run lower[{idx}]={lower_run[idx]:.12g} != rebuilt {lower_rebuilt[idx]:.12g} (abs diff={diff[idx]:.3e})"
            )

        if not np.allclose(upper_run, upper_rebuilt, atol=atol_bounds, rtol=0.0):
            diff = np.abs(upper_run - upper_rebuilt)
            idx = int(np.argmax(diff))
            raise AssertionError(
                f"Upper bound mismatch at t={t}: run upper[{idx}]={upper_run[idx]:.12g} != rebuilt {upper_rebuilt[idx]:.12g} (abs diff={diff[idx]:.3e})"
            )

        # Now compute volumes via the class API (this will internally use the same compute_volume)
        vol_at, vol_std_at = cs.compute_volume(t, None, lower_rebuilt, upper_rebuilt,
                                               {'mu_hat': mu_hat_orig, 'radius': r_orig})

        vol_run = float(volumes_run[t])
        std_run = float(volume_stds_run[t])

        # If both runs reported zero std -> require near exact equality.
        if (vol_std_at == 0.0) and (std_run == 0.0):
            if not np.isclose(vol_run, vol_at, atol=1e-12, rtol=0.0):
                raise AssertionError(
                    f"Volume mismatch (exact) at t={t}: run vol={vol_run:.12g} != compute_volume vol={vol_at:.12g}"
                )
        else:
            # allow difference up to a few sigma of the larger std (account for Monte-Carlo noise)
            allowed = max(1e-8, 5.0 * max(vol_std_at, std_run))
            if abs(vol_run - vol_at) > allowed:
                raise AssertionError(
                    f"Volume mismatch (approx) at t={t}: run vol={vol_run:.12g} != compute_volume vol={vol_at:.12g} "
                    f"(diff={abs(vol_run - vol_at):.3e} > allowed={allowed:.3e}). "
                    f"(stds: run={std_run:.3e}, compute={vol_std_at:.3e})"
                )

    print(f"All {len(indices)} checks passed (sampled T={T}).")
    return True


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # EmpiricalBernsteinConfSphere | BanachSphere
    CS = EmpiricalBernsteinConfSphere

    OPTIMISTIC_RESCALE = False
    compute_volume_at_matches_run_test(CS, OPTIMISTIC_RESCALE, n_checks=100, atol_bounds=1e-12)
    main(CS, OPTIMISTIC_RESCALE, D=2, N=1_000, SKIP_FIRST_N=0)

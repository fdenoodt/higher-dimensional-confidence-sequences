from adaptive_sample.conf_sequences.experiments import Experiments
from adaptive_sample.shared_code import DataDistributionChecker
from shared.decorators import init_decorator_without_torch, timer_decorator
from shared.hyperparameters import Hyperparameters


class ConfSeqHyperparameters(Hyperparameters):
    @staticmethod
    def get():
        h = {
            'method': 'confidence_sequence',  # method to use
            'N': 1_000,  # 3_000,  # number of samples to generate
            'batch_size': 1,  # batch size for confidence sequences
            'stop_early_threshold': 0.1,  # threshold for narrow confidence intervals
            'alpha': 0.05,  # significance level for confidence intervals
            'data_type': 'uniform',  # type of data to generate (uniform, gaussian, fixed, etc.)
            'cut_to_simplex': False,  # whether to cut data to simplex [0,1]^D. Only applicable if higher dimension.

            # Experiments
            'individual_run_experiment': False,  # (1)
            'volume_vs_dim_experiment': False,  # (2)
            'volume_vs_time_experiment': False,  # (3)
            'perform_l1_err_experiment': False,  # (4)
            'perform_coverage_experiment': False,  # (5)
            'coverage_vs_time_experiment': False,  # (6)

            ### newly added experiments
            'volume_vs_p_experiment': False,  # parabola_power
            'volume_vs_C_experiment': False,
            'volume_vs_parabola_gr_sz_experiment': False,

            # Experiment parameters
            'perform_l1_err_experiment_nb_runs': 5_00,
            'perform_coverage_experiment_nb_runs': 5_00,
            'calculate_volume': False,  # whether to calculate volume of confidence sets

            'volume_vs_dim_experiment_max_dim': 20,
            'volume_vs_dim_experiment_N': 1_000,

            'volume_vs_time_experiment_time_points': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'volume_vs_time_experiment_D': 10,

            'coverage_vs_time_experiment_time_points': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'coverage_vs_time_experiment_D': 10,
            'coverage_vs_time_experiment_repeats': 200,

            'volume_vs_p_experiment_Ps': range(1, 21),
            # try various `parabola_power` vals, and set `parabola_adaptive` False
            'volume_vs_p_experiment_N': 1_000,

            # the options are hard coded
            # 'volume_vs_C_experiment_Ps': range(1, 21),
            'volume_vs_C_experiment_N': 1_000,
            # important: `parabola_power` needs to be fixed then, e.g. 8, and set `parabola_adaptive` False
            'volume_vs_C_experiment_fixed_parabola_pow': 12,

            'volume_vs_parabola_gr_sz_experiment_grid_sizes': [101, 201, 301, 401, 501, 601, 700, 801, 901, 1001,
                                                               1101, 1201, 1301, 1401, 1501, 1601, 1701, 1801, 1901,
                                                               2001, 2101, 2101, 2201, 2301, 2401, 2501, 2601, 2701,
                                                               2801, 2901, 3001, 3101, 3201, 3301, 3401, 3501, 3601,
                                                               3701, 3801, 3901, 2001, 2101,
                                                               ],
            'volume_vs_parabola_gr_sz_experiment_N': 1_000,  # this is time steps, not grid size

            'num_intervals': 1,
            'plot_interval_widths': True,
            'interval_types': [
                # Higher-dimensional hedged capital methods:
                # "CONF_SPHERE",
                # "BANACH_SPHERE",
                "HEDGE_nd_GRID",
                # "HEDGE_nd_BBX",
                "HEDGE_nd_ELLIP",  # "HEDGE_nd_ELLIP_SAFE",
                # "HEDGE_nd_ELLIP_BBX", #"HEDGE_nd_ELLIP_BBX_SAFE",
                # "HEDGE_nd_ELLIP_BBX_REGUL",
                "HEDGE_nd_BONF",

                # "HORSE_RACE", # simplex
                # "HORSE_RACE_BOUNDED",  # [0,1]^d
                # "HEDGE_nd_MAX",
                # "MULTIPLICATIVE",

            ],

            'parabola_power': None,  # None | 8
            'parabola_adaptive': True,  # will choose power if True
            'parabola_grid_size': 2001,
            'parabola_C_touchpoint': 'automatic',  # automatic | <scalar>
            # div = (D ** (1.0 / power)) -> #touchpointC = C / div

            'cache_hedged': 'EXACT',  # None, 'APPROX', 'EXACT'
            'grid_resolution': 100,  # resolution for grid search in hedged computation and horse racing
            'plot_grid_2d': True,  # code won't crash for higher dims or other methods so can just leave True
            'plot_experiments': True,
            'verbose_progress': True,  # whether to print progress

            'conf_sphere_optimistic_rescale': True,  # whether to use optimistic rescaling in conf sphere.

            'extend_f_domain_for_parabola_fit': True,

            # 'perform_volume_sanity_checks': True,  # whether to perform sanity checks on volume calculations

            'seed': 42,
            'dataset_path': './',  # not really used yet
            'log_path': './logs/cs/',  # not really used yet
            'save_plots_locally': True,
            'dataset': 'NA',
            'run_name': '',
        }
        h = ConfSeqHyperparameters.overwrite_args_cli(h)
        ConfSeqHyperparameters.apply_sanity_checks(h)
        return h

    @staticmethod
    def apply_sanity_checks(h):
        # assert h['batch_size'] >= 5, "batch_size should be at least 5 to avoid too small batches t, z-test, etc."
        print(f"Warning: batch_size={h['batch_size']} is set to {h['batch_size']}, "
              f"this is not recommended for t-test, z-test, etc. ")
        assert h['stop_early_threshold'] > 0, "stop_early_threshold must be a positive float"
        assert 0 < h['alpha'] < 1, "alpha must be in (0, 1)"

        assert DataDistributionChecker.is_legal_distribution(h['data_type']), \
            f"data_type {h['data_type']} is not recognized. Please check available distributions."

        if h['interval_types'] == "HORSE_RACE":
            assert h['cut_to_simplex'] == True

        if h['interval_types'] == "HORSE_RACE_BOUNDED":
            assert h['cut_to_simplex'] == False

        # assert h['alpha'] == 0.05, "see `ParabolaApproximation` lazy coding for why"

        if h['conf_sphere_optimistic_rescale'] and ('CONF_SPHERE' in h['interval_types'] or
                                                    'BANACH_SPHERE' in h['interval_types']):
            # assert the used data types are compatible with optimistic rescaling
            assert DataDistributionChecker.supports_optimistic_rescaling(h['data_type']), \
                (f"Data type {h['data_type']} does not support optimistic rescaling. "
                 f"The confidence sequence would not be valid.")

        assert (isinstance(h['parabola_power'], int) and h['parabola_power'] >= 1) or h['parabola_adaptive'], \
            "parabola_power must be a positive integer >= 1, or parabola_adaptive must be True."

        # ensure if HEDGE_nd_ELLIP_SAFE or HEDGE_nd_ELLIP_BBX_SAFE  is used, then no-non-safe versions are not used (i have shitty code, i know haha
        # its due class ParabolaApproximation:
        #     @staticmethod
        #     def approximate_parabola(h: dict, f: Callable[[float], float], in `f_approximation.py`:
        # line:                 conservative=any("SAFE" in interval for interval in h['interval_types']),
        if any("SAFE" in interval for interval in h['interval_types']):
            for interval in h['interval_types']:
                assert "SAFE" in interval or ("ELLIP" not in interval), \
                    "If using SAFE versions of HEDGE_nd_ELLIP or HEDGE_nd_ELLIP_BBX, cannot use non-SAFE versions."


@init_decorator_without_torch
@timer_decorator
def main(h: dict):
    N = h['N']
    batch_size = h['batch_size']
    alpha = h['alpha']
    data_type = h['data_type']
    stop_early_threshold = h['stop_early_threshold']

    if h['individual_run_experiment']:
        Experiments.individual_run_experiment(h, N, alpha, batch_size, data_type, stop_early_threshold)

    if h['volume_vs_dim_experiment']:
        Experiments.volume_vs_dim_experiment(h)

    if h['volume_vs_p_experiment']:
        Experiments.volume_vs_p_experiment(h)

    if h['volume_vs_C_experiment']:
        Experiments.volume_vs_C_experiment(h)

    if h['volume_vs_parabola_gr_sz_experiment']:
        Experiments.volume_vs_parabola_gr_sz_experiment(h)

    if h['volume_vs_time_experiment']:
        Experiments.volume_vs_time_experiment(h)

    if h['coverage_vs_time_experiment']:
        Experiments.coverage_vs_time_experiment(h)

    if h['perform_l1_err_experiment']:
        Experiments.l1_error_experiment(h, N, alpha, batch_size, data_type, stop_early_threshold)

    if h['perform_coverage_experiment']:
        Experiments.coverage_experiment(h, N, alpha, batch_size, data_type, stop_early_threshold)


if __name__ == "__main__":
    h_ = ConfSeqHyperparameters.get()
    main(h_)

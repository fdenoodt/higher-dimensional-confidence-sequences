import asyncio
import sys

MAIN_PY = "main.py"
MAX_CONCURRENT = 2

methods = [
    # "CONF_SPHERE",
    # "BANACH_SPHERE",
    "HEDGE_nd_BBX", # C^bbox
    # "HEDGE_nd_PAIRBLOCK",
    # # "HEDGE_nd_ELLIP",
    "HEDGE_nd_ELLIP_SAFE", # C^\ell_p safe
    # # "HEDGE_nd_ELLIP_BBX",
    "HEDGE_nd_ELLIP_BBX_SAFE", # C^bbox \ell_p safe
    "HEDGE_nd_BONF", # C^bonf
    # # "HORSE_RACE_BOUNDED"
    "HEDGE_nd_GRID", # C^port
    # "NORMALIZED_ELLIP"
]
data_types = [
    "bernoulli_mix_2d",
    "circle_2d",
]


def _fmt_list(values):
    return "[" + ",".join(str(v) for v in values) + "]"


async def _run_one(cmd, sem):
    async with sem:
        print(f"Executing command: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(*cmd)
        return await proc.wait()


def generate_volume_vs_time_commands():
    # if small_scale:
    #     time_points = [2, 5, 10, 15, 20, 25]
    #     d = 2
    #
    if small_scale:
        time_points = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        d = 2
    else:
        time_points = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                       1000]
        d = 5

    cmds = []
    for method in methods:
        for data_type in data_types:
            cmds.append([
                sys.executable, MAIN_PY,
                "--plot_grid_2d", "False",
                "--plot_experiments", "False",
                "--data_type", data_type,
                "--volume_vs_time_experiment", "True",
                "--volume_vs_time_experiment_time_points", _fmt_list(time_points),
                "--volume_vs_time_experiment_D", str(d),
                "--interval_types", "['%s']" % method,
                "--conf_sphere_optimistic_rescale", "False" if data_type == "bernoulli_mix_2d" else "True",
            ])
    return cmds


def generate_coverage_vs_time_commands():
    if small_scale:
        time_points = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        d = 2
        repeats = 200
    else:
        time_points = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                       1000]
        d = 5
        repeats = 200

    cmds = []
    for method in methods:
        for data_type in data_types:
            cmds.append([
                sys.executable, MAIN_PY,
                "--plot_grid_2d", "False",
                "--plot_experiments", "False",
                "--data_type", data_type,
                "--coverage_vs_time_experiment", "True",
                "--coverage_vs_time_experiment_time_points", _fmt_list(time_points),
                "--coverage_vs_time_experiment_D", str(d),
                "--coverage_vs_time_experiment_repeats", str(repeats),
                "--interval_types", "['%s']" % method,
                "--conf_sphere_optimistic_rescale", "False" if data_type == "bernoulli_mix_2d" else "True",
            ])
    return cmds


def generate_volume_vs_dim_commands():
    cmds = []
    if small_scale:
        volume_vs_dim_max_dim = 10
        volume_vs_dim_n = 100
    else:
        volume_vs_dim_max_dim = 100  # before 25
        volume_vs_dim_n = 1000

    for method in methods:
        for data_type in data_types:
            if method == 'HORSE_RACE_BOUNDED':
                orig_vol = volume_vs_dim_max_dim
                volume_vs_dim_max_dim = 2

            # if method == 'HEDGE_nd_PAIRBLOCK':
            #     volume_vs_dim_max_dim = min(volume_vs_dim_max_dim, 50)

            cmds.append([
                sys.executable, MAIN_PY,
                "--plot_grid_2d", "False",
                "--plot_experiments", "False",
                "--data_type", data_type,
                "--volume_vs_dim_experiment", "True",
                "--volume_vs_dim_experiment_max_dim", str(volume_vs_dim_max_dim),
                "--volume_vs_dim_experiment_N", str(volume_vs_dim_n),
                "--interval_types", "['%s']" % method,
                "--conf_sphere_optimistic_rescale", "False" if data_type == "bernoulli_mix_2d" else "True",
            ])

            # fix volume again:
            if method == 'HORSE_RACE_BOUNDED':
                volume_vs_dim_max_dim = orig_vol

    return cmds


###
# Ablations
###
def generate_volume_vs_p_commands():
    abl_methods = ['HEDGE_nd_ELLIP']
    abl_data_types = ["bernoulli_mix_5d", "circle_5d", "bernoulli_mix_10d", "circle_10d"]
    cmds = []
    n = 1000

    for method in abl_methods:
        for data_type in abl_data_types:
            cmds.append([
                sys.executable, MAIN_PY,
                "--plot_grid_2d", "False",
                "--plot_experiments", "False",
                "--data_type", data_type,

                "--volume_vs_p_experiment", "True",
                "--volume_vs_p_experiment_Ps", _fmt_list(range(1, 21)),
                "--volume_vs_p_experiment_N", str(n),

                "--interval_types", "['%s']" % method,
                "--conf_sphere_optimistic_rescale", "False" if data_type == "bernoulli_mix_2d" else "True",
            ])

    return cmds


def generate_volume_vs_C_commands():
    abl_methods = ['HEDGE_nd_ELLIP']
    abl_data_types = ["bernoulli_mix_5d", "circle_5d", "bernoulli_mix_10d", "circle_10d"]
    cmds = []
    n = 1000  # Observations
    p = 12

    for method in abl_methods:
        for data_type in abl_data_types:
            cmds.append([
                sys.executable, MAIN_PY,
                "--plot_grid_2d", "False",
                "--plot_experiments", "False",
                "--data_type", data_type,

                "--volume_vs_C_experiment", "True",
                "--volume_vs_C_experiment_fixed_parabola_pow", str(p),
                "--volume_vs_C_experiment_N", str(n),

                "--interval_types", "['%s']" % method,
                "--conf_sphere_optimistic_rescale", "False" if data_type == "bernoulli_mix_2d" else "True",
            ])

    return cmds


def generate_volume_vs_parabola_gr_sz_commands():
    abl_methods = ['HEDGE_nd_ELLIP_SAFE']  # NOTE: its safe version here
    abl_data_types = ["bernoulli_mix_5d", "circle_5d", "bernoulli_mix_10d", "circle_10d"]
    cmds = []
    n = 1000  # Observations

    for method in abl_methods:
        for data_type in abl_data_types:
            cmds.append([
                sys.executable, MAIN_PY,
                "--plot_grid_2d", "False",
                "--plot_experiments", "False",
                "--data_type", data_type,

                "--volume_vs_parabola_gr_sz_experiment", "True",
                "--volume_vs_parabola_gr_sz_experiment_grid_sizes",
                str([101, 201, 301, 401, 501, 601, 700, 801, 901, 1001,
                     1101, 1201, 1301, 1401, 1501, 1601, 1701, 1801, 1901,
                     2001, 2101, 2101, 2201, 2301, 2401, 2501, 2601, 2701,
                     2801, 2901, 3001, 3101, 3201, 3301, 3401, 3501, 3601,
                     3701, 3801, 3901, 2001, 2101,
                     ]),
                "--volume_vs_parabola_gr_sz_experiment_N", str(n),

                "--interval_types", "['%s']" % method,
                "--conf_sphere_optimistic_rescale", "False" if data_type == "bernoulli_mix_2d" else "True",
            ])

    return cmds


async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    cmds = []

    # Main experiments
    cmds.extend(generate_volume_vs_time_commands())
    cmds.extend(generate_coverage_vs_time_commands())
    cmds.extend(generate_volume_vs_dim_commands())

    # Ablations for l_p-ellipsoid
    cmds.extend(generate_volume_vs_p_commands())
    cmds.extend(generate_volume_vs_C_commands())
    cmds.extend(generate_volume_vs_parabola_gr_sz_commands())

    if only_print_without_execute:
        for cmd in cmds:
            print(f"Command: {' '.join(cmd)}")
        return

    tasks = [asyncio.create_task(_run_one(cmd, sem)) for cmd in cmds]
    exit_codes = await asyncio.gather(*tasks)
    failures = [i for i, code in enumerate(exit_codes) if code != 0]
    if failures:
        raise SystemExit(f"{len(failures)} commands failed: {failures}")


if __name__ == "__main__":
    small_scale = False

    only_print_without_execute = False
    asyncio.run(main())

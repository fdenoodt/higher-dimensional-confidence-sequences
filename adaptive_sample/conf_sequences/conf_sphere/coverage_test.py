import numpy as np
import time
import math
import matplotlib.pyplot as plt
from typing import Callable, Tuple

from adaptive_sample.conf_sequences.conf_sphere import EmpiricalBernsteinConfSphere
from tqdm import tqdm  # Add this import at the top if not already present
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------- Monte-Carlo simultaneous coverage test ----------
def empirical_coverage_test(
        ConfSphereClass: type,
        sample_fn: Callable[[int, int, int], np.ndarray],
        T: int = 200,
        d: int = 2,
        n_trials: int = 2000,
        alpha: float = 0.05,
        rng_seed: int = 1234,
        verbose: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Run n_trials Monte-Carlo experiments and measure simultaneous coverage
    of the confidence sequence produced by ConfSphereClass.

    Returns:
      empirical_coverage: fraction of trials where the CS covered the true mean for all t=1..T
      survival: array length T with fraction of trials that still contain mean at time t
    """
    rng = np.random.RandomState(rng_seed)
    survival_counts = np.zeros(T, dtype=int)

    # Prepare hyperparams dictionary expected by your class
    h_template = {
        'alpha': alpha,
        'batch_size': 1,
        'data_type': f'fixed_{d}d',
        'N': T,
        'cut_to_simplex': False,
        'plot_grid_2d': False,
        'calculate_volume': False,
        'verbose_progress': False,
        'D': d
    }

    def run_trial(trial_seed: int) -> Tuple[bool, np.ndarray]:
        """Run a single trial and return whether it covered all t and survival counts."""
        trial_rng = np.random.RandomState(trial_seed)
        X = sample_fn(N=T, d=d, seed=trial_rng.randint(2 ** 31 - 1))
        true_mean = X.mean(axis=0)

        h = dict(h_template)
        cs = ConfSphereClass(h, X)
        covered_all_t = True
        trial_survival_counts = np.zeros(T, dtype=int)

        for t in range(1, T + 1):
            cs.update(t, X[t - 1:t])
            mu_hat_scaled, r_scaled = cs._rebuild_state_up_to(t)
            mu_hat_orig = mu_hat_scaled * cs.rescale_factor + cs.box_center
            r_orig = float(r_scaled * cs.rescale_factor)

            dist = np.linalg.norm(true_mean - mu_hat_orig)
            is_in = (dist <= (r_orig + 1e-12))
            if is_in:
                trial_survival_counts[t - 1] += 1
            else:
                covered_all_t = False

        return covered_all_t, trial_survival_counts

    start = time.time()
    successes = 0

    # Use ThreadPoolExecutor for parallel trials
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_trial, rng.randint(2 ** 31 - 1)): i for i in range(n_trials)}
        for future in tqdm(as_completed(futures), total=n_trials, desc="Trials", unit="trial"):
            covered_all_t, trial_survival_counts = future.result()
            if covered_all_t:
                successes += 1
            survival_counts += trial_survival_counts

    empirical_coverage = successes / float(n_trials)
    survival = survival_counts.astype(float) / float(n_trials)
    duration = time.time() - start

    if verbose:
        print(
            f"Done: trials={n_trials}, T={T}, empirical simultaneous coverage = {empirical_coverage:.4f} (α={alpha}), time={duration:.1f}s")
    return empirical_coverage, survival


# ---------- example sample function (you already have several) ----------
def sample_mixture_1(N, d, seed=None):
    rng = np.random.RandomState(seed)
    W = rng.binomial(1, 0.5, size=(N,))
    X = np.zeros((N, d))
    for i in range(N):
        if W[i] == 1:
            X[i] = rng.beta(2, 8, size=(d,))
        else:
            # Bernoulli scaled to [0,1] (0 or 1)
            X[i] = rng.binomial(1, 0.6, size=(d,))
    return X


def sample_fixed_2d(N, d, seed=None):
    return np.ones((N, d)) * [0.7, 0.3]


# ---------- example main runner ----------
if __name__ == "__main__":
    # EmpiricalBernsteinConfSphere | BanachSphere
    CS_CLASS = EmpiricalBernsteinConfSphere  # replace with your class name

    # parameters
    T = 1_000  # horizon to check
    d = 2
    N_TRIALS = 1000  # 1000 or more for stability; use fewer if slow
    ALPHA = 0.05

    cov, survival = empirical_coverage_test(
        CS_CLASS,
        sample_mixture_1,  # sample_mixture_1 | sample_fixed_2d
        T=T,
        d=d,
        n_trials=N_TRIALS,
        alpha=ALPHA,
        rng_seed=41,
        verbose=True
    )

    # Plot survival curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, T + 1), survival, label="Empirical survival (coverage at t)")
    plt.axhline(1 - ALPHA, color="C1", linestyle="--", label=f"Target 1-α = {1 - ALPHA:.2f}")
    plt.xlabel("t")
    plt.ylabel("Fraction of trials with mean inside CS at time t")
    plt.title(f"Empirical survival (simultaneous coverage) over {N_TRIALS} trials\nSimultaneous coverage ≈ {cov:.3f}")
    plt.legend()
    plt.grid(True)
    plt.show()

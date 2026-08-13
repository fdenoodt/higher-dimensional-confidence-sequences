import re
from enum import Enum
from typing import Tuple, Any

import numpy as np
from scipy.special import ndtr
from scipy.stats import beta as sp_beta


HIGHLY_CORRELATED_RHO = 0.98


def generate_samples(D: int, N: int, radii_per_dim: np.ndarray,
                     center: np.ndarray, pow: int = 2) -> (
        Tuple)[np.ndarray, np.ndarray]:
    assert pow >= 2 and pow % 2 == 0, "Only even values of p >= 2 are supported."

    assert pow > 0, "pow must be positive"
    radii_per_dim = np.asarray(radii_per_dim).reshape(1, D)
    center = np.asarray(center).reshape(1, D)

    # 1) sample directions on the L_pow sphere
    if pow == 2:
        # special-case: standard normal -> directions on Euclidean sphere (fast)
        vecs = np.random.randn(N, D)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
    else:
        # general-case: sample coordinates X with density ~ exp(-|x|^p)
        # |X|^p ~ Gamma(shape=1/p, scale=1), sign random
        shape = 1.0 / float(pow)
        T = np.random.gamma(shape=shape, scale=1.0, size=(N, D))  # T = |X|^p
        R = T ** (1.0 / float(pow))  # |X|
        signs = np.random.choice([-1.0, 1.0], size=(N, D))
        vecs = signs * R
        # normalize by L_p norm to lie on the L_p *sphere*
        lp_norms = (np.sum(np.abs(vecs) ** float(pow), axis=1, keepdims=True)) ** (1.0 / float(pow))
        vecs = vecs / lp_norms

    # 2) uniform radius in the ball (volume scales as r^D for any p)
    base_radii = np.random.rand(N) ** (1.0 / D)  # in [0,1], density ~ r^{D-1}
    scaled_dirs = vecs * base_radii[:, None]

    # 3) scale per-dimension by radii and translate by center
    samples = center + scaled_dirs * radii_per_dim

    true_mean = center.flatten()
    return samples, true_mean


class DistributionType(Enum):
    UNIFORM = "uniform"
    UNIFORM2 = "uniform2"
    GAUSSIAN = "gaussian"
    BETA = "beta"
    DIRICHLET = "dirichlet"
    FIXED = "fixed"

    # higher dimensional
    FIXED_ND = "fixed_nd"
    UNIFORM_ND = "uniform_nd"  # Doesn't sum to 1
    HIGHLY_CORRELATED_ND = "highly_correlated_nd"  # Doesn't sum to 1
    CIRCLE_ND = "circle_nd"  # points on the surface of a unit circle
    ELLIPSE_ND = "ellipse_nd"  # points on the surface of an ellipse
    BERNOULLI_ND = "bernoulli_nd"  # Bernoulli samples with p=0.5 in {0, 1}
    BERNOULLI_MIX_ND = "bernoulli_mix_nd"  # Mix of Bernoulli and shifted Bernoulli


class DataDistributionChecker:
    @staticmethod
    def dim(h: dict) -> int:
        """Return the number of output dimensions for the distribution described by h."""
        distribution_type, args, dim = DataDistributionChecker.check_distribution(h["data_type"])
        return dim

    @staticmethod
    def check_distribution(type_str: str) -> Tuple[DistributionType, Any, int]:
        """Parse a distribution description string and return a tuple
        ret: (DistributionType, args, dim)
        """

        if type_str == "uniform":
            return DistributionType.UNIFORM, [0.5, 0.8], 1

        if type_str == "uniform2":
            return DistributionType.UNIFORM2, [0.0, 1.0], 1

        if type_str == "fixed":
            return DistributionType.FIXED, [0.7], 1

        # fixed_Nd: matches e.g. "fixed_3d" -> d=3
        m = re.match(r"^fixed_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            # create a mean vector that sums to 1 with a dominant first dimension
            if nb_dims == 1:
                mean_vector = [0.7]
            else:
                mean_dim1 = 0.7
                remaining = (1.0 - mean_dim1) / max(1, nb_dims - 1)
                mean_vector = [mean_dim1] + [remaining] * (nb_dims - 1)
            return DistributionType.FIXED_ND, mean_vector, nb_dims

        if type_str == "gaussian":
            return DistributionType.GAUSSIAN, [0.0, 1.0], 1

        # beta: beta_a=1.0_b=2.0  (both floats)
        m = re.match(r"^beta_a=(?P<a>-?[0-9.]+(?:\.[0-9]+)?)_b=(?P<b>-?[0-9.]+(?:\.[0-9]+)?)$", type_str)
        if m:
            a = float(m.group("a"))
            b = float(m.group("b"))
            return DistributionType.BETA, [a, b], 1

        # dirichlet with repetition: dirichlet_a=0.5_0.3*10  -> base=0.5_0.3, multiplier=10
        m = re.match(r"^dirichlet_a=(?P<base>[0-9._-]+)\*(?P<mult>\d+)$", type_str)
        if m:
            base_values = m.group("base")
            multiplier = int(m.group("mult"))
            a = [float(x) for x in base_values.split("_")]
            a = a * multiplier
            return DistributionType.DIRICHLET, a, len(a)

        # dirichlet without repetition: dirichlet_a=0.5_0.3_0.2
        m = re.match(r"^dirichlet_a=(?P<vals>[0-9._-]+)$", type_str)
        if m:
            a = [float(x) for x in m.group("vals").split("_")]
            return DistributionType.DIRICHLET, a, len(a)

        # highly_correlated_Nd: highly_correlated_3d
        m = re.match(r"^highly_correlated_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            mean_vector = [0.5] * nb_dims
            return DistributionType.HIGHLY_CORRELATED_ND, mean_vector, nb_dims

        # uniform_nd: support both "uniform_nd_3d" and "uniform_3d" for compatibility
        m = re.match(r"^uniform(?:_nd)?_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            # we return the dimension as args (keeps compatibility with the old code)
            return DistributionType.UNIFORM_ND, nb_dims, nb_dims

        # circle_nd: circle_3d
        m = re.match(r"^circle_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            return DistributionType.CIRCLE_ND, nb_dims, nb_dims

        # ellipse_nd: ellipse_3d
        m = re.match(r"^ellipse_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            return DistributionType.ELLIPSE_ND, nb_dims, nb_dims

        # bernoulli_nd: bernoulli_3d
        m = re.match(r"^bernoulli_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            return DistributionType.BERNOULLI_ND, nb_dims, nb_dims

        # bernoulli_mix_nd: bernoulli_mix_3d
        m = re.match(r"^bernoulli_mix_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            return DistributionType.BERNOULLI_MIX_ND, nb_dims, nb_dims

        raise ValueError(f"Unknown distribution type: {type_str}")

    @staticmethod
    def get_data(type_str: str, N: int) -> Tuple[np.ndarray, Any]:
        """Generate N samples according to the distribution described by type_str.

        Returns (samples, true_mean).
        """
        distribution_type, args, dim = DataDistributionChecker.check_distribution(type_str)

        if distribution_type == DistributionType.UNIFORM:
            lo, hi = args[0], args[1]
            true_mean = 0.5 * (lo + hi)
            samples = np.random.rand(N) * (hi - lo) + lo
            return samples, true_mean

        if distribution_type == DistributionType.UNIFORM2:
            # uniform on [0,1]
            true_mean = 0.5
            return np.random.rand(N), true_mean

        if distribution_type == DistributionType.FIXED:
            val = args[0]
            return np.full(N, val), val

        if distribution_type == DistributionType.FIXED_ND:
            # args is the mean vector
            mean_vector = np.asarray(args, dtype=float)
            samples = np.tile(mean_vector, (N, 1))
            true_mean = mean_vector
            return samples, true_mean

        if distribution_type == DistributionType.GAUSSIAN:
            mu, sigma = args[0], args[1]
            return np.random.normal(loc=mu, scale=sigma, size=N), mu

        if distribution_type == DistributionType.BETA:
            a, b = float(args[0]), float(args[1])
            true_mean = a / (a + b)
            data = sp_beta.rvs(a=a, b=b, size=N)
            np.random.shuffle(data)
            return data, true_mean

        if distribution_type == DistributionType.DIRICHLET:
            a = list(args)
            data = np.random.dirichlet(a, size=N)
            true_mean = np.array(a) / float(np.sum(a))
            # shuffle along the first axis if needed
            idx = np.arange(len(data))
            np.random.shuffle(idx)
            return data[idx], true_mean

        if distribution_type == DistributionType.HIGHLY_CORRELATED_ND:
            # Gaussian copula with equicorrelated latent variables.  Applying
            # the standard-normal CDF leaves every coordinate marginally
            # Uniform[0, 1], while rho close to one concentrates observations
            # tightly around the diagonal.
            mean_vector = np.asarray(args, dtype=float)
            common = np.random.randn(N, 1)
            independent = np.random.randn(N, dim)
            latent = (
                np.sqrt(HIGHLY_CORRELATED_RHO) * common
                + np.sqrt(1.0 - HIGHLY_CORRELATED_RHO) * independent
            )
            samples = ndtr(latent)
            true_mean = mean_vector
            return samples, true_mean

        if distribution_type == DistributionType.UNIFORM_ND:
            d = int(args)
            samples = np.random.rand(N, d)
            true_mean = np.ones(d) * 0.5
            return samples, true_mean

        if distribution_type == DistributionType.CIRCLE_ND or distribution_type == DistributionType.ELLIPSE_ND:
            D = int(args)

            samples, true_mean = generate_samples(
                D, N,
                radii_per_dim=np.array([0.5] + [0.25 / 2] * (D - 1))
                if distribution_type == DistributionType.ELLIPSE_ND else np.full(D, 0.5),
                center=np.full(D, 0.5)
            )

            return samples, true_mean

        if distribution_type == DistributionType.BERNOULLI_ND:
            d = int(args)
            # Bernoulli samples with p=0.5, giving values in {0, 1}
            samples = np.random.binomial(n=1, p=0.5, size=(N, d)).astype(float)
            true_mean = np.ones(d) * 0.5
            return samples, true_mean

        if distribution_type == DistributionType.BERNOULLI_MIX_ND:
            d = int(args)
            samples = np.zeros((N, d))
            # First half: standard Bernoulli(p=0.5) in {0, 1}
            half_d = d // 2
            samples[:, :half_d] = np.random.binomial(n=1, p=0.5, size=(N, half_d)).astype(float)
            # Second half: Bernoulli(p=0.5) shifted to [0.5, 1]
            # If 0 -> 0.5, if 1 -> 1.0
            samples[:, half_d:] = (np.random.binomial(n=1, p=0.5, size=(N, d - half_d)).astype(float) + 1.0) / 2.0
            # True mean: 0.5 for first half, 0.75 for second half
            true_mean = np.concatenate([np.ones(half_d) * 0.5, np.ones(d - half_d) * 0.75])
            return samples, true_mean

        raise ValueError(f"Unknown data type: {type_str}")

    @staticmethod
    def supports_optimistic_rescaling(type_str: str) -> bool:
        """Returns True if the distribution type supports optimistic rescaling."""
        distribution_type, args, dim = DataDistributionChecker.check_distribution(type_str)

        if distribution_type in {
            DistributionType.FIXED_ND,
            DistributionType.CIRCLE_ND,
            DistributionType.ELLIPSE_ND,
        }:
            return True

        return False

    @staticmethod
    def is_legal_distribution(type_str: str) -> bool:
        """Returns True if the distribution type is legal/supported."""
        try:
            DataDistributionChecker.check_distribution(type_str)
            return True
        except ValueError:
            return False


class StaticRunTracker:

    @staticmethod
    def get_run_number() -> int:
        """
        Returns the current run number.
        This is a static method that can be called without an instance.
        """
        if not hasattr(StaticRunTracker, "_run_number"):
            StaticRunTracker._run_number = 0
        return StaticRunTracker._run_number

    @staticmethod
    def increment_run_number():
        """
        Increments the run number by 1.
        This is a static method that can be called without an instance.
        """
        if not hasattr(StaticRunTracker, "_run_number"):
            StaticRunTracker._run_number = 0
        StaticRunTracker._run_number += 1

    @staticmethod
    def reset_run_number():
        """
        Resets the run number to 0.
        This is a static method that can be called without an instance.
        """
        StaticRunTracker._run_number = 0


class TikzChecker:
    @staticmethod
    def is_tikz_installed() -> bool:
        """
        Checks if TikZ is installed on the system.
        Returns True if TikZ is installed, False otherwise.
        """
        try:
            import subprocess
            result = subprocess.run(['pdflatex', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0
        except Exception as e:
            print(f"Error checking TikZ installation: {e}")
            return False


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def plot_2d_data(type_str: str):
        # Parameters
        N = 10_000  # Number of samples

        # Retrieve data
        samples, true_mean = DataDistributionChecker.get_data(type_str, N)

        # samples, true_mean = generate_samples(
        #     D=2,
        #     N=N,
        #     radii_per_dim=np.array([0.5, 0.25]),
        #     # radii_per_dim=np.array([0.25, 0.25]),
        #     center=np.array([0.5, 0.5]),
        #     pow=4
        # )

        # Plot the data
        plt.figure(figsize=(8, 6))
        plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.6, label="Samples")
        plt.scatter(true_mean[0], true_mean[1], color="red", label="True Mean", zorder=5)
        plt.title(f"2D Data Plot ({type_str})")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()


    if __name__ == "__main__":
        # uniform_2d | uniform_nd_2d | highly_correlated_nd_2d | fixed_nd_2d | circle_2d | ellipse_2d | bernoulli_2d | bernoulli_mix_2d
        plot_2d_data("bernoulli_mix_2d")

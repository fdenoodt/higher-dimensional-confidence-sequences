import numpy as np

from adaptive_sample.shared_code import (
    HIGHLY_CORRELATED_RHO,
    DataDistributionChecker,
)


def test_highly_correlated_distribution_is_bounded_and_correlated():
    state = np.random.get_state()
    np.random.seed(20260726)
    try:
        samples, true_mean = DataDistributionChecker.get_data(
            "highly_correlated_4d", 50_000
        )
    finally:
        np.random.set_state(state)

    assert samples.shape == (50_000, 4)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)
    np.testing.assert_allclose(true_mean, np.full(4, 0.5))
    np.testing.assert_allclose(samples.mean(axis=0), true_mean, atol=0.01)

    correlations = np.corrcoef(samples, rowvar=False)
    off_diagonal = correlations[np.triu_indices(4, k=1)]
    assert np.all(off_diagonal > 0.95)
    assert HIGHLY_CORRELATED_RHO == 0.98


def test_highly_correlated_coordinates_retain_full_uniform_range():
    state = np.random.get_state()
    np.random.seed(20260727)
    try:
        samples, _ = DataDistributionChecker.get_data(
            "highly_correlated_3d", 100_000
        )
    finally:
        np.random.set_state(state)

    assert np.all(samples.min(axis=0) < 0.01)
    assert np.all(samples.max(axis=0) > 0.99)

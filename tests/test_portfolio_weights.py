import numpy as np
import pytest

from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_convex_sum import (
    _ellipse_params_from_hparams,
    _portfolio_weights,
)
from adaptive_sample.helper.ellipsoid_tools import EllipsoidTools


def test_default_weights_are_uniform():
    np.testing.assert_allclose(_portfolio_weights({}, 4), np.full(4, 0.25))


@pytest.mark.parametrize(
    "weights",
    [
        [0.5, 0.5, 0.0],
        [0.5, 0.5, -1e-3],
        [0.5, np.nan, 0.5],
        [0.2, 0.2, 0.2],
        [0.5, 0.5],
    ],
)
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        _portfolio_weights({"portfolio_weights": weights}, 3)


def test_equal_weight_ellipsoid_matches_original_formula():
    hparams = np.array(
        [
            [0.4, 0.10, 0.7],
            [0.6, 0.15, 0.9],
            [0.5, 0.20, 1.1],
        ]
    )
    C = 20.0
    power = 8
    expected = EllipsoidTools.get_ellipse_params_from_parabola(
        hparams[:, 0].tolist(),
        hparams[:, 1].tolist(),
        hparams[:, 2].tolist(),
        Beta=C,
        D=3,
        pow=power,
    )
    actual = _ellipse_params_from_hparams(
        C, hparams, np.full(3, 1.0 / 3.0), power
    )
    np.testing.assert_allclose(actual, expected)


def test_weighted_ellipsoid_axes_lie_on_weighted_boundary():
    hparams = np.array(
        [
            [0.4, 0.10, 0.7],
            [0.6, 0.15, 0.9],
            [0.5, 0.20, 1.1],
        ]
    )
    weights = np.array([0.6, 0.3, 0.1])
    C = 20.0
    power = 8
    ellipse = _ellipse_params_from_hparams(C, hparams, weights, power)

    for axis in range(3):
        point = hparams[:, 0].copy()
        point[axis] += ellipse[axis, 1]
        fitted_wealths = (
            np.abs(point - hparams[:, 0]) ** power / hparams[:, 1] ** power
            - hparams[:, 2]
        )
        assert np.dot(weights, fitted_wealths) == pytest.approx(C)

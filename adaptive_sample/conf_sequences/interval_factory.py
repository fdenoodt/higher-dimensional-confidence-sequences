from typing import Tuple

from adaptive_sample.conf_sequences import BaseConfidenceSequence
from adaptive_sample.conf_sequences.conf_sphere import BanachSphere
from adaptive_sample.conf_sequences.conf_sphere.chugg_conf_sphere import EmpiricalBernsteinConfSphere
from adaptive_sample.conf_sequences.conf_sphere.selfnormalized_conf_ellipsoid import NormalizedConfEllipsoid
from adaptive_sample.conf_sequences.hedged_capital import HedgedCapitalCI, HedgedCapitalConvexSumGrid, \
    HedgedCapitalConvexSumBoundingBox, HedgedCapitalConvexSumEllipsoid, HedgedCapitalConvexSumBBxEllip, \
    HedgedCapitalCIMultiDimensionalBonferroni
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_1d import CachedHedgedCapitalCI, \
    ExactCachedHedgedCapitalCI
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_max import \
    HedgedCapitalCIMultiDimensionalMax
from adaptive_sample.conf_sequences.horse_race import UniversalPortfolioCS, BoundedVectorUPCS


class IntervalFactory:
    """
    Factory class to create confidence intervals.
    """

    @staticmethod
    def create_instance(name: str, h: dict, data) -> Tuple[BaseConfidenceSequence, str, str]:
        interval_map = { # lambda for lazy evaluation
            "HEDGE_CI": hedged_capital_1d(h, data),
            "HEDGE_nd_GRID": lambda: (HedgedCapitalConvexSumGrid(h, data), "Hedged (grid)", "pink"),
            "HEDGE_nd_BBX": lambda: (HedgedCapitalConvexSumBoundingBox(h, data), "Hedged (BBX)", "black"),

            "HEDGE_nd_ELLIP": lambda: (HedgedCapitalConvexSumEllipsoid(h, data), "Hedged (ELLIP)", "blue"),
            "HEDGE_nd_ELLIP_BBX": lambda: (HedgedCapitalConvexSumBBxEllip(h, data), "Hedged (ELLIP + BBX)", "green"),

            "HEDGE_nd_ELLIP_SAFE": lambda: (HedgedCapitalConvexSumEllipsoid(h, data), "Hedged (ELLIP-S)", "lightblue"),
            "HEDGE_nd_ELLIP_BBX_SAFE": lambda: (HedgedCapitalConvexSumBBxEllip(h, data), "Hedged (ELLIP-S + BBX)", "lightgreen"),

            "HEDGE_nd_MAX": lambda: (HedgedCapitalCIMultiDimensionalMax(h, data), "Hedged (max)", "purple"),
            "HEDGE_nd_BONF": lambda: (HedgedCapitalCIMultiDimensionalBonferroni(h, data), "Hedged (Bonferroni)", "orange"),
            #
            "HORSE_RACE": lambda: (UniversalPortfolioCS(h, data), "UP", "red"),
            "HORSE_RACE_BOUNDED": lambda: (BoundedVectorUPCS(h, data), "Bounded UP", "brown"),
            #
            "CONF_SPHERE": lambda: (EmpiricalBernsteinConfSphere(h, data), "Conf Sphere", "cyan"),
            "BANACH_SPHERE": lambda: (BanachSphere(h, data), "Banach Sphere", "magenta"),
            "NORMALIZED_ELLIP": lambda: (NormalizedConfEllipsoid(h, data), "Normalized Conf Ellipsoid", "green"),
        }

        if name not in interval_map:
            raise ValueError(f"Unknown confidence interval type: {name}")

        return interval_map[name]()

def hedged_capital_1d(h, data):
    color = "pink"
    if h['cache_hedged'] == None:
        return lambda: (HedgedCapitalCI(h, data), "Hedged", color)
    elif h['cache_hedged'] == 'APPROX':
        return lambda: (CachedHedgedCapitalCI(h, data), "Hedged (cached)", color)
    elif h['cache_hedged'] == 'EXACT':
        return lambda: (ExactCachedHedgedCapitalCI(h, data), "Hedged (exact)", color)
    else:
        raise ValueError(f"Unknown hedged capital caching option: {h['cache_hedged']}. "
                         f"Expected None, 'APPROX', or 'EXACT'.")

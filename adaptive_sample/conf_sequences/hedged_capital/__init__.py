from typing import Type, List

import numpy as np

from shared.decorators import timer_decorator
from adaptive_sample.conf_sequences import BaseConfidenceSequence
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_1d import HedgedCapitalCI, \
    ExactCachedHedgedCapitalCI, CachedHedgedCapitalCI
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_bonf import \
    HedgedCapitalCIMultiDimensionalBonferroni
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_convex_sum import HedgedCapitalConvexSum, \
    HedgedCapitalConvexSumBBxEllip, HedgedCapitalConvexSumEllipsoid, HedgedCapitalConvexSumBoundingBox, \
    HedgedCapitalConvexSumGrid


@timer_decorator
def main(cs_class: Type[BaseConfidenceSequence]):
    # set fixed seed etc
    np.random.seed(42)

    h = {
        'alpha': 0.05,
        'batch_size': 1,
        'data_type': 'fixed_2d',
        'N': 5,
        'cut_to_simplex': False,
        'grid_resolution': 25,
        'plot_grid_2d': True,
        'calculate_volume': False,
        'hedged_computation': 'grid_search',
        'save_plots_locally': False,
        'use_wandb': False,
    }
    # data = np.array([[0.5, 0.5]] * h['N'])
    data = np.array([[0.7, 0.3]] * h['N'])
    cs = cs_class(h, data)  # e.g. HedgedCapitalCIMultiDimensional

    ts, means, lowers, uppers = cs.run()

    # for l, u in zip(lowers, uppers):
    #     print(f"[{l[0]:2f} - {u[0]:2f}], [{l[1]:2f} - {u[1]:2f}]")

    return lowers, uppers


if __name__ == "__main__":
    def _santity_test():
        lwH, upH = main(cs_class=HedgedCapitalConvexSumGrid)
        lwB, upB = main(cs_class=HedgedCapitalCIMultiDimensionalBonferroni)

        d = 0
        lwH_d = lwH[:, d]
        upH_d = upH[:, d]

        lwB_d = lwB[:, d]
        upB_d = upB[:, d]

        # Only look at lower values
        for t in range(len(lwH_d)):
            print(f"t={t}: {lwH_d[t]:.4f} >= {lwB_d[t]:.4f} ? {lwH_d[t] >= lwB_d[t]} -- {lwH_d[t] - lwB_d[t]:.4f}")

        # assert bonferoni always has looser bounds than max
        assert np.all(lwB <= lwH), "Bonferroni lower bounds should be looser than max bounds"
        assert np.all(upB >= upH), "Bonferroni upper bounds should be looser than max bounds"
        print("Sanity test passed! \n")


    _santity_test()

    for cs_class in [
        HedgedCapitalConvexSumGrid,
        HedgedCapitalConvexSumBoundingBox,
        HedgedCapitalConvexSumEllipsoid,
        HedgedCapitalConvexSumBBxEllip  # intersection of bounding box and ellipsoid
    ]:
        print(f"\nRunning with {cs_class.__name__}")
        lw, up = main(cs_class=cs_class)
        # print widths
        widths = up - lw
        print(f"Widths for {cs_class.__name__}:")
        for t, width in enumerate(widths):
            print(f"\tt={t}: {width[0]:.4f}, {width[1]:.4f}")  # 2 dimensions

from typing import List

import numpy as np

from adaptive_sample.conf_sequences.hedged_capital import HedgedCapitalCI, ExactCachedHedgedCapitalCI, \
    CachedHedgedCapitalCI


class HedgedFactory:
    @staticmethod
    def build_and_cache_Khedgeds(h: dict, data: np.ndarray) -> List[HedgedCapitalCI]:
        """Create 1-D HedgedCapitalCI calculators and run them on each coordinate."""
        D = data.shape[1]

        if h['cache_hedged'] == None:
            Kd_calculators = [HedgedCapitalCI(h, data[:, d]) for d in range(D)]
        elif h['cache_hedged'] == 'EXACT':
            Kd_calculators = [ExactCachedHedgedCapitalCI(h, data[:, d]) for d in range(D)]
        elif h['cache_hedged'] == 'APPROX':
            Kd_calculators = [CachedHedgedCapitalCI(h, data[:, d]) for d in range(D)]
        else:
            raise ValueError(f"Unknown hedged capital caching method: {h['cache_hedged']}")

        # for d, calc in enumerate(Kd_calculators):
        #     calc.run()  # TODO: only reason we have this run is this run for calc.means i think
        return Kd_calculators

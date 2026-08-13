import math
import numpy as np
from adaptive_sample.helper.hypercube import HypercubeGrid
from adaptive_sample.helper.simplex import Simplex


class GridCSHelper:
    @staticmethod
    def confidence_set(h: dict, is_in_cs: callable, D: int, grid_resolution=50):
        # Generate grid points in probability simplex
        if h['cut_to_simplex']:
            grid = Simplex.create_simplex_grid(D, grid_resolution)
        else:
            grid = HypercubeGrid.create_grid(D, grid_resolution)

        in_cs = [is_in_cs(m) for m in grid]
        return grid, in_cs

    @staticmethod
    def compute_cs_volume(h: dict, grid, in_cs, D: int):
        """Compute volume of confidence set via grid approximation"""

        if not h['calculate_volume']:
            return 0.0

        if h['cut_to_simplex']:
            # Volume = fraction of points in CS times simplex volume
            volume_ratio = np.mean(in_cs)
            simplex_vol = np.sqrt(D) / math.factorial(D - 1)
            return volume_ratio * simplex_vol
        else:
            # For hypercube, volume is simply the fraction of points in CS
            return np.mean(in_cs)

import numpy as np
from functools import lru_cache


class HypercubeGrid:
    """
    Generate a regular grid on [0,1]^D with points spaced by 1/(resolution-1)
    along each axis.

    Usage:
        grid = HypercubeGrid.create_grid(D=3, resolution=5)
        # Clear cache:
        HypercubeGrid.create_grid.cache_clear()

    Notes:
    - The number of points = resolution**D, so memory grows fast. Use a lazy
      iterator (create_grid_iter) or sampling for large D/resolution.
    - resolution must be >= 2 (so denominators are > 0).
    """

    @staticmethod
    @lru_cache(maxsize=None)
    def create_grid(D: int, resolution: int) -> np.ndarray:
        if D < 1:
            raise ValueError("D must be >= 1")
        if resolution < 2:
            raise ValueError("resolution must be >= 2")

        denom = resolution - 1

        # Use numpy.indices to build integer grid, reshape to (n_points, D)
        # indices has shape (D, resolution, resolution, ..., resolution)
        # .reshape(D, -1).T -> shape (resolution**D, D)
        idx = np.indices((resolution,) * D, dtype=np.int64)
        flat = idx.reshape(D, -1).T.astype(float)  # shape (n_points, D)

        grid = flat / float(denom)  # convert to [0,1] multiples

        # mark as read-only to protect cached object
        grid.setflags(write=False)
        return grid

    @staticmethod
    def create_grid_iter(D: int, resolution: int):
        """
        Lazy iterator over the grid (not cached). Yields rows of length D.
        Useful when resolution**D is too large to keep in memory.
        """
        if D < 1:
            raise ValueError("D must be >= 1")
        if resolution < 2:
            raise ValueError("resolution must be >= 2")

        denom = resolution - 1
        # produce indices with nested loops in lexicographic order
        # equivalent to itertools.product(range(resolution), repeat=D)
        import itertools
        for comb in itertools.product(range(resolution), repeat=D):
            yield np.array(comb, dtype=float) / float(denom)

if __name__ == "__main__":
    # Example usage
    grid = HypercubeGrid.create_grid(D=3, resolution=5)
    print("Hypercube grid points (D=3, resolution=5):")
    print(grid)

    # Example of lazy iterator
    print("\nHypercube grid points (lazy iterator):")
    for point in HypercubeGrid.create_grid_iter(D=2, resolution=3):
        print(point)

    # Example usage
    grid = HypercubeGrid.create_grid(D=1, resolution=5)
    print("Hypercube grid points (D=1, resolution=5):")
    print(grid)
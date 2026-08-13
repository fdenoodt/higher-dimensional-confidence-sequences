import numpy as np
from functools import lru_cache
from itertools import combinations

class Simplex:
    """
    Create barycentric grids on the (D-1)-simplex with caching.

    Usage:
        Simplex.create_simplex_grid(3, 5)
        Simplex.create_simplex_grid.cache_clear()  # clear cache if needed
    """

    @staticmethod
    @lru_cache(maxsize=None)
    def create_simplex_grid(D: int, resolution: int):
        """Create grid points in the probability simplex (cached)."""
        if resolution < 2:
            raise ValueError("resolution must be >= 2 (so denominator resolution-1 > 0)")
        if D == 2:
            arr = Simplex._line_grid(resolution)
        elif D == 3:
            arr = Simplex._triangle_grid(resolution)
        elif D == 4:
            arr = Simplex._tetrahedron_grid(resolution)
        elif D > 4:
            arr = Simplex._barycentric_grid(D, resolution)
        else:
            raise ValueError("D must be >= 2")

        # guard: clip small negative numerical noise, renormalize rows exactly to sum to 1
        arr = np.array(arr, dtype=float)  # ensure float copy
        tol = 1e-12
        # clip tiny negatives to zero, and tiny >1 to 1
        arr = np.clip(arr, 0.0, 1.0)
        # renormalize row-wise (avoid division by zero: any-zero rows shouldn't exist)
        row_sums = arr.sum(axis=1, keepdims=True)
        # If row_sums has tiny deviation from 1, fix by dividing (should be safe)
        arr = arr / row_sums

        # mark cached array as read-only to avoid accidental in-place mutation
        arr.setflags(write=False)
        return arr

    @staticmethod
    def _line_grid(resolution: int):
        """Grid for 1-simplex (line segment)"""
        denom = resolution - 1
        xs = np.arange(resolution) / denom
        pts = np.stack([xs, 1 - xs], axis=1)
        return pts

    @staticmethod
    def _triangle_grid(resolution: int):
        """Grid for 2-simplex (triangle)"""
        denom = resolution - 1
        points = []
        for i in range(resolution):
            for j in range(resolution - i):
                m1 = i / denom
                m2 = j / denom
                m3 = 1.0 - m1 - m2
                points.append([m1, m2, m3])
        return np.array(points)

    @staticmethod
    def _tetrahedron_grid(resolution: int):
        """Grid for 3-simplex (tetrahedron)"""
        denom = resolution - 1
        points = []
        for i in range(resolution):
            for j in range(resolution - i):
                for k in range(resolution - i - j):
                    m1 = i / denom
                    m2 = j / denom
                    m3 = k / denom
                    m4 = 1.0 - m1 - m2 - m3
                    points.append([m1, m2, m3, m4])
        return np.array(points)

    @staticmethod
    @lru_cache(maxsize=None)
    def _barycentric_grid(D: int, resolution: int):
        """
        Faster generic barycentric grid via combinations (stars-and-bars).
        Returns an array of shape (n_points, D).
        Cached independently as well.
        """
        denom = resolution - 1
        if denom < 1:
            # resolution==1 handled earlier, but keep guard
            return np.array([[1.0] + [0.0] * (D - 1)])

        # number of slots to place (denom stars + D-1 bars) = denom + D - 1
        Nslots = denom + D - 1
        pts = []
        # choose positions of the (D-1) bars among Nslots positions
        # each combination of bar positions maps uniquely to a nonnegative integer solution
        for bars in combinations(range(Nslots), D - 1):
            prev = -1
            counts = []
            for b in bars:
                counts.append(b - prev - 1)
                prev = b
            # last count:
            counts.append(Nslots - prev - 1)
            # now counts is a length-D integer vector summing to denom
            pts.append([c / denom for c in counts])

        return np.array(pts)


if __name__ == "__main__":
    # Example usage
    simplex = Simplex()
    grid_3d = simplex.create_simplex_grid(3, 5)  # 2-simplex (triangle) with resolution 5
    print("3D Simplex Grid:\n", grid_3d)

    grid_4d = simplex.create_simplex_grid(4, 5)  # 3-simplex (tetrahedron) with resolution 5
    print("4D Simplex Grid:\n", grid_4d)

    grid_5d = simplex.create_simplex_grid(5, 5)  # 4-simplex with resolution 5
    print("5D Simplex Grid:\n", grid_5d)

    # grid_6d = simplex.create_simplex_grid(6, 50)  # 5-simplex with resolution 5
    # print("6D Simplex Grid:\n", grid_6d)

    # validate correctness
    assert np.allclose(grid_3d.sum(axis=1), 1.0), "3D simplex grid points do not sum to 1"
    assert np.allclose(grid_4d.sum(axis=1), 1.0), "4D simplex grid points do not sum to 1"
    assert np.allclose(grid_5d.sum(axis=1), 1.0), "5D simplex grid points do not sum to 1"
    # assert np.allclose(grid_6d.sum(axis=1), 1.0), "6D simplex grid points do not sum to 1"

    grid_2d = simplex.create_simplex_grid(2, 10)  # 5-simplex with resolution 5
    print("2D Simplex Grid:\n", grid_2d)

    grid_2d_bary = simplex._barycentric_grid(2, 10)  # 5-simplex with resolution 5
    print("2D Barycentric Grid:\n", grid_2d_bary)

    # assert same lists (up to tollerance)
    assert np.allclose(grid_2d, grid_2d_bary), "2D simplex grid does not match barycentric grid"

    ### test caching
    import time

    for i in range(10):
        start_time = time.time()
        g = simplex.create_simplex_grid(6, 50)
        elapsed = time.time() - start_time
        print(f"Iteration {i + 1}: Generated grid in {elapsed:.4f} seconds")

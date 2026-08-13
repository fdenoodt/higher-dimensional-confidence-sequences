import numpy as np


class HypercubeTools:
    @staticmethod
    def hypercube_and_simplex_intersection(lowers, uppers, D):
        assert len(lowers) == len(uppers) == D

        lowers_inters = np.zeros(D)  # intersection
        uppers_inters = np.zeros(D)

        # intersect with simplex if requested
        # adjust each dim: [l_d, u_d] intersected with sum=1 constraint
        for d in range(D):
            other_l = lowers[np.arange(D) != d].sum()
            other_u = uppers[np.arange(D) != d].sum()
            lowers_inters[d] = max(lowers[d], 1 - other_u)
            uppers_inters[d] = min(uppers[d], 1 - other_l)

        assert len(lowers_inters) == len(uppers_inters) == D, \
            "Lowers and uppers must have the same length as the number of dimensions D."

        assert np.all(lowers_inters <= uppers_inters), \
            "Lower bounds must be less than or equal to upper bounds in the intersection."

        return lowers_inters, uppers_inters

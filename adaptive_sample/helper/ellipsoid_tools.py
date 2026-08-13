import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


class EllipsoidTools:
    @staticmethod
    def build_S(c):
        v1 = c / np.linalg.norm(c)
        # simple Gram–Schmidt for v2:
        a = np.array([1.0, 0.0, 0.0])
        if np.allclose(a, v1):
            a = np.array([0.0, 1.0, 0.0])
        v2 = a - v1 * np.dot(a, v1)
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        S = np.vstack((v1, v2, v3))
        return S

    @staticmethod
    def build_S_householder(c, d: int):
        c_norm = np.linalg.norm(c)
        u = c / c_norm

        v = u.copy()
        v[0] = v[0] - 1

        H = np.identity(d) - (2 * np.outer(v, v)) / np.dot(v, v)
        # H:
        # 1) orthogonal
        # 2) Hu = e_1
        # 3) He_1 = u
        # 4) Symmetric
        return H

    @staticmethod
    def ellipsoid_rotated(q, Q, c, gamma, S):
        c_norm = np.linalg.norm(c)
        q_p = S @ q - (gamma / c_norm ** 2) * (S @ c)  # note the ** 2 which isn't in my equation!
        Q_p = S @ Q @ S.T
        return q_p, Q_p

    @staticmethod
    def ellipse_intersection_with_axis(q_p, Q_p):
        M = np.linalg.inv(Q_p)  # inverse of a symmetric matrix is symmetric! (nxn)
        m11 = M[0, 0]
        mbar = M[1:, 0]  # also mbar^T
        Mbar = M[1:, 1:]

        q1p = q_p[0]
        qbar = q_p[1:]

        # center in y-space
        wbar = qbar + (np.linalg.inv(Mbar) @ mbar) * q1p
        w_p = np.concatenate(([0], wbar))

        # shape in y-space
        alpha = 1 - (q1p ** 2) * (m11 - mbar.T @ np.linalg.inv(Mbar) @ mbar)
        Wbar = alpha * np.linalg.inv(Mbar)  # is W_p[1:, 1:]

        W_p = np.zeros_like(Q_p)  # np.zeros((3, 3))
        W_p[1:, 1:] = Wbar
        return w_p, W_p

    @staticmethod
    def ellipsoid_rotate_back(S, w_p, W_p, c, gamma):
        c_norm = np.linalg.norm(c)
        # w = S.T @ (w_p + (gamma / c_norm ** 2) * (S @ c))
        w = (S.T @ w_p) + (gamma / c_norm ** 2) * c

        W = S.T @ W_p @ S
        return w, W

    @staticmethod
    def sanity_checks(ellipse_pts, c, gamma, q, Q, atol=1e-6):
        """
        Robust sanity checks.

        ellipse_pts : array (d, N)
        c          : (d,)
        q          : (d,)
        Q          : (d,d) (shape matrix; original ellipsoid equation uses inv(Q))
        """
        d = len(q)
        D, num_pts = ellipse_pts.shape
        assert d == D
        assert num_pts > 0

        X = ellipse_pts

        # Plane residuals
        plane_vals = c @ X  # shape (N,)
        plane_resid = np.abs(plane_vals - gamma)

        # Ellipsoid quadratic residuals: (x-q)^T invQ (x-q) - 1
        invQ = np.linalg.inv(Q)
        delta = X - q[:, None]  # (d, N)
        vals = np.einsum('ij,ij->j', delta, invQ @ delta)  # length N
        ellip_resid = np.abs(vals - 1.0)

        ok_plane = np.allclose(plane_resid, 0.0, atol=atol)
        ok_ellip = np.allclose(ellip_resid, 0.0, atol=atol)
        assert ok_plane
        assert ok_ellip

    @staticmethod
    def intersection_ellipsoid_hyperplane(q: np.ndarray, Q: np.ndarray, c: np.ndarray, gamma: float, d: int) -> Tuple[
        np.ndarray, np.ndarray]:
        # E = { x | (x-q)^T Q (x-q) <= 1 }
        # H = { x | c^T x = gamma }

        # 3) build orthonormal S so first row is c/||c||
        # S = build_S(c)
        S = EllipsoidTools.build_S_householder(c, d)

        # 4) transform ellipsoid to y‐coordinates
        q_p, Q_p = EllipsoidTools.ellipsoid_rotated(q, Q, c, gamma, S)

        # 5) intersect with {y1=0} analytically
        w_p, W_p = EllipsoidTools.ellipse_intersection_with_axis(q_p, Q_p)

        # 6) back to x-space
        w, W = EllipsoidTools.ellipsoid_rotate_back(S, w_p, W_p, c, gamma)
        return w, W

    @staticmethod
    def extract_non_zero_eigens(W):
        # 7) extract the two nonzero eigens for the ellipse
        evals, evecs = np.linalg.eigh(W)
        mask = evals > 1e-6
        axes = np.sqrt(evals[mask])
        dirs = evecs[:, mask]
        return axes, dirs

    # def parametrize_ellipse(w, evecs, evals, num_datapoints=200):
    #     # evecs is an (dxk) mtx where k is the number of non-neg eig vecs
    #     # returns:
    #     # - theta (200): a linspace of theta vals
    #     # - ellipse: (n, 200): coordinates of ellipse
    #     theta_pts = np.linspace(0, 2 * np.pi, num_datapoints)
    #     ellipse_pts = w[:, None] + evecs @ np.diag(evals) @ np.vstack((np.cos(theta_pts), np.sin(theta_pts)))
    #     return theta_pts, ellipse_pts
    @staticmethod
    def parametrize_ellipsoid(w, evecs, evals, resolution=50, max_grid_dim=2, num_samples=2000):
        """
            Generic ellipsoid parameterization in any dimension.

            w      : (d,) center
            evecs  : (d, k) orthonormal eigenvectors
            evals  : (k,) positive radii (sqrt of eigenvalues)
            resolution : number of points per angular dimension (for small k)
            max_grid_dim : if k-1 > this, use random sampling instead of full meshgrid
            num_samples : number of random points to sample in high dimensions

            Returns:
                points : (d, N) points on ellipsoid
            """

        def _hypersphere_coords(k, *angles):
            coords = np.zeros(k)
            for i in range(k):
                if i == 0:
                    coords[i] = np.cos(angles[0])
                elif i < k - 1:
                    coords[i] = np.prod(np.sin(angles[:i])) * np.cos(angles[i])
                else:
                    coords[i] = np.prod(np.sin(angles[:i]))
            return coords

        k = len(evals)

        if k == 0:
            return w[:, None]
        elif k == 1:  # orig ellipsoid is 2d
            # return only the boundary points for the 1D "ellipsoid" (two endpoints),
            # otherwise interior samples will NOT satisfy the quadratic = 1 check.
            t = np.array([-1.0, 1.0])  # just endpoints on unit "sphere" S^0
            sphere_coords = t[np.newaxis, :]  # shape (1, 2)
            # t = np.linspace(-1, 1, resolution)
            # sphere_coords = t[np.newaxis, :]
        elif (k - 1) <= max_grid_dim:  # orig ellipsoid is 3d, 4d
            # Small dimension: do full meshgrid
            grids = []
            for i in range(k - 1):
                if i == k - 2:
                    grids.append(np.linspace(0, 2 * np.pi, resolution))
                else:
                    grids.append(np.linspace(0, np.pi, resolution))
            mesh = np.meshgrid(*grids, indexing="ij")
            angles_all = np.stack([m.ravel() for m in mesh], axis=-1)
            sphere_coords = np.array([_hypersphere_coords(k, *ang) for ang in angles_all]).T
        else:  # orig elipsoid is 5d or higher
            # print("approximating ellipsoid through sampling.")
            # High dimension: use random normal sampling to get uniform points on sphere
            X = np.random.normal(size=(k, num_samples))
            X /= np.linalg.norm(X, axis=0, keepdims=True)  # normalize to unit sphere
            sphere_coords = X

        # Scale and project into ambient space
        points = w[:, None] + evecs @ (np.diag(evals) @ sphere_coords)
        return points

    @staticmethod
    def compute_bounding_box(w, evals, evecs, D: int):
        # returns dict{x: (lw, up), y: ...}

        # parametric form for higher dimensional:
        # evecs: d×k matrix whose columns are v1..vk
        # evals: array([a1,a2,..,ak])
        # w: center in R^n
        bounding_box = np.zeros((D, 2))
        for idx in range(D):
            # compute half‐width
            delta = np.linalg.norm(evals * evecs[idx, :])  # evals*V[j,:] is elementwise
            bounding_box[idx] = (w[idx] - delta, w[idx] + delta)

        # old 2d code:
        #    using the parametric formula: x_j = w_j ± sqrt((a v1_j)^2 + (b v2_j)^2)
        # bounding_box = {}
        # a, b = evals
        # v1, v2 = evecs.T  # each is length‐3 row
        # for j, name in enumerate(['x', 'y', 'z']):
        #     delta = np.sqrt((a * v1[j]) ** 2 + (b * v2[j]) ** 2)
        #     bounding_box[name] = (w[j] - delta, w[j] + delta)

        return bounding_box

    @staticmethod
    def plot_intersected_ellipse(ellipse_pts, gamma, c, bounding_box, D: int):
        if D == 2:
            return EllipsoidTools._plot_intersected_ellipse_2d(ellipse_pts, gamma, c, bounding_box, D)
        elif D == 3:
            return EllipsoidTools._plot_intersected_ellipse_3d(ellipse_pts, gamma, c, bounding_box, D)
        elif D == 4:
            return EllipsoidTools._plot_intersected_ellipse_4d(ellipse_pts, gamma, c, bounding_box, D)
        else:
            return

    @staticmethod
    def _plot_intersected_ellipse_2d(ellipse_pts, gamma, c, bounding_box, D: int):
        assert D == 2

        pts = np.asarray(ellipse_pts)
        # normalize to (2, N)
        if pts.ndim == 1:
            pts = pts[:, None]
        if pts.shape[0] != 2 and pts.shape[1] == 2:
            pts = pts.T
        if pts.shape[0] != 2:
            raise ValueError("ellipse_pts must be shape (2, N) or (N, 2)")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(pts[0, :], pts[1, :], 'b-', lw=2, label='intersection')

        # plot the hyperplane (a line) for context
        extr = 4
        xx = np.linspace(-extr, extr, 200)
        # avoid division by zero
        if abs(c[1]) < 1e-12:
            yy = np.full_like(xx, np.nan)
        else:
            yy = (gamma - c[0] * xx) / c[1]
        ax.plot(xx, yy, color='C1', alpha=0.7, label='hyperplane')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # axis cross lines
        axis_range = np.linspace(-10, 10, 50)
        ax.plot(axis_range, np.zeros_like(axis_range), 'k--', lw=1)
        ax.plot(np.zeros_like(axis_range), axis_range, 'k--', lw=1)

        ax.set_xlim([-extr, extr])
        ax.set_ylim([-extr, extr])

        print("Coordinate‐wise extents (min, max):")
        for idx in range(D):
            lo, hi = bounding_box[idx]
            print(f" Axis {idx + 1}: ({lo:.3f}, {hi:.3f})")

        # bounding box corners and edges (square)
        lo_x, hi_x = bounding_box[0]
        lo_y, hi_y = bounding_box[1]
        vertices = [
            (lo_x, lo_y),
            (lo_x, hi_y),
            (hi_x, lo_y),
            (hi_x, hi_y),
        ]
        edges_2d = [(0, 1), (1, 3), (3, 2), (2, 0)]
        for edge in edges_2d:
            p = np.array([vertices[edge[0]], vertices[edge[1]]])
            ax.plot(p[:, 0], p[:, 1], 'r-', lw=2)

        ax.legend()
        plt.show()

    @staticmethod
    def _plot_intersected_ellipse_3d(ellipse_pts, gamma, c, bounding_box, D: int):
        # plots intersected ellipse and orig hyperplane. (not orig ellipsoid)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(ellipse_pts[0], ellipse_pts[1], ellipse_pts[2], 'b-', lw=2)

        # plane for context
        extr = 4

        xx, yy = np.meshgrid(np.linspace(-extr, extr, 10), np.linspace(-extr, extr, 10))
        # surface
        zz = (gamma - c[0] * xx - c[1] * yy) / c[2]
        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, linewidth=0, antialiased=True,
                        cmap='viridis', alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # plot axis lines: x=0, y=0, z=0
        axis_range = np.linspace(-10, 10, 50)
        ax.plot(axis_range, np.zeros_like(axis_range), np.zeros_like(axis_range), 'k--', lw=1)
        ax.plot(np.zeros_like(axis_range), axis_range, np.zeros_like(axis_range), 'k--', lw=1)
        ax.plot(np.zeros_like(axis_range), np.zeros_like(axis_range), axis_range, 'k--', lw=1)

        # set x, y, z range between [-2, 2] for each axis
        ax.set_xlim([-extr, extr])
        ax.set_ylim([-extr, extr])
        ax.set_zlim([-extr, extr])

        print("Coordinate‐wise extents (min, max):")
        for idx in range(D):
            lo, hi = bounding_box[idx]
            print(f" Axis {idx + 1}: ({lo:.3f}, {hi:.3f})")

        # Display the cube defined by these intervals
        lo_x, hi_x = bounding_box[0]
        lo_y, hi_y = bounding_box[1]
        lo_z, hi_z = bounding_box[2]

        # Define the 8 vertices of the cube
        vertices = [
            (lo_x, lo_y, lo_z),
            (lo_x, lo_y, hi_z),
            (lo_x, hi_y, lo_z),
            (lo_x, hi_y, hi_z),
            (hi_x, lo_y, lo_z),
            (hi_x, lo_y, hi_z),
            (hi_x, hi_y, lo_z),
            (hi_x, hi_y, hi_z)
        ]

        # Define the 12 edges as pairs of vertex indices
        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7)
        ]

        # Plot cube edges on the current 3D evals (ax)
        for edge in edges:
            pts = np.array([vertices[edge[0]], vertices[edge[1]]])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'r-', lw=2)

        plt.show()

    @staticmethod
    def _plot_intersected_ellipse_4d(ellipse_pts, gamma, c, bounding_box, D: int):
        """
        Plot an already-sampled intersection point cloud (ellipse_pts) that lies
        in a 4D hyperplane c^T x = gamma. Projects the 4D points into a 3D
        orthonormal basis of the hyperplane and plots them. Also projects the
        4D bounding-box corners and draws the edges.

        Parameters
        ----------
        ellipse_pts : array-like, shape (4, N) or (N, 4)
            Sampled points lying in the intersection (in 4D).
        gamma : float
            Plane offset (not used for plotting, kept for API parity).
        c : array-like, shape (4,)
            Plane normal.
        bounding_box : array-like, shape (4,2)
            Per-coordinate (lo, hi) intervals.
        D : int
            Ambient dimension (must be 4).
        """
        assert D == 4, "This plotting function is for D==4."

        pts = np.asarray(ellipse_pts)
        # normalize shape to (D, N)
        if pts.ndim == 1:
            pts = pts[:, None]
        if pts.shape[0] != D and pts.shape[1] == D:
            pts = pts.T
        if pts.shape[0] != D:
            raise ValueError("ellipse_pts must be shape (4, N) or (N, 4)")

        # compute centroid of provided points to center the plotted coords (approx center w)
        centroid = np.mean(pts, axis=1)  # shape (4,)

        # Build an orthonormal basis B for the hyperplane (columns span nullspace of c^T)
        # Use SVD trick: for column vector c, U[:,1:] spans nullspace
        U, _, _ = np.linalg.svd(c.reshape(-1, 1))
        B = U[:, 1:]  # shape (4,3) orthonormal columns

        # Project points into hyperplane coords y = B.T @ (x - centroid)
        y = B.T @ (pts - centroid[:, None])  # shape (3, N)

        # Start plotting in 3D (these y coords are a faithful representation of the
        # intersection ellipsoid in hyperplane coordinates)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

        # scatter the sampled points (no guaranteed mesh ordering since points may be random)
        ax.scatter(y[0, :], y[1, :], y[2, :], s=10, alpha=0.7, color='C0', label='intersection samples')

        # mark approximate center
        ax.scatter([0.0], [0.0], [0.0], color='k', s=40, label='centroid (approx center)')

        # Project bounding box corners (if provided) and draw edges
        if bounding_box is not None:
            b = np.asarray(bounding_box)
            if b.shape != (D, 2):
                raise ValueError("bounding_box must be shape (4,2)")

            # enumerate 2^4 = 16 corners
            corners = []
            for i in range(16):
                bits = [(i >> bit) & 1 for bit in range(D)]
                corner = np.array([b[didx, bit] for didx, bit in enumerate(bits)])
                corners.append(corner)
            corners = np.array(corners)  # (16,4)

            # project corners into hyperplane coordinates w.r.t. centroid: y_corner = B.T @ (corner - centroid)
            y_corners = (B.T @ (corners.T - centroid[:, None])).T  # (16,3)

            # scatter corner points
            ax.scatter(y_corners[:, 0], y_corners[:, 1], y_corners[:, 2], color='r', s=20,
                       label='projected box corners')

            # draw edges between corners that differ by Hamming distance 1
            for i in range(16):
                for j in range(i + 1, 16):
                    if bin(i ^ j).count("1") == 1:
                        p1 = y_corners[i];
                        p2 = y_corners[j]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', lw=0.8, alpha=0.6)

        # axis labels and polish
        ax.set_xlabel("hyperplane coord 1")
        ax.set_ylabel("hyperplane coord 2")
        ax.set_zlabel("hyperplane coord 3")
        ax.set_title("4D ellipsoid ∩ hyperplane (plotted in hyperplane coordinates)")
        ax.legend()

        # auto scale: use max range of plotted y to set symmetric limits
        if y.size > 0:
            m = np.max(np.abs(y))
            if m == 0:
                m = 1.0
            ax.set_xlim([-m, m])
            ax.set_ylim([-m, m])
            ax.set_zlim([-m, m])

        # print bounding box extents (original coordinates) like the other plotters
        print("Coordinate‐wise extents (min, max):")
        for idx in range(D):
            lo, hi = bounding_box[idx]
            print(f" Axis {idx + 1}: ({lo:.3f}, {hi:.3f})")

        plt.show()

    @staticmethod
    def ellipsoid_and_hyperplane_touch(q, Q, c, gamma, tol=1e-12):
        """
        Return True if ellipsoid E(q,Q) (with equation (x-q)^T Q^{-1} (x-q) <= 1)
        intersects (or touches) the hyperplane c^T x = gamma.

        Uses formula:
            dist = (|gamma - c^T q| - sqrt(c^T Q c)) / ||c||
        Intersection (non-empty) if dist <= 0.

        tol : small nonnegative tolerance to allow numerical slack.
        """
        q = np.asarray(q, dtype=float)
        c = np.asarray(c, dtype=float)
        Q = np.asarray(Q, dtype=float)

        if c.size != q.size:
            raise ValueError("c and q must have same length")

        cnorm = np.linalg.norm(c)
        if cnorm == 0:
            raise ValueError("plane normal c must be non-zero")

        # compute c^T Q c (should be >= 0 for positive-definite Q)
        ctQc = float(c.T @ Q @ c)
        if ctQc < -1e-15:
            # Q invalid or numerical pathology
            raise ValueError(f"c^T Q c is negative ({ctQc}); check Q positive-definite")

        radius = np.sqrt(max(ctQc, 0.0))
        dist = (abs(float(gamma) - float(c.T @ q)) - radius) / cnorm
        return dist <= tol

    @staticmethod
    def _setup_ellipsoid_nd(dim: int, gamma):
        if dim == 2:
            q = np.array([1.0, 2.0])
            Q = np.diag([4.0, 9.0])  # E = { x | (x-q)^T Q (x-q) <= 1 }

            # 2) define hyperplane c^T x = gamma
            c = np.array([1.0, 1.0])
        elif dim == 3:
            ###
            q = np.array([1.0, 2.0, 3.0])
            Q = np.diag([4.0, 9.0, 16.0])  # E = { x | (x-q)^T Q (x-q) <= 1 }

            # 2) define hyperplane c^T x = gamma
            c = np.array([1.0, 1.0, -1.0])
        elif dim == 4:
            q = np.array([1.0, 2.0, 3.0, 4.0])
            Q = np.diag([4.0, 9.0, 16.0, 25.0])  # E = { x | (x-q)^T Q (x-q) <= 1 }

            # 2) define hyperplane c^T x = gamma
            c = np.array([1.0, 1.0, -1.0, 1])
        elif dim > 4:
            max_scale = 100.0
            slack = 1.01

            q = np.arange(1, dim + 1, dtype=float)

            # modest diagonal pattern (keeps ellipsoids reasonable)
            # base_vals = np.array([4.0, 9.0, 16.0, 25.0, 36.0, 49.0])
            base_vals = np.array([4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 100_000.0])

            repeats = int(np.ceil(dim / base_vals.size))
            diag_vals = np.tile(base_vals, repeats)[:dim]
            Q = np.diag(diag_vals)

            # simple repeated plane-normal pattern
            base = np.array([1.0, 1.0, -1.0, 1.0])
            c = np.tile(base, int(np.ceil(dim / base.size)))[:dim]

            # ensure intersection: if current support radius < required distance, scale Q
            ctq = float(c @ q)
            desired = abs(float(gamma) - ctq)
            ctQc = float(c.T @ Q @ c)
            current_radius = np.sqrt(max(ctQc, 1e-12))

            if current_radius < desired:
                alpha = ((desired * slack) / current_radius) ** 2
                if alpha > max_scale:
                    alpha = max_scale
                Q = alpha * Q

            assert len(c) == dim
            assert len(q) == dim
            assert Q.shape == (dim, dim)
        else:
            raise ValueError("dim == 1 impossible.")

        assert EllipsoidTools.ellipsoid_and_hyperplane_touch(q, Q, c, gamma), "hyperplane misses ellipsoid!"

        return q, Q, c

    #######################################
    ### probably old code for when i still computed the intersecton points of ellipse and hyperplane myself (only for in 2d)
    @staticmethod
    def D(a: float, b: float, c: float) -> float:
        # Discriminant of the quadratic equation ax^2 + bx + c
        return b ** 2 - 4 * a * c

    @staticmethod
    def root(a: float, b: float, c: float) -> Tuple[float, float]:
        """
        Returns the roots of the quadratic equation ax^2 + bx + c = 0
        """
        d = EllipsoidTools.D(a, b, c)
        if d < 0:
            # return None, None  # No real roots
            raise ValueError("The equation has no real roots.")
        elif d == 0:
            # return -b / (2 * a), -b / (2 * a)  # One root
            raise ValueError("The equation has one root, but two roots are expected.")
        else:
            sqrt_d = np.sqrt(d)
            return (-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)

    @staticmethod
    def find_intersection_points_ellipse_probability_simplex(
            a: float, a_prime: float, b: float, b_prime: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # out: [(x1, y1), (x2, y2)]
        # denominator is d^2 * b^2
        denom = b_prime ** 2 * b ** 2

        # quadratic coefficients for x
        A = (b_prime ** 2 + b ** 2) / denom
        B = (-2 * a * b_prime ** 2 - 2 * b ** 2 + 2 * a_prime * b ** 2) / denom
        C = (
                    b_prime ** 2 * a ** 2
                    + b ** 2
                    - 2 * a_prime * b ** 2
                    + a_prime ** 2 * b ** 2
                    - b_prime ** 2 * b ** 2
            ) / denom

        # solve for x
        x1, x2 = EllipsoidTools.root(A, B, C)

        def _y(x0: float) -> Tuple[float, float]:
            # compute (y-h)^2 = d^2(1 - (x0-a)^2/b^2)
            tmp = b_prime ** 2 * (1 - (x0 - a) ** 2 / b ** 2)
            tmp = max(tmp, 0.0)
            return a_prime - np.sqrt(tmp), a_prime + np.sqrt(tmp)

        points = []
        for x0 in (x1, x2):
            y_low, y_high = _y(x0)
            # pick the one on the line x+y=1
            y0 = y_low if np.isclose(x0 + y_low, 1.0) else y_high
            points.append((x0, y0))

        return tuple(points)

    #####

    @staticmethod
    def get_ellipse_params_from_parabola(a_list: list[float], b_list: list[float], c_list: list[float], Beta: float,
                                         D: float, pow: int) -> \
            np.ndarray:
        # expects parabola has form: (x-a)^p/b^p - c = Beta
        # out: [(a, b), (a, b), ...] so a tuple for every dimensinon
        # returns params for ellipse:  e.g. (x-a)^p/b^p + (y-a')^p/b'^p = 1
        assert len(a_list) == len(b_list) == len(c_list) == D  # check dimension

        ellipse_params = np.empty((D, 2))
        for idx, (a, b) in enumerate(zip(a_list, b_list)):  # iterate over all dimensions
            ellipse_a = a
            # ellipse_b = b * np.sqrt((D * Beta) + np.sum(c_list))
            ellipse_b = b * (((D * Beta) + np.sum(c_list)) ** (1 / pow))
            ellipse_params[idx] = (ellipse_a, ellipse_b)

        return ellipse_params


def main():
    for d in range(2, 5):
        start_time = time.time()  # Start timing the iteration
        # 1) define ellipsoid center q and shape Q
        gamma = 1.0  # H = { x | c^T x = gamma }
        q, Q, c = EllipsoidTools._setup_ellipsoid_nd(d, gamma)

        w, W = EllipsoidTools.intersection_ellipsoid_hyperplane(q, Q, c, gamma, d)

        # 7) extract the two nonzero eigens for the ellipse
        evals, evecs = EllipsoidTools.extract_non_zero_eigens(W)

        # ** only for visualization **
        # 8) parametrize the ellipse
        ellipse_pts = EllipsoidTools.parametrize_ellipsoid(w, evecs,
                                                           evals)  # a list of theta vals and ellipse coordinates
        EllipsoidTools.sanity_checks(ellipse_pts, c, gamma, q, Q)

        # 10) compute min/max extents along each coordinate axis
        bounding_box = EllipsoidTools.compute_bounding_box(w, evals, evecs, d)

        EllipsoidTools.plot_intersected_ellipse(ellipse_pts, gamma, c, bounding_box, d)
        end_time = time.time()  # End timing the iteration
        print(f"Time for dim {d}: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()

import numpy as np
import scipy.sparse as sp
import osqp

from typing import List, Tuple, Optional

from config import GlobalConfig

# Cubic spline QP planner
class SLSplineQPOptimizer:
    def __init__(self, config : GlobalConfig):
        self.config = config
        self.sl_qp_spec = config.sl_qp_spec

        self.ego_half_width = config.ego_extent_y + config.lat_buffer_m
        self.ego_lr = config.rear_wheel_base

        self.num_vars = self.sl_qp_spec.num_vars
        self.num_samples = self.sl_qp_spec.num_samples

    def set_knots(self, knots : np.ndarray) -> None:
        self.knots = knots
        self.num_segs = len(knots) - 1

    def P0_seg_i(self, d : float) -> np.ndarray:
        return np.array([
            [d,        (d**2)/2, (d**3)/3, (d**4)/4],
            [(d**2)/2, (d**3)/3, (d**4)/4, (d**5)/5],
            [(d**3)/3, (d**4)/4, (d**5)/5, (d**6)/6],
            [(d**4)/4, (d**5)/5, (d**6)/6, (d**7)/7],
        ])

    def P1_seg_i(self, d : float) -> np.ndarray:
        return np.array([
            [0, 0,        0,             0           ],
            [0, d,        d**2,          d**3        ],
            [0, d**2,     4*(d**3)/3,    3*(d**4)/2  ],
            [0, d**3,     3*(d**4)/2,    9*(d**5)/5  ],
        ])

    def P2_seg_i(self, d : float) -> np.ndarray:
        return np.array([
            [0, 0, 0,            0        ],
            [0, 0, 0,            0        ],
            [0, 0, 4*d,          6*(d**2) ],
            [0, 0, 6*(d**2),     12*(d**3)],
        ])

    def P3_seg_i(self, d : float) -> np.ndarray:
        return np.array([
            [0, 0, 0,   0 ],
            [0, 0, 0,   0 ],
            [0, 0, 0,   0 ],
            [0, 0, 0, 36*d],
        ])

    def q0_seg_i(self, d : float, seg_idx : int, g_AB : np.ndarray):
        a_k = g_AB[seg_idx, 0]
        b_k = g_AB[seg_idx, 1]
        term1 = a_k * np.array([d, (d**2)/2, (d**3)/3, (d**4)/4])
        term2 = b_k * np.array([(d**2)/2, (d**3)/3, (d**4)/4, (d**5)/5])
        return term1 + term2

    def b0(self, tau : float):
        return np.array([
            1.0, tau, tau**2, tau**3
        ], dtype=np.float32)

    def b1(self, tau : float):
        return np.array([
            0.0, 1.0, 2.0*tau, 3.0*(tau**2)
        ], dtype=np.float32)

    def b2(self, tau : float):
        return np.array([
            0.0, 0.0, 2.0, 6.0*tau
        ], dtype=np.float32)

    def block_row(self, seg_idx : int, local_row : np.ndarray) -> np.ndarray:
        row = np.zeros(self.num_segs * self.num_vars)
        start_idx = seg_idx * self.num_vars
        row[start_idx : start_idx + self.num_vars] = local_row
        return row

    def find_segment(self, s : float) -> Tuple[int, float]:
        i = np.searchsorted(self.knots, s, side="right") - 1
        i = max(0, min(i, self.num_segs - 1))
        tau = s - self.knots[i]

        return i, tau

    def get_reference_dp_path(self, dp_path : np.ndarray) -> np.ndarray:
        s_dp = dp_path[:, 0]
        l_dp = dp_path[:, 1]

        l_knots = np.interp(self.knots, s_dp, l_dp)
        g_AB = np.zeros((self.num_segs, 2))
        g_AB[:, 0] = l_knots[:-1]
        g_AB[:, 1] = (l_knots[1:] - l_knots[:-1]) / (self.knots[1:] - self.knots[:-1] + 1e-6)

        return g_AB

    def solve(
        self,
        dp_path : np.ndarray,
        dp_bounds : np.ndarray,
        init_state : Tuple[float, float, float]
    ) -> Optional[np.ndarray]:
        # -----------------------------------------------------------------------------
        # 1. Vectorized Pre-computations
        # -----------------------------------------------------------------------------
        s_dp, l_dp = dp_path[:, 0], dp_path[:, 1]
        lb_dp, ub_dp = dp_bounds[:, 0], dp_bounds[:, 1]

        g_AB = self.get_reference_dp_path(dp_path)

        ds_sample = (self.knots[-1] - self.knots[0]) / self.num_samples
        s_samples = np.arange(self.knots[0], self.knots[-1] + 1e-3, ds_sample)

        # Vectorized array lookups
        l_low = np.interp(s_samples, s_dp, lb_dp)
        l_high = np.interp(s_samples, s_dp, ub_dp)

        seg_indices = np.searchsorted(self.knots, s_samples, side="right") - 1
        seg_indices = np.clip(seg_indices, 0, self.num_segs - 1)

        # -----------------------------------------------------------------------------
        # 2. Objective Formulation (P, q)
        # -----------------------------------------------------------------------------
        total_vars = self.num_vars * self.num_segs
        P_mat = np.zeros((total_vars, total_vars), dtype=np.float32)
        q_vec = np.zeros(total_vars, dtype=np.float32)

        for i in range(self.num_segs):
            d_seg_i = self.knots[i + 1] - self.knots[i]
            H_i = 2.0 * (
                self.sl_qp_spec.W_L * self.P0_seg_i(d_seg_i) +
                self.sl_qp_spec.W_dL * self.P1_seg_i(d_seg_i) +
                self.sl_qp_spec.W_ddL * self.P2_seg_i(d_seg_i) +
                self.sl_qp_spec.W_dddL * self.P3_seg_i(d_seg_i)
            )

            idx = self.num_vars * i
            P_mat[idx:idx+self.num_vars, idx:idx+self.num_vars] += H_i
            q_vec[idx:idx+self.num_vars] += -2.0 * self.sl_qp_spec.W_L * self.q0_seg_i(d_seg_i, i, g_AB)

        P = sp.csc_matrix(np.triu(P_mat))

        # -----------------------------------------------------------------------------
        # Optimization constraints formulation
        # -----------------------------------------------------------------------------

        A_rows = []
        lower = []
        upper = []

        def add_constraint(row, lo, hi):
            A_rows.append(row)
            lower.append(lo)
            upper.append(hi)

        # Initial State
        l0, dl0, ddl0 = init_state

        fs0 = self.block_row(0, self.b0(0.0))
        dfs0 = self.block_row(0, self.b1(0.0))
        ddfs0 = self.block_row(0, self.b2(0.0))
        add_constraint(fs0, l0, l0)
        add_constraint(dfs0, dl0, dl0)
        add_constraint(ddfs0, ddl0, ddl0)

        # Vehicle point corridor bounds
        ds_sample = (self.knots[-1] - self.knots[0]) / self.num_samples
        s_samples = np.arange(self.knots[0], self.knots[-1] + 1e-3, ds_sample)
        l_low = np.interp(s_samples, s_dp, lb_dp)
        l_high = np.interp(s_samples, s_dp, ub_dp)

        w = self.ego_half_width
        for s, lo, hi in zip(s_samples, l_low, l_high):
            i, tau = self.find_segment(s)
            front_row = self.block_row(i, self.b0(tau) + self.ego_lr * self.b1(tau))
            back_row  = self.block_row(i, self.b0(tau) - self.ego_lr * self.b1(tau))

            # 2. Shift the constants to the boundaries
            # If the left corner (L + w) must be < hi, then L must be < hi - w
            # If the right corner (L - w) must be > lo, then L must be > lo + w
            eff_lo = lo + w
            eff_hi = hi - w

            # 3. Prevent Primal Infeasibility (Inverted Bounds)
            # If the corridor is physically too narrow for the car, we must relax
            # the constraint temporarily so the QP solver doesn't crash.
            if eff_lo > eff_hi:
                mid_point = (eff_lo + eff_hi) / 2.0
                eff_lo = mid_point - 0.01
                eff_hi = mid_point + 0.01

            # Skip inequalities here; equality constraints handle s=0 entirely.
            if s == s_samples[0]:
                continue

            add_constraint(front_row, eff_lo, eff_hi)
            add_constraint(back_row, eff_lo, eff_hi)

        # C2 continuity at knot junctions
        for i in range(self.num_segs - 1):
            d_seg_i = self.knots[i+1] - self.knots[i]
            # f_i(d) - f_{i+1}(0) = 0
            # f'_i(d) - f'_{i+1}(0) = 0
            # f''_i(d) - f''_{i+1}(0) = 0
            c0_cont = self.block_row(i, self.b0(d_seg_i)) - self.block_row(i + 1, self.b0(0.0))
            c1_cont = self.block_row(i, self.b1(d_seg_i)) - self.block_row(i + 1, self.b1(0.0))
            c2_cont = self.block_row(i, self.b2(d_seg_i)) - self.block_row(i + 1, self.b2(0.0))

            add_constraint(c0_cont, 0.0, 0.0)
            add_constraint(c1_cont, 0.0, 0.0)
            add_constraint(c2_cont, 0.0, 0.0)

        # Heading and curvature limits
        max_heading = self.sl_qp_spec.dL_max
        max_curvature = self.sl_qp_spec.ddL_max
        for idx, s in enumerate(s_samples):
            if idx == 0:
                continue

            i, tau = self.find_segment(s)

            heading = self.block_row(i, self.b1(tau))
            curvature = self.block_row(i, self.b2(tau))

            add_constraint(heading, -max_heading, max_heading)
            add_constraint(curvature, -max_curvature, max_curvature)

        A = sp.csc_matrix(np.vstack(A_rows))
        l_vec = np.array(lower)
        u_vec = np.array(upper)

        # -----------------------------------------------------------------------------
        # QP solve
        # -----------------------------------------------------------------------------

        prob = osqp.OSQP()
        prob.setup(
            P=P,
            q=q_vec,
            A=A,
            l=l_vec,
            u=u_vec,
            verbose=False,
            polish=True,           # Smooths out ADMM numerical jitter
            adaptive_rho=True,     # Adapts the penalty parameter for poorly scaled splines
            max_iter=10000,        # Give it a bit more runway
            eps_abs=1e-4,          # Slight relaxation on absolute tolerance
            eps_rel=1e-4           # Slight relaxation on relative tolerance
        )
        res = prob.solve()

        if res.info.status_val != 1:
            print(f"[Warn] Spline QP Solver Failed: {res.info.status}")
            return None

        # -----------------------------------------------------------------------------
        # 5. Vectorized Path Reconstruction
        # -----------------------------------------------------------------------------
        m_per_point = 1.0 / self.config.points_per_meter
        s_dense = np.arange(self.knots[0], self.knots[-1] + 1e-3, m_per_point)

        # Vectorized lookup and polynomial evaluation
        i_dense = np.searchsorted(self.knots, s_dense, side="right") - 1
        i_dense = np.clip(i_dense, 0, self.num_segs - 1)
        taus_dense = s_dense - self.knots[i_dense]

        coeffs = res.x.reshape((self.num_segs, 4))
        c_dense = coeffs[i_dense] # Shape: (len(s_dense), 4)

        b0_dense = np.vstack([np.ones_like(taus_dense), taus_dense, taus_dense**2, taus_dense**3]).T
        l_dense = np.sum(c_dense * b0_dense, axis=1)

        return np.column_stack((s_dense, l_dense))

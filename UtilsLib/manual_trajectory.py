import matplotlib.markers
import numpy as np
import scipy
from scipy.interpolate import CubicSpline

import matplotlib
import matplotlib.pyplot as plt


class MINCOTrajectory:

    def __init__(
        self,
        T: np.ndarray,
        p: np.ndarray,
        init_p: np.ndarray,
        init_v: np.ndarray,
        init_a: np.ndarray,
        final_p: np.ndarray,
        final_v: np.ndarray,
        final_a: np.ndarray,
    ):
        """Minimum control effort (MINCO) polynomial trajectory.

        Parameters
        ----------
        T : np.ndarray
            Time interval lengths. Shape = (N, ).
        p : np.ndarray
            Flatten output on time stamps. Shape = (N - 1, 3).
        init_p, init_v, init_a : np.ndarray
            Initial position, velocity and acceleration. Shape = (3, ).
        final_p, final_v, final_a : np.ndarray
            Final position, velocity and acceleration. Shape = (3, ).
        """
        # Dimension check.
        assert len(T.shape) == 1
        self.N = len(T)

        assert p.shape == (self.N - 1, 3)

        assert init_p.shape == (3, )
        assert init_v.shape == (3, )
        assert init_a.shape == (3, )
        assert final_p.shape == (3, )
        assert final_v.shape == (3, )
        assert final_a.shape == (3, )

        # Flatten variables.
        self.T = T.copy()
        self.p = p.copy()

        self.t = np.cumsum(T)

        # Bases and their derivatives.
        self.b = lambda tau: np.array([tau ** i for i in range(6)])
        self.db = lambda n, tau: np.array([
            0.0 if i < n
            else np.prod(range(i, i - n, -1)) * (tau ** (i - n)) 
            for i in range(6)
        ])

        # A and D matrices.
        self.A = np.concatenate(
            [
                scipy.linalg.block_diag(*[self._A_sub(i) for i in range(self.N)]),
                np.zeros(shape=(3, 6 * self.N))
            ],
            axis=0
        ) + np.concatenate(
            [
                np.zeros(shape=(3, 6 * self.N)),
                scipy.linalg.block_diag(*[self._A_sup(i) for i in range(1, self.N + 1)]),
            ],
            axis=0
        )
        assert self.A.shape == (6 * self.N, 6 * self.N)

        D_stacks = list()
        for i in range(self.N):
            if i == 0:
                D_stacks.append(init_p[None, :])
                D_stacks.append(init_v[None, :])
                D_stacks.append(init_a[None, :])
            if i == self.N - 1:
                D_stacks.append(final_p[None, :])
                D_stacks.append(final_v[None, :])
                D_stacks.append(final_a[None, :])
            else:
                D_stacks.append(p[i, None])
                D_stacks.append(np.zeros(shape=(5, 3)))
        self.D = np.concatenate(D_stacks, axis=0)
        assert self.D.shape == (6 * self.N, 3)

        self.C = np.linalg.inv(self.A) @ self.D
        
    def _A_sub(self, i: int) -> np.ndarray:
        r"""Get :math:`\underline{A}_i`.
        """
        assert 0 <= i and i <= self.N - 1

        if i == 0:
            return np.stack(
                [self.b(0), self.db(1, 0), self.db(2, 0)],
                axis=0
            )
        
        return np.stack(
            [np.zeros(shape=(6,)), -self.b(0)] + [-self.db(n, 0) for n in range(1, 5)],
            axis=0
        )
        
    def _A_sup(self, i: int) -> np.ndarray:
        r"""Get :math:`\overline{A}_i`.
        """
        assert 1 <= i and i <= self.N

        # Paper is 1-indexing and code is 0-indexing.
        T = self.T[i - 1]

        if i == self.N:
            return np.stack(
                [self.b(T), self.db(1, T), self.db(2, T)],
                axis=0
            )
    
        return np.stack(
            [self.b(T), self.b(T)] + [self.db(n, T) for n in range(1, 5)],
            axis=0
        )
    
    def _get(self, order, t):
        # Determine which polynomial segment.
        i = np.digitize(t, self.t)
        assert 0 <= i and i < self.N
        
        # Segment starting time.
        t_start = 0 if i == 0 else self.t[i - 1]

        # Coefficients.
        c = self.C[6 * i : 6 * (i + 1), :]

        # Bases.
        if order == 0:
            bases = self.b(t - t_start)
        else:
            bases = self.db(order, t - t_start)
        
        return c.T @ bases
    
    def get_position(self, t):
        return self._get(0, t)

    def get_velocity(self, t):
        return self._get(1, t)
    
    def get_acceleration(self, t):
        return self._get(2, t)
    
    def get_pva(self, t):
        p = self.get_position(t)
        v = self.get_velocity(t)
        a = self.get_acceleration(t)

        return p, v, a
    

class ManualCircleTrajectory:

    leg_direction_vframe: np.ndarray

    def __init__(
        self,
        init_p: np.ndarray,
        init_vn: float, init_ve: float,
        leg_length_ref=10000.0,
        h_ref=1000.0,
        v_ref=325.0,
    ):
        """Manual trajectory for test.

        Note. It is required that the aircraft is flying horizontally when initialize.

        Parameters
        ----------
        init_p : np.ndarray
            Initial position in NED frame.
        init_vn, init_ve : float
            Initial north- and east-bound velocities.
        leg_length_ref : float, optional
            Reference length for each leg, unit: m. By default 10000 (10 km).
        h_ref : float, optional
            Climb height reference, unit: m. By default 1000 (1 km).
        v_ref : float, optional
            Reference airspeed, unit: m.s^{-1}. By default 325.
        """
        # Rotation matrix from velocity frame to inertial frame.
        init_chi = np.arctan2(-init_ve, init_vn)
        # init_chi -= np.pi / 2
        r_v_to_i = np.array([
            [np.cos(init_chi), np.sin(init_chi), 0.0],
            [-np.sin(init_chi), np.cos(init_chi), 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        self.leg_direction_vframe = self.__class__.leg_direction_vframe.copy()
        # Scale the leg directions so the shortest one is longer than leg_length_ref.
        min_leg_length = np.min(
            # Assume that only make turns in horizontal planes, 
            # so only use leg_direction_vframe[0:1] to normalize leg lengths.
            np.linalg.norm(self.leg_direction_vframe[:, 0:2], ord=2, axis=-1)
        )
        self.leg_direction_vframe[:, 0:2] *= (np.float64(leg_length_ref) / min_leg_length)
        self.leg_direction_vframe[:, -1] *= h_ref

        # Construct path points in velocity frame.
        # p = np.array([0.0, 0.0, 0.0])
        p_last = np.array([0.0, 0.0, 0.0])
        p_vframe = list()
        p_vframe.append(p_last.copy())
        for leg in self.leg_direction_vframe:
            p_last += leg
            p_vframe.append(p_last.copy())
        p_vframe = np.array(p_vframe)

        # Convert path points from velocity frame to inertial frame.
        self.p_inertial = (r_v_to_i @ p_vframe.T).T + init_p

        # Allocate time for each leg so that the velocities on each leg are v_ref.
        t = np.linalg.norm(self.leg_direction_vframe, ord=2, axis=-1) / v_ref
        # Scale t to ensure that velocity and acceleration are not too large.
        t *= 1.1
        self.t_cum = np.append([0.0], np.cumsum(t))

        self.t_max = self.t_cum[-1] - 0.0001
        
        # Force p[0] == p[-1].
        assert np.all(np.isclose(self.p_inertial[-1, :], self.p_inertial[0, :]))
        self.p_inertial[-1, :] = self.p_inertial[0, :]

        # Construct cubic spline trajectory.
        self.cubic_spline = [
            # CubicSpline(self.t_cum, self.p_inertial[:, 0], bc_type="periodic"),
            # CubicSpline(self.t_cum, self.p_inertial[:, 1], bc_type="periodic"),
            # CubicSpline(self.t_cum, self.p_inertial[:, 2], bc_type="periodic"),
            CubicSpline(self.t_cum, self.p_inertial[:, 0], bc_type=((1, init_vn), (1, init_vn))),
            CubicSpline(self.t_cum, self.p_inertial[:, 1], bc_type=((1, init_ve), (1, init_ve))),
            CubicSpline(self.t_cum, self.p_inertial[:, 2], bc_type=((1, 0), (1, 0))),
        ]

        # Construct MINCO trajectory.
        self.minco = MINCOTrajectory(
            t, self.p_inertial[1 : -1], 
            init_p=init_p, init_v=np.array([init_vn, init_ve, 0]), init_a=np.zeros_like(init_p),
            final_p=init_p, final_v=np.array([init_vn, init_ve, 0]), final_a=np.zeros_like(init_p),
        )

    def get_cubic_spline_pva(self, tau: float):
        tau = tau % self.t_max
        p = np.array([cs(tau) for cs in self.cubic_spline])
        v = np.array([cs.derivative(1)(tau) for cs in self.cubic_spline])
        a = np.array([cs.derivative(2)(tau) for cs in self.cubic_spline])
        return p, v, a
    
    def get_minco_pva(self, tau: float):
        tau = tau % self.t_max
        p, v, a = self.minco.get_pva(tau)
        return p, v, a

    def visualize_trajectory(self, t_0=0.0, t_f=1000.0, n_samples_per_sec=10):
        taus = np.linspace(t_0, t_f, int((t_f - t_0) * n_samples_per_sec))
        cubic_spline_p = list()
        minco_p = list()
        
        for tau in taus:
            cubic_spline_p.append(self.get_cubic_spline_pva(tau)[0])
            minco_p.append(self.get_minco_pva(tau)[0])

        cubic_spline_p = np.array(cubic_spline_p)
        minco_p = np.array(minco_p)

        fig = plt.figure()
        ax = fig.subplots(1, 1, subplot_kw={"projection": "3d"})

        # Visualize path points.
        ax.plot(self.p_inertial[:, 0], self.p_inertial[:, 1], self.p_inertial[:, 2], "bo--", alpha=0.5)

        # Visualize cubic spline trajectory.
        # cubic_spline_p, _, _ = self.get_cubic_spline_pva(taus)
        ax.plot(cubic_spline_p[:, 0], cubic_spline_p[:, 1], cubic_spline_p[:, 2], "red", alpha=0.75)

        # Visualize MINCO trajectory.
        # minco_p, _, _ = self.get_minco_pva(taus)
        ax.plot(minco_p[:, 0], minco_p[:, 1], minco_p[:, 2], "green", alpha=0.75)

        ax.xaxis.set_label_text("N [m]")
        ax.yaxis.set_inverted(True)
        ax.yaxis.set_label_text("E [m]")
        ax.zaxis.set_inverted(True)
        ax.zaxis.set_label_text("D [m]")

        ax.axis("scaled")
        ax.legend(["Path Points", "Cubic Spline", "MINCO"])

    def visualize_va(self, t_0=0.0, t_f=1000.0, n_samples_per_sec=10):
        taus = np.linspace(t_0, t_f, int((t_f - t_0) * n_samples_per_sec))
        cubic_spline_v = list()
        minco_v = list()
        cubic_spline_a = list()
        minco_a = list()
        
        for tau in taus:
            cubic_spline_v.append(self.get_cubic_spline_pva(tau)[1])
            minco_v.append(self.get_minco_pva(tau)[1])
            cubic_spline_a.append(self.get_cubic_spline_pva(tau)[2])
            minco_a.append(self.get_minco_pva(tau)[2])

        cubic_spline_v = np.array(cubic_spline_v)
        minco_v = np.array(minco_v)
        cubic_spline_a = np.array(cubic_spline_a)
        minco_a = np.array(minco_a)

        fig = plt.figure()
        ax = fig.subplots(2, 1)
        ax[0].plot(taus, np.linalg.norm(cubic_spline_v, ord=2, axis=-1), "red")
        ax[0].plot(taus, np.linalg.norm(minco_v, ord=2, axis=-1), "green")
        for t in self.t_cum:
            ax[0].axvline(t, color="gray", ls="--")
        ax[0].set_xlabel(r"$t$ [s]")
        ax[0].set_ylabel(r"$v$ [m/s]")
        ax[0].legend(["Cubic Spline", "MINCO"])

        ax[1].plot(taus, np.linalg.norm(cubic_spline_a, ord=2, axis=-1), "red")
        ax[1].plot(taus, np.linalg.norm(minco_a, ord=2, axis=-1), "green")
        for t in self.t_cum:
            ax[1].axvline(t, color="gray", ls="--")
        ax[1].set_xlabel(r"$t$ [s]")
        ax[1].set_ylabel(r"$a$ [m/s^2]")
        ax[1].legend(["Cubic Spline", "MINCO"])

    
class ExpediteClimbAndTurnDescentTrajectory(ManualCircleTrajectory):

    leg_direction_vframe = np.array([
        [10, 0, -1],
        [3, 0, -3],
        [3, 0, -10],
        [3, 0, -3],
        [4, 2, 0],
        [0, 6, 0],
        [-5, 3, 0],
        [-10, 5, 2],
        [-10, 0, 4],
        [-4, -4, 2],
        [0, -5, 2],
        [0, -6, 0],
        [-2, -2, 0],
        [-10, 0, 0], 
        [-4, 4, 2],
        [0, 7, 2],
        [3, 3, 2],
        [3, 0, 1],
        [6, -3, 0],
        [10, -10, 0]
    ], dtype=np.float64)


class EightShapeTrajectory(ManualCircleTrajectory):

    leg_direction_vframe = np.array([
        [2, 0, 0],
        [1, -1, 0],
        [0, -2, 0],
        [-1, -1, 0],
        # [2, 0, 0],

        [-2, 0, -1],
        [-1, 0, -2],
        [-1, 0, -2],
        [-2, 0, -1],

        [-1, -1, 0],
        [0, -2, 0],
        [1, -1, 0],
        [2, 0, 0],
        [1, 1, 0],

        [0, 2, 1],
        [0, 1, 2],
        [0, 1, 2],
        [0, 2, 1],

        [1, 1, 0],
    ], dtype=np.float64)
    # leg_direction_vframe = np.append(
    #     leg_direction_vframe, 
    #     [-np.sum(leg_direction_vframe, axis=0)], 
    #     axis=0
    # )

    
class SomeTrajectory(ManualCircleTrajectory):

    p = np.array([
        [   -1.0,   -1.0,    0.0    ],
        [    0.2,   -1.7,    1.0    ],
        [    1.5,   -0.5,    1.5    ],
        [    0.0,    0.0,    2.0    ],
        [   -2.3,    1.0,    2.0    ],
        [    0.1,    2.5,    1.2    ],
        [    2.1,    1.0,    0.5    ],
        [    3.6,    1.2,    0.0    ]
    ]
    )

    leg_direction_vframe = np.diff(p, n=1, axis=0)
    leg_direction_vframe = np.append(
        leg_direction_vframe, 
        [-np.sum(leg_direction_vframe, axis=0)], 
        axis=0
    )

if __name__ == "__main__":
    # trajectory = SomeTrajectory(
    trajectory = ExpediteClimbAndTurnDescentTrajectory(
        init_p=np.array([0.0, 0.0, 0.0]),
        init_vn=240.0, init_ve=0.0, leg_length_ref=10000, h_ref=800, v_ref=240.0
        # init_vn=5, init_ve=0, leg_length_ref=20, v_ref=5
        # init_vn=230.0, init_ve=230.0,
    )
    trajectory.visualize_trajectory(0.0, trajectory.t_max)
    trajectory.visualize_va(0.0, trajectory.t_max)
    plt.show()
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# ------------------ 轨迹生成模块 ------------------ #
from scipy.interpolate import splprep, splev


class SmoothTrajectory:
    def __init__(self, waypoints, airspeed=325.0):
        self.pts = np.array(waypoints).T  # shape: (3, N)
        self.airspeed = airspeed          # m/s
        self._init_spline()

    def _init_spline(self):
        # 参数化轨迹: B样条拟合
        self.tck, self.u = splprep(self.pts, s=0)

        # 计算总路径长度（用作时间参考）
        sampled = np.array(splev(np.linspace(0, 1, 500), self.tck)).T
        dist = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
        self.total_time = np.sum(dist) / self.airspeed

    def get_ref(self, t):
        u = np.clip(t / self.total_time, 0, 1)
        pos = np.array(splev(u, self.tck))
        vel = np.array(splev(u, self.tck, der=1)) * (1 / self.total_time)
        acc = np.array(splev(u, self.tck, der=2)) * (1 / self.total_time**2)

        yaw = np.arctan2(vel[1], vel[0])
        return pos, vel, acc, np.array([0, 0, yaw])  # roll/pitch可留0

# 示例调用
waypoints = [
    [0, 0, -7500],
    [2000, 1000, -7500],
    [4000, 3000, -7500],
    [6000, 6000, -7500]
]
traj = SmoothTrajectory(waypoints, airspeed=325)

for t in np.linspace(0, traj.total_time, 100):
    pos, vel, acc, rpy = traj.get_ref(t)
    print("POS:", pos, "VEL:", vel, "ACC:", acc, "YAW:", rpy[2])



class LemniscateTrajectory:
    def __init__(self, A=100, omega=None, speed_target=320, entry_pos=None, entry_vel=None, heading_offset=None):
        if omega is None:
            omega = speed_target / (A * np.sqrt(5))
        self.A = A
        self.omega = omega
        self.entry_pos = entry_pos if entry_pos is not None else np.array([0.0, 0.0, 0.0])
        
        # heading_offset可以外部传进来，优先级高于entry_vel
        if heading_offset is not None:
            self.heading_offset = heading_offset
        else:
            self.heading_offset = self._compute_shift_angle(entry_vel)

    def get_trajectory(self, t):
        A = self.A
        w = self.omega

        north = A * np.cos(w * t)
        east  = A * np.sin(2 * w * t)

        d_north = -A * w * np.sin(w * t)
        d_east  = 2 * A * w * np.cos(2 * w * t)

        yaw = np.arctan2(d_east, d_north)  # 不加offset

        pos_ned = np.array([north, east, 0])
        rpy_ned = np.array([0, 0, yaw])

        return pos_ned, rpy_ned
    

    def get_shifted_trajectory(self, t):
        A = self.A
        w = self.omega

        # 计算轨迹点
        north = A * np.cos(w * t)
        east  = A * np.sin(2 * w * t)

        # 计算导数（速度分量）
        d_north = -A * w * np.sin(w * t)
        d_east  = 2 * A * w * np.cos(2 * w * t)

        dd_north = -A * w**2 * np.cos(w * t)
        dd_east  = -4 * A * w**2 * np.sin(2 * w * t)

        # 未旋转前的位置和速度
        pos_local = np.array([north, east, 0])
        vel_local = np.array([d_north, d_east, 0])
        acc_local = np.array([dd_north, dd_east, 0])

        # 旋转水平面位置
        pos_horizontal = rotate_2d(pos_local[:2], self.heading_offset)
        pos_rotated = np.array([pos_horizontal[0], pos_horizontal[1], pos_local[2]])
        pos_global = self.entry_pos + pos_rotated

        # 旋转水平面速度
        vel_horizontal = rotate_2d(vel_local[:2], self.heading_offset)
        vel_rotated = np.array([vel_horizontal[0], vel_horizontal[1], vel_local[2]])
        
        
        acc_horizontal = rotate_2d(acc_local[:2], self.heading_offset)
        acc_rotated = np.array([acc_horizontal[0], acc_horizontal[1], acc_local[2]])
        # 计算 yaw (航向角)
        yaw_local = np.arctan2(d_east, d_north)
        yaw_global = yaw_local + self.heading_offset

        # 姿态
        roll = 10 * np.sin(0.1 * t) * np.pi / 180
        pitch = 5 * np.sin(0.2 * t) * np.pi / 180
        rpy_global = np.array([roll, pitch, yaw_global])

        return pos_global, vel_rotated, acc_rotated, rpy_global


    @staticmethod
    def _compute_shift_angle(current_vel):
        return np.arctan2(current_vel[1], current_vel[0])  # east / north


class CardioidTrajectory:
    def __init__(self, A=1.0, omega=1.0):
        """
        初始化心形轨迹生成器
        :param A: 振幅系数（控制轨迹大小）
        :param omega: 角速度（theta = omega * t）
        """
        self.A = A
        self.omega = omega

    def get_target(self, t):
        """
        获取给定时间 t 下的期望位置、速度、加速度、偏航角
        :param t: 时间（单位：秒）
        :return: target_pos, target_vel, target_acc, target_rpy
        """
        theta = self.omega * t
        dtheta = self.omega
        ddtheta = 0.0  # 匀速角速度

        a = self.A

        # --- 位置 ---
        x = a * np.sin(2 * theta)
        y = 2 * a * np.sin(theta)**2

        # --- 速度 ---
        dx_dtheta = 2 * a * np.cos(2 * theta)
        dy_dtheta = 4 * a * np.sin(theta) * np.cos(theta)

        vx = dx_dtheta * dtheta
        vy = dy_dtheta * dtheta

        # --- 加速度 ---
        ddx = (-4 * a * np.sin(2 * theta)) * dtheta**2
        ddy = (4 * a * np.cos(2 * theta)) * dtheta**2

        # --- yaw角（方向） ---
        yaw = np.arctan2(vy, vx)

        target_pos = np.array([x, y])
        target_vel = np.array([vx, vy])
        target_acc = np.array([ddx, ddy])
        target_rpy = np.array([0.0, 0.0, yaw])

        return target_pos, target_vel, target_acc, target_rpy


def rotate_2d(vec, angle_rad):
    """2D旋转：绕垂直轴旋转 (north-east 平面)"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s],
                  [s,  c]])
    return R @ vec

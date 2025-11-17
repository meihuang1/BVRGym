import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

class FixWingVisualizer:
    def __init__(self, A=100, follow=True, show_traj=True):
        """
        纯NED坐标输入，前左上显示系输出
        """
        self.A = A
        self.scale = A * 0.1 / 10.0
        self.follow = follow
        self.show_traj = show_traj
        self._init_model()
        self.history = {key: [] for key in ['x','y','z','roll','pitch','yaw']}
        self.state = [0,0,0,0,0,0]
        
        # fig1 局部观察姿态
        self.fig1 = plt.figure()
        self.ax1 = self.fig1.add_subplot(111, projection='3d')
        self.ax1.set_xlabel('Front')
        self.ax1.set_ylabel('Left')
        self.ax1.set_zlabel('Up')
        self.xy_range = A * 1.5
        self.z_range = A * 0.5
        self.ax1.set_xlim(-self.xy_range, self.xy_range)
        self.ax1.set_ylim(-self.xy_range, self.xy_range)

        # fig2 全局观察路径
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        
        
    def _init_model(self):
        self.nose = np.array([10.0, 0.0, 0.0]) * self.scale
        self.rear_left = np.array([0.0, -3.5,  0.0]) * self.scale
        self.rear_right = np.array([0.0, 3.5,  0.0]) * self.scale
        self.rear_bottom = np.array([0.0, 0.0, 2.0]) * self.scale
        self.tail_top = np.array([0.0, 0.0, 2]) * self.scale
        self.edges = [
            ([self.rear_left, self.rear_right], 'red'),
            ([self.rear_left, self.rear_bottom], 'red'),
            ([self.rear_right, self.rear_bottom], 'red'),
            ([self.nose, self.rear_left], 'blue'),
            ([self.nose, self.rear_right], 'blue'),
            ([self.nose, self.rear_bottom], 'blue'),
            ([self.rear_left, self.tail_top], 'green'),
            ([self.rear_right, self.tail_top], 'green'),
        ]

    def ned_to_display(self, pos):
        """
        NED --> 前左上
        """
        x_n, y_e, z_d = pos
        return np.array([x_n, -y_e, -z_d])

    def set_state(self, pos_ned, rpy_ned):
        x, y, z = self.ned_to_display(pos_ned)
        roll, pitch, yaw = rpy_ned
        self.state = [x, y, z, roll, pitch, yaw]
        self.history['x'].append(x)
        self.history['y'].append(y)
        self.history['z'].append(z)
        self.history['roll'].append(roll)
        self.history['pitch'].append(pitch)
        self.history['yaw'].append(yaw)

    def show(self):
        ## ==============Show fig1
        self.ax1.cla()
        self.ax1.set_xlabel('Front')
        self.ax1.set_ylabel('Left')
        self.ax1.set_zlabel('Up')
        x,y,z,roll,pitch,yaw = self.state

        if self.follow:
            self.ax1.set_xlim(x - self.xy_range/2, x + self.xy_range/2)
            self.ax1.set_ylim(y - self.xy_range/2, y + self.xy_range/2)
            self.ax1.set_zlim(z - self.z_range/2, z + self.z_range/2)
        else:
            self.ax1.set_xlim(-self.xy_range, self.xy_range)
            self.ax1.set_ylim(-self.xy_range, self.xy_range)
            self.ax1.set_zlim(-self.z_range, self.z_range)

        if self.show_traj:
            self.ax1.plot(self.history['x'], self.history['y'], self.history['z'], 'b--', linewidth=1)   

        pos = np.array([x, y, z])

        # 关键：NED中yaw是绕Down轴，前左上是绕Up轴，所以yaw取反
        r = R.from_euler('ZYX', [-yaw, pitch, roll])  

        for edge, color in self.edges:
            p1 = r.apply(edge[0]) + pos
            p2 = r.apply(edge[1]) + pos
            self.ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=2)

        ## ===============Show fig2
        self.ax2.cla()
        self.ax2.set_title("Global View")
        self.ax2.plot(self.history['x'], self.history['y'], self.history['z'], 'b--', linewidth=1)   
        self.ax2.scatter(x, y, z, c='r')
        plt.pause(0.01)

def rotate_2d(vec, angle_rad):
    """2D旋转：绕垂直轴旋转 (north-east 平面)"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s],
                  [s,  c]])
    return R @ vec

# ------------------ 轨迹模块 ------------------ #
# 轨迹生成测试
def rotate_2d(vec, angle_rad):
    """2D旋转：绕垂直轴旋转 (north-east 平面)"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s],
                  [s,  c]])
    return R @ vec

class TrajectoryGenerator:
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

import numpy as np

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

class PositionController:
    def __init__(self, 
                 kp=1.0, kv=1.0,
                 HEIGHT_K_P=0.05, HEIGHT_K_I=0.01, 
                 K_MOTOR=10.0, thrust_op=0.5,
                 airspeed_ref=320.0, 
                 max_throttle=1.0, min_throttle=0.0,
                 dt=0.05):
        self.kp = kp
        self.kv = kv
        self.g_acc = 9.81

        self.HEIGHT_K_P = HEIGHT_K_P
        self.HEIGHT_K_I = HEIGHT_K_I
        self.K_MOTOR = K_MOTOR
        self.thrust_op = thrust_op
        self.airspeed_ref = airspeed_ref
        self.max_throttle = max_throttle
        self.min_throttle = min_throttle
        self.dt = dt

        self.I_ez = 0.0  # 初始化积分项

    def update(self, cur_pos_ned, cur_vel_ned, cur_rpy_ned,
                     exp_pos_ned, exp_vel_ned, exp_rpy_ned, exp_acc_ned, vehicle_airspeed):
        # 位置 & 速度误差
        pos_err = cur_pos_ned - exp_pos_ned
        vel_err = cur_vel_ned - exp_vel_ned

        # 总控制加速度 (NED系)
        controls = - self.kp * pos_err - self.kv * vel_err + exp_acc_ned

        # Body系下的加速度分量
        v_north = cur_vel_ned[0]
        v_east = cur_vel_ned[1]
        v_current = np.linalg.norm([v_north, v_east]) + 1e-6  # 防止除零

        g_acc = 9.81

        # accx_body 方向
        accx_body = (controls[0] * v_north + controls[1] * v_east) / v_current / np.cos(cur_rpy_ned[1]) - g_acc * np.sin(cur_rpy_ned[1])
        # accy_body 方向
        accy_body = (-controls[0] * v_east + controls[1] * v_north) / v_current

        print(f"accx_body: {accx_body:.2f}, accy_body: {accy_body:.2f}, v_current: {v_current:.2f}")
        accx_body = np.clip(accx_body, -10, 10)  # 限制加速度范围
        # pitch控制
        self.I_ez -= pos_err[2] * self.HEIGHT_K_I * np.clip(0.015, 0, self.dt)
        self.I_ez = np.clip(self.I_ez, -0.1, 0.1)

        pitch_target = -self.HEIGHT_K_P * pos_err[2] + self.I_ez
        pitch_target = np.clip(-pitch_target, -0.4, 0.2)

        # roll控制
        # roll_cmd = -np.arctan(accy_body / self.g_acc)
        roll_target = -np.clip(np.arctan(accy_body / self.g_acc), -0.7, 0.7)

        # yaw控制
        yaw_target = exp_rpy_ned[2]

        # throttle控制
        real_thrust_op = self.thrust_op * vehicle_airspeed / self.airspeed_ref
        throttle_cmd = accx_body / self.K_MOTOR + real_thrust_op
        throttle_cmd = np.clip(throttle_cmd, 0, 1)
        throttle_cmd = np.clip(throttle_cmd, self.min_throttle, self.max_throttle)
        

        roll_target = np.degrees(roll_target)
        pitch_target = np.degrees(pitch_target)  # 转换为度数
        yaw_target = np.degrees(yaw_target)  # 转换为度数
        
        # 输出控制指令(角度、归一化油门开度)
        return roll_target, pitch_target, yaw_target, throttle_cmd

# ------------------ F16 Functional Control Test ------------------ #
# 该脚本用于测试 F16 飞机的功能性控制
# 主要功能包括：
# 1. 初始化 F16 模型
# 2. 重置飞机初始状态
import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R


class NEDTransformer:
    """
    GPS转NED，统一在NED坐标系下工作
    """
    def __init__(self):
        # WGS84 椭球体模型
        self.ecef_transformer = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
        self.ref_ecef = None
        self.ref_lat = None
        self.ref_lon = None
        self.ref_alt = None
        self.ned_matrix = None

    def reset(self, ref_lat, ref_lon, ref_alt):
        """设置参考点 (初始点)"""
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt

        x0, y0, z0 = self.ecef_transformer.transform(ref_lon, ref_lat, ref_alt)
        self.ref_ecef = np.array([x0, y0, z0])
        self.ned_matrix = self._compute_ned_matrix(ref_lat, ref_lon)
        print(f"Reference set at lat={ref_lat}, lon={ref_lon}, alt={ref_alt}")
        print(f"Reference ECEF = {self.ref_ecef}")

    def gps_to_ned(self, lat, lon, alt):
        """单点转换：GPS -> NED"""
        x, y, z = self.ecef_transformer.transform(lon, lat, alt)
        cur_ecef = np.array([x, y, z])
        delta_ecef = cur_ecef - self.ref_ecef
        pos_ned = self.ned_matrix @ delta_ecef
        return pos_ned  # NED坐标 [North, East, Down]

    def _compute_ned_matrix(self, lat_deg, lon_deg):
        """计算NED旋转矩阵"""
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        # 注意这里矩阵已经变成 NED 格式
        ned_matrix = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon,            cos_lon,           0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
        ])
        return ned_matrix

    def update(self, current_gps, current_rpy):
        """
        核心统一接口：
        - current_gps: (lat, lon, alt)
        - current_rpy: (roll, pitch, yaw)  【假定已在NED frame下】
        """
        pos_ned = self.gps_to_ned(*current_gps)

        return pos_ned, current_rpy

# ------------------ 主程序 ------------------ #
if __name__ == "__main__":
    traj = TrajectoryGenerator(A=1000)
    viz = FixWingVisualizer(A=1000)
    for i in range(5000):
        pos, rpy = traj.get_trajectory(i * 0.05)
        viz.set_state(pos, rpy)
        viz.show()

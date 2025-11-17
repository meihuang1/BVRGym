
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Any, Optional
# from test_traj_gen import SmoothTrajectory

class FixWingVisualizer:
    """
    纯 NED 坐标可视化（North-East-Down）。
    支持：
        • 局部 3D 姿态实时展示（Fig-1）
        • 全局 3D 航迹实时展示（Fig-2）
        • 位置误差实时曲线（Fig-3）
        • 最终离线 Debug 图（位置/速度误差、控制指令等）
    """

    # ------------------------------------------------------------------ #
    #                        初始化 & 配置                                #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        A: float = 100.0,
        dt: float = 0.05,
        *,
        realtime: bool = True,      # <<< 是否实时刷新 3 个窗口
        record_debug: bool = True   # <<< 是否记录 & 绘制离线 Debug 图
    ) -> None:

        # ----------- 基本参数 ----------- #
        self.A, self.dt = A, dt
        self.realtime = realtime
        self.record_debug = record_debug

        # ----------- 尺寸、比例 ----------- #
        self.scale = A * 0.1 / 10.0
        self.xy_range = A * 1.5
        self.z_range = A * 0.5

        # ----------- 数据缓存 ----------- #
        self.t_index = 0                           # 离散时刻计数
        self.hist: Dict[str, List[float]] = {k: [] for k in
            ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'exp_x', 'exp_y', 'exp_z']}
        self.fuel_hist: List[float] = []
        self.ctrl_log: Dict[str, Any] = {}

        # ----------- 飞机几何 ----------- #
        self._build_airframe()

        # ----------- 其他需要数据 ----------#
        self.extra_log: Dict[str, List[Any]] = {}  # <<< 添加这一行（已在上条回答建议过）


        # ----------- Figure／Axes ----------- #
        if self.realtime:
            self._create_rt_figures()

    # ------------------------------------------------------------------ #
    #                        内部工具函数                                #
    # ------------------------------------------------------------------ #
    def _build_airframe(self) -> None:
        """定义简易三角机翼几何（NED系）"""
        self.nose        = np.array([10.0,  0.0,   0.0]) * self.scale
        self.rear_left   = np.array([ 0.0,  3.5,   0.0]) * self.scale
        self.rear_right  = np.array([ 0.0, -3.5,   0.0]) * self.scale
        self.rear_bottom = np.array([ 0.0,  0.0,  -2.0]) * self.scale
        self.tail_top    = np.array([ 0.0,  0.0,  -2.0]) * self.scale
        self.edges = [
            ([self.rear_left,  self.rear_right],  'red'),
            ([self.rear_left,  self.rear_bottom], 'red'),
            ([self.rear_right, self.rear_bottom], 'red'),
            ([self.nose,       self.rear_left],   'blue'),
            ([self.nose,       self.rear_right],  'blue'),
            ([self.nose,       self.rear_bottom], 'blue'),
            ([self.rear_left,  self.tail_top],    'green'),
            ([self.rear_right, self.tail_top],    'green'),
        ]

    def _create_rt_figures(self) -> None:
        """实时可视化窗口（3 个）"""
        # Fig-1：局部姿态
        self.fig1 = plt.figure("Local View")
        self.ax1  = self.fig1.add_subplot(111, projection='3d')
        self.ax1.set_xlabel('North'); self.ax1.set_ylabel('East'); self.ax1.set_zlabel('Down')

        # Fig-2：全局航迹
        self.fig2 = plt.figure("Global View")
        self.ax2  = self.fig2.add_subplot(111, projection='3d')
        self.ax2.set_xlabel('North'); self.ax2.set_ylabel('East'); self.ax2.set_zlabel('Down')

        # Fig-3：实时误差
        self.fig3  = plt.figure("Position Error (RT)")
        self.ax3_1 = self.fig3.add_subplot(311)
        self.ax3_2 = self.fig3.add_subplot(312)
        self.ax3_3 = self.fig3.add_subplot(313)

    # ------------------------------------------------------------------ #
    #                     对外接口：写入状态 / 控制日志                   #
    # ------------------------------------------------------------------ #
    def set_state(
        self,
        pos_ned: np.ndarray,
        rpy_ned: Tuple[float, float, float],
        exp_pos_ned: np.ndarray,
        *,
        others: Optional[Dict[str, Any]] = None     #  ← 这里改
    ) -> None:
        """写入状态；others 可携带任意附加数据（油量、电量、机号……）"""
        # ======== 1. 主状态写入 =========
        n, e, d    = pos_ned
        r, p, yraw = rpy_ned

        self.hist['x'].append(n);  self.hist['y'].append(e);  self.hist['z'].append(d)
        self.hist['roll'].append(r); self.hist['pitch'].append(p); self.hist['yaw'].append(yraw)

        self.hist['exp_x'].append(exp_pos_ned[0])
        self.hist['exp_y'].append(exp_pos_ned[1])
        self.hist['exp_z'].append(exp_pos_ned[2])

        # ======== 2. 额外数据写入 =========
        if others and self.record_debug:
            for k, v in others.items():
                # 若第一次见到这个 key 先建表
                self.extra_log.setdefault(k, []).append(v)

        # ======== 3. 实时绘图 =========
        if self.realtime:
            self.step()

    def set_ctrl_data(self, data: Dict[str, Any]) -> None:
        """保存控制器日志，供 show_final() 离线绘图"""
        if self.record_debug:
            self.ctrl_log = data

    # ------------------------------------------------------------------ #
    #                 实时刷新（仅在 realtime == True 时调用）            #
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        """实时刷新 3 个窗口；不需要时可不调用"""
        if not self.realtime:
            return

        # 1. Local view -------------------------------------------------- #
        self.ax1.cla()
        x, y, z = self.hist['x'][-1], self.hist['y'][-1], self.hist['z'][-1]
        r, p, yaw = self.hist['roll'][-1], self.hist['pitch'][-1], self.hist['yaw'][-1]

        # 追随 or 固定范围
        if True:
            self.ax1.set_xlim(x-self.xy_range/2, x+self.xy_range/2)
            self.ax1.set_ylim(y-self.xy_range/2, y+self.xy_range/2)
            self.ax1.set_zlim(z-self.z_range/2, z+self.z_range/2)
        # 绘局部轨迹
        self.ax1.plot(self.hist['x'], self.hist['y'], self.hist['z'], 'b--', lw=1)

        # 画机体
        rot = R.from_euler('ZYX', [yaw, p, r])
        for seg, c in self.edges:
            p1, p2 = rot.apply(seg) + np.array([x, y, z])
            self.ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=c, lw=2)

        # 2. Global view -------------------------------------------------- #
        self.ax2.cla()
        self.ax2.plot(self.hist['x'], self.hist['y'], self.hist['z'], 'b--', lw=1)
        self.ax2.plot(self.hist['exp_x'], self.hist['exp_y'], self.hist['exp_z'], 'r--', lw=1)
        self.ax2.scatter(x, y, z, c='r')

        # 3. Realtime error ---------------------------------------------- #
        t = self.t_index * self.dt
        x_err = self.hist['x'][-1] - self.hist['exp_x'][-1]
        y_err = self.hist['y'][-1] - self.hist['exp_y'][-1]
        z_err = self.hist['z'][-1] - self.hist['exp_z'][-1]

        if self.t_index == 0:  # first point
            self.err_t = [t]; self.err_x=[x_err]; self.err_y=[y_err]; self.err_z=[z_err]
        else:
            self.err_t.append(t); self.err_x.append(x_err); self.err_y.append(y_err); self.err_z.append(z_err)

        self.ax3_1.cla(); self.ax3_2.cla(); self.ax3_3.cla()
        self.ax3_1.plot(self.err_t, self.err_x, label='X'); self.ax3_1.legend(); self.ax3_1.set_ylabel('X err [m]')
        self.ax3_2.plot(self.err_t, self.err_y, label='Y'); self.ax3_2.legend(); self.ax3_2.set_ylabel('Y err [m]')
        self.ax3_3.plot(self.err_t, self.err_z, label='Z'); self.ax3_3.legend(); self.ax3_3.set_ylabel('Z err [m]')
        self.ax3_3.set_xlabel('Time [s]')

        self.t_index += 1
        plt.pause(0.01)

    # ------------------------------------------------------------------ #
    #                      离线 Debug 图（仿真结束后）                    #
    # ------------------------------------------------------------------ #
    def show_final(self) -> None:
        """绘制总航迹 + Debug 曲线（record_debug==True 时）"""

        # ------- 航迹 -------- #
        fig = plt.figure("Global Trajectory (final)")
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('North'); ax.set_ylabel('East'); ax.set_zlabel('Down')
        ax.plot(self.hist['x'], self.hist['y'], self.hist['z'], 'b--', lw=1, label='Real')
        ax.plot(self.hist['exp_x'], self.hist['exp_y'], self.hist['exp_z'], 'r--', lw=1, label='Reference')
        ax.legend()

        if not self.record_debug or not self.ctrl_log:
            plt.show(); return

        # ------- Debug 曲线 -------- #
        times = np.array(self.ctrl_log['times'])
        fig2, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        fig2.suptitle("Position Controller Debug")

        # ① 位置误差
        pos_err = np.array(self.ctrl_log['pos_errors'])
        axs[0].plot(times, pos_err); axs[0].set_ylabel("Pos err [m]"); axs[0].legend(['N','E','D']); axs[0].grid()

        # ② 速度误差
        vel_err = np.array(self.ctrl_log['vel_errors'])
        axs[1].plot(times, vel_err); axs[1].set_ylabel("Vel err [m/s]"); axs[1].legend(['Vn','Ve','Vd']); axs[1].grid()

        # ③ 控制加速度
        ctrl_acc = np.array(self.ctrl_log['controls'])
        axs[2].plot(times, ctrl_acc); axs[2].set_ylabel("Ctrl Acc [m/s²]"); axs[2].legend(['ax','ay','az']); axs[2].grid()

        # ④ 油门 + RPY
        throttle = np.array(self.ctrl_log['throttle_cmd'])
        rpy      = np.array(self.ctrl_log['rpy_cmd'])
        axs[3].plot(times, rpy);       axs[3].legend(['roll','pitch','yaw'])
        ax_t = axs[3].twinx()
        ax_t.plot(times, throttle, 'k-', label='Throttle'); ax_t.set_ylabel("Throttle"); ax_t.legend(loc="upper right")

        # ⑤ 油料（可选）
        if 'fuel' in self.extra_log:
            fig3, ax_fuel = plt.subplots(1, 1, figsize=(10, 3))
            fuel = np.array(self.extra_log['fuel'])
            ax_fuel.plot(times, fuel, 'g-', label='Fuel')
            ax_fuel.set_title("Fuel Consumption")
            ax_fuel.set_ylabel("Fuel [lbs]")
            ax_fuel.set_xlabel("Time [s]")
            ax_fuel.grid()
            ax_fuel.legend()



        plt.tight_layout()
        plt.show()










# # ------------------ 模块测试 ------------------ #
# if __name__ == "__main__":
#     r = 10000
#     t = np.linspace(0, 10*np.pi, r)
#     waypoints = np.array([
#         r * np.cos(t - np.pi/2) + r,
#         r * np.sin(t - np.pi/2) + r,
#         0 * np.ones_like(t)
#     ]).T  # 1000 x 3

#     traj = SmoothTrajectory(waypoints, airspeed=325)
#     print("Total time:", traj.total_time)

#     viz = FixWingVisualizer(A=1000)
#     for i in range(5000):
#         pos, vel, acc, rpy = traj.get_ref(i * 0.1)
#         # pos, rpy = traj.get_trajectory(i * 0.05)
#         viz.set_state(pos, rpy, (0,0,0))
#         viz.step()

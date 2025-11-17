# fleet.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation as R
from UtilsLib.aircraft_unit import AircraftUnit

class Fleet:
    """管理一组 AircraftUnit，并统一日志与可视化."""
    def __init__(self, conf=None, realtime=False, record_debug=True):
        self.conf = conf
        self.dt, self.inner_steps = conf.dt, conf.general['r_pid']
        self.units: list[AircraftUnit] = self._make_units(realtime, record_debug)

        # —— 集中存放 log，方便后期分析/作图 ——
        self.traj_log = {idx: [] for idx in range(len(self.units))}      # 实际轨迹
        self.exp_traj_log = {idx: [] for idx in range(len(self.units))}  # 期望轨迹
        self.rpy_log = {idx: [] for idx in range(len(self.units))}       # 姿态角

    # ------------------------------------------------------------------
    def _make_units(self, realtime, record_debug):
        units = []
        for idx in range(self.conf.num_aircraft):
            key  = f'aircraft{idx+1}'
            init = self.conf.init_state[key]

            u = AircraftUnit()
            u.init_f16(conf = self.conf,
                       lat=init['lat'], lon=init['lon'],
                       alt=init['alt'], vel=init['vel'],
                       heading=init['heading'])
            u.init_pos_ctrl(conf = self.conf)
            u.init_viz(A=1000, dt=self.dt,
                       realtime=realtime, record_debug=record_debug)
            u.set_cfg(self.conf)
            units.append(u)
        return units
    # ------------------------------------------------------------------
    def step(self, k):
        """执行单个仿真步并记录航迹."""
        for idx, ac in enumerate(self.units):
            pos, rpy = ac.get_pos_ned()
            vel      = ac.get_vel_ned()

            # —— 起飞阶段：给一点操纵量让飞机加速 —
            if k < 50:
                alt_start=self.conf.init_state[f'aircraft{idx+1}']['alt']
                alt_max, alt_min = self.conf.ctrl['alt_max'], self.conf.ctrl['alt_min']
                if alt_max == alt_min:
                    raise ValueError("alt_max and alt_min must not be equal")
                alt_ctrl = -1 + (alt_start - alt_min)/(alt_max - alt_min) * 2
                ac.f16.step_BVR([1, alt_ctrl, 0.3], 0)
                
            # —— build traj（一次性） —
            if k == 50 and ac.first_switch:
                ac.build_traj(pos, vel,
                              leg_lenth_ref=10_000, h_ref=400,
                              v_ref=max(30.0, np.linalg.norm(vel[:2])))
                ac.f16.fdm.set_dt(self.dt / self.inner_steps)

            # —— 期望量 —
            if ac.traj:
                tau = (k - 50) * self.dt
                exp_p, exp_v, exp_a = ac.traj.get_minco_pva(tau)
                exp_rpy = np.zeros(3)
            else:
                exp_p, exp_v, exp_a, exp_rpy = pos, vel, np.zeros(3), np.zeros(3)

            # —— 控制 —
            roll, pitch, yaw, thr = ac.pos_ctrl.update(
                pos, vel, rpy, exp_p, exp_v, exp_rpy, exp_a)
            ac.f16.set_roll_PID(roll)
            ac.f16.set_pitch_PID(pitch)
            ac.f16.set_yaw_PID(psi_ref=yaw)
            ac.f16.set_throttle(thr)

            for _ in range(self.inner_steps):
                ac.f16.fdm.run()

            # —— 记录 / 可视化 —
            ac.viz.set_state(pos_ned=pos, rpy_ned=rpy, exp_pos_ned=exp_p)
            self.traj_log[idx].append(pos.copy())
            self.exp_traj_log[idx].append(exp_p.copy())
            self.rpy_log[idx].append(rpy.copy())

    # ------------------------------------------------------------------
    def show_all_final(self):
        """显示所有飞机的最终调试图（位置/速度误差、控制指令等）"""
        for idx, ac in enumerate(self.units):
            print(f"生成飞机 {idx} 的调试图...")
            # 保存控制器日志数据
            ac.viz.set_ctrl_data(ac.pos_ctrl.get_log_data())
            # 显示最终调试图
            ac.viz.show_final()
    
    # ------------------------------------------------------------------
    def plot_all_static(self):
        """一次性把全部航迹画在同一张 3D 图里（实际轨迹 vs 期望轨迹）"""
        fig = plt.figure(figsize=(14, 7))
        
        # 左图：实际轨迹 vs 期望轨迹
        ax1 = fig.add_subplot(121, projection='3d')
        colors = plt.cm.get_cmap('tab10', len(self.units))

        for idx in range(len(self.units)):
            # 实际轨迹（实线）
            p = np.array(self.traj_log[idx])
            ax1.plot(p[:, 0], p[:, 1], p[:, 2],
                    color=colors(idx), linewidth=2, label=f'AC{idx} 实际')
            ax1.scatter(p[0, 0], p[0, 1], p[0, 2],
                       marker='^', s=100, color=colors(idx))
            
            # 期望轨迹（虚线）
            if len(self.exp_traj_log[idx]) > 0:
                p_exp = np.array(self.exp_traj_log[idx])
                ax1.plot(p_exp[:, 0], p_exp[:, 1], p_exp[:, 2],
                        color=colors(idx), linewidth=1, linestyle='--', 
                        alpha=0.5, label=f'AC{idx} 期望')
        
        ax1.set_xlabel('North [m]')
        ax1.set_ylabel('East [m]')
        ax1.set_zlabel('Down [m]')
        ax1.set_title('航迹对比：实际 vs 期望', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.invert_zaxis()
        
        # 右图：只显示实际轨迹（更清晰）
        ax2 = fig.add_subplot(122, projection='3d')
        for idx in range(len(self.units)):
            p = np.array(self.traj_log[idx])
            ax2.plot(p[:, 0], p[:, 1], p[:, 2],
                    color=colors(idx), linewidth=2, label=f'AC{idx}')
            ax2.scatter(p[0, 0], p[0, 1], p[0, 2],
                       marker='^', s=100, color=colors(idx))
            ax2.scatter(p[-1, 0], p[-1, 1], p[-1, 2],
                       marker='*', s=200, color=colors(idx))
        
        ax2.set_xlabel('North [m]')
        ax2.set_ylabel('East [m]')
        ax2.set_zlabel('Down [m]')
        ax2.set_title('实际航迹（起点△ 终点★）', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.invert_zaxis()
        
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def animate(self, interval=50, ac_idx=0, save_gif=False,
                gif_filename1='trajectory.gif', gif_filename2='attitude.gif'):
        """
        双窗口动画：轨迹动画 + 飞机姿态动画，可保存为GIF
        
        参数:
            interval: 帧间隔（毫秒），越小播放越快
            ac_idx: 显示哪架飞机的姿态（默认第0架）
            save_gif: 是否保存为GIF文件
            gif_filename1: 轨迹动画GIF文件名
            gif_filename2: 姿态动画GIF文件名
        """
        def ned_point_to_plot(pt: np.ndarray) -> np.ndarray:
            arr = np.array(pt, dtype=float).copy()
            arr[2] = -arr[2]  # Down -> Altitude
            return arr

        def ned_path_to_plot(path: list[np.ndarray]) -> np.ndarray:
            arr = np.array(path, dtype=float).copy()
            if arr.ndim == 1:
                return ned_point_to_plot(arr)
            arr[:, 2] = -arr[:, 2]
            return arr

        # ========== 窗口1：轨迹动画（只显示期望轨迹+实际位置） ========== #
        fig1 = plt.figure("轨迹动画", figsize=(12, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        
        # 期望轨迹（红色半透明虚线，一直显示全部）
        for idx in range(len(self.units)):
            if len(self.exp_traj_log[idx]) > 0:
                p_exp_full = ned_path_to_plot(self.exp_traj_log[idx])
                ax1.plot(p_exp_full[:, 0], p_exp_full[:, 1], p_exp_full[:, 2],
                        'r--', linewidth=2, alpha=0.4, label='期望轨迹' if idx == 0 else '')
        
        # 实际位置（蓝色圆点，运动）
        actual_scatters = [ax1.plot([], [], [], 'o', color='blue', 
                           markersize=14, markeredgecolor='darkblue',
                           markeredgewidth=1.5, label='实际位置' if idx == 0 else '')[0]
                    for idx in range(len(self.units))]
        
        title_text1 = ax1.text2D(
            0.05, 0.95, '', transform=ax1.transAxes,
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # ========== 窗口2：飞机姿态动画（显示期望轨迹+实际轨迹） ========== #
        fig2 = plt.figure("飞机姿态", figsize=(10, 8))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        # 期望轨迹（红色虚线）
        exp_trail_line, = ax2.plot([], [], [], 'r--', linewidth=2, alpha=0.5, label='Reference Path')
        
        # 实际轨迹（蓝色实线）
        actual_trail_line, = ax2.plot([], [], [], 'b-', linewidth=2, alpha=0.7, label='Actual Path')
        
        # 飞机几何（简化模型）
        scale = 10.0
        nose = np.array([10.0, 0.0, 0.0]) * scale
        rear_left = np.array([0.0, 3.5, 0.0]) * scale
        rear_right = np.array([0.0, -3.5, 0.0]) * scale
        rear_bottom = np.array([0.0, 0.0, -2.0]) * scale
        tail_top = np.array([0.0, 0.0, -2.0]) * scale
        
        edges = [
            ([rear_left, rear_right], 'red'),
            ([rear_left, rear_bottom], 'red'),
            ([rear_right, rear_bottom], 'red'),
            ([nose, rear_left], 'blue'),
            ([nose, rear_right], 'blue'),
            ([nose, rear_bottom], 'blue'),
            ([rear_left, tail_top], 'green'),
            ([rear_right, tail_top], 'green'),
        ]
        
        # 姿态线条
        attitude_lines = []
        for seg, c in edges:
            line, = ax2.plot([], [], [], color=c, linewidth=2)
            attitude_lines.append(line)
        
        title_text2 = ax2.text2D(
            0.05, 0.95, '', transform=ax2.transAxes,
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # ========== 初始化函数 ========== #
        def init():
            # 窗口1初始化
            all_pos = np.concatenate([np.array(t) for t in self.traj_log.values()])
            margin_xy = 5000
            margin_z = 500
            ax1.set_xlim(all_pos[:, 0].min() - margin_xy, all_pos[:, 0].max() + margin_xy)
            ax1.set_ylim(all_pos[:, 1].min() - margin_xy, all_pos[:, 1].max() + margin_xy)
            altitudes = -all_pos[:, 2]
            ax1.set_zlim(altitudes.min() - margin_z, altitudes.max() + margin_z)
            ax1.set_xlabel('North [m]', fontsize=11)
            ax1.set_ylabel('East [m]', fontsize=11)
            ax1.set_zlabel('Altitude [m]', fontsize=11)
            ax1.set_title('Trajectory Animation (red dashed = reference, blue dot = actual)',
                          fontsize=13, fontweight='bold', pad=15)
            ax1.legend(loc='upper right', fontsize=9)
            
            # 窗口2初始化（会在update中动态设置）
            ax2.set_xlabel('North [m]', fontsize=11)
            ax2.set_ylabel('East [m]', fontsize=11)
            ax2.set_zlabel('Altitude [m]', fontsize=11)
            ax2.set_title(f'Aircraft {ac_idx} Attitude (reference vs actual path)',
                          fontsize=13, fontweight='bold', pad=15)
            ax2.legend(loc='upper right', fontsize=9)
            
            return actual_scatters + attitude_lines + [exp_trail_line, actual_trail_line, title_text1, title_text2]
        
        # ========== 更新函数 ========== #
        def update(frame):
            skip_frames = max(1, len(self.traj_log[0]) // 500)
            actual_frame = frame * skip_frames
            time_sec = actual_frame * self.dt
            
            # 窗口1更新：轨迹动画（只显示期望轨迹+实际位置）
            title_text1.set_text(
                f'Time: {time_sec:.1f}s  |  Frame: {actual_frame}/{max(len(t) for t in self.traj_log.values())}')
            
            for i in range(len(self.units)):
                if actual_frame < len(self.traj_log[i]):
                    # 实际位置（蓝点）
                    p_actual = ned_point_to_plot(self.traj_log[i][actual_frame])
                    actual_scatters[i].set_data([p_actual[0]], [p_actual[1]])
                    actual_scatters[i].set_3d_properties([p_actual[2]])
            
            # 窗口2更新：飞机姿态（显示期望轨迹+实际轨迹）
            if actual_frame < len(self.traj_log[ac_idx]) and actual_frame < len(self.rpy_log[ac_idx]):
                pos = self.traj_log[ac_idx][actual_frame]
                rpy = self.rpy_log[ac_idx][actual_frame]  # [roll, pitch, yaw]
                
                # 更新标题
                title_text2.set_text(
                    f'Time: {time_sec:.1f}s  |  Roll: {np.degrees(rpy[0]):.1f}°  '
                    f'Pitch: {np.degrees(rpy[1]):.1f}°  Yaw: {np.degrees(rpy[2]):.1f}°')
                
                # 相机跟随（局部视角）
                xy_range = 2000
                z_range = 1000
                ax2.set_xlim(pos[0] - xy_range/2, pos[0] + xy_range/2)
                ax2.set_ylim(pos[1] - xy_range/2, pos[1] + xy_range/2)
                alt = -pos[2]
                ax2.set_zlim(alt - z_range/2, alt + z_range/2)
                
                # 绘制期望轨迹（局部视图中的一段）
                if actual_frame > 0:
                    trail_start = max(0, actual_frame - 200)
                    # 期望轨迹
                    if len(self.exp_traj_log[ac_idx]) > 0:
                        exp_trail_data = ned_path_to_plot(
                            self.exp_traj_log[ac_idx][trail_start:actual_frame+1])
                        if len(exp_trail_data) > 0:
                            exp_trail_line.set_data(exp_trail_data[:, 0], exp_trail_data[:, 1])
                            exp_trail_line.set_3d_properties(exp_trail_data[:, 2])
                    
                    # 实际轨迹
                    actual_trail_data = ned_path_to_plot(
                        self.traj_log[ac_idx][trail_start:actual_frame+1])
                    if len(actual_trail_data) > 0:
                        actual_trail_line.set_data(actual_trail_data[:, 0], actual_trail_data[:, 1])
                        actual_trail_line.set_3d_properties(actual_trail_data[:, 2])
                
                # 绘制飞机姿态
                rot = R.from_euler('ZYX', [rpy[2], rpy[1], rpy[0]])  # yaw, pitch, roll
                for line, (seg, c) in zip(attitude_lines, edges):
                    p1, p2 = rot.apply(seg) + pos
                    p1_plot = ned_point_to_plot(p1)
                    p2_plot = ned_point_to_plot(p2)
                    line.set_data([p1_plot[0], p2_plot[0]], [p1_plot[1], p2_plot[1]])
                    line.set_3d_properties([p1_plot[2], p2_plot[2]])
            
            return actual_scatters + attitude_lines + [exp_trail_line, actual_trail_line, title_text1, title_text2]
        
        # ========== 启动动画 ========== #
        max_frames = max(len(t) for t in self.traj_log.values())
        skip_frames = max(1, max_frames // 500)
        total_frames = max_frames // skip_frames
        
        ani1 = FuncAnimation(fig1, update, frames=total_frames,
                             init_func=init, blit=False, interval=interval, repeat=True)
        ani2 = FuncAnimation(fig2, update, frames=total_frames,
                             init_func=init, blit=False, interval=interval, repeat=True)
        
        # 保存为GIF
        if save_gif:
            print(f"\n正在保存GIF文件...")
            print(f"  - 轨迹动画: {gif_filename1}")
            writer = PillowWriter(fps=int(1000/interval))
            ani1.save(gif_filename1, writer=writer)
            print(f"  - 姿态动画: {gif_filename2}")
            ani2.save(gif_filename2, writer=writer)
            print("GIF保存完成！")
        else:
            plt.show()
    # ------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from UtilsLib.test_traj_gen import *
# ------------------ 可视化模块 ------------------ #

# class FixWingVisualizer_:
#     def __init__(self, A=100, follow=True, show_traj=True, dt = 0.05):
#         """
#         纯NED坐标输入，前左上显示系输出
#         """
#         self.dt = dt
#         self.T = 0  # 时间计数器
        
#         self.A = A
#         self.scale = A * 0.1 / 10.0
#         self.follow = follow
#         self.show_traj = show_traj
#         self._init_model()
#         self.history = {key: [] for key in ['x','y','z','roll','pitch','yaw','exp_x','exp_y','exp_z']}
#         self.state = [0,0,0,0,0,0]
        
#         # fig1 局部观察姿态
#         self.fig1 = plt.figure()
#         self.ax1 = self.fig1.add_subplot(111, projection='3d')
#         self.ax1.set_xlabel('Front')
#         self.ax1.set_ylabel('Left')
#         self.ax1.set_zlabel('Up')
#         self.xy_range = A * 1.5
#         self.z_range = A * 0.5
#         self.ax1.set_xlim(-self.xy_range, self.xy_range)
#         self.ax1.set_ylim(-self.xy_range, self.xy_range)

#         # fig2 全局观察路径
#         self.fig2 = plt.figure()
#         self.ax2 = self.fig2.add_subplot(111, projection='3d')
        
#         self.fig3 = plt.figure("Position Errors")
#         self.ax3_1 = self.fig3.add_subplot(311)
#         self.ax3_2 = self.fig3.add_subplot(312)
#         self.ax3_3 = self.fig3.add_subplot(313)

#         self.pos_error_times = []  # 时间轴
#         self.pos_err_x = []
#         self.pos_err_y = []
#         self.pos_err_z = []
#         self.fuel_data = []
#         self.data = {}

        
#         self.first_frame = True
#         # self.fig3 = plt.figure()
#         # self.ax4 = self.fig3.add_subplot(111, projection='3d')
        
#     def _init_model(self):
#         # self.nose = np.array([10.0, 0.0, 0.0]) * self.scale
#         # self.rear_left = np.array([0.0, -3.5,  0.0]) * self.scale
#         # self.rear_right = np.array([0.0, 3.5,  0.0]) * self.scale
#         # self.rear_bottom = np.array([0.0, 0.0, 2.0]) * self.scale
#         # self.tail_top = np.array([0.0, 0.0, 2]) * self.scale
#         self.nose = np.array([10.0, 0.0, 0.0]) * self.scale
#         self.rear_left = np.array([0.0, 3.5,  0.0]) * self.scale  # y 正表示 East
#         self.rear_right = np.array([0.0, -3.5,  0.0]) * self.scale
#         self.rear_bottom = np.array([0.0, 0.0, -2.0]) * self.scale  # z 负表示 Down
#         self.tail_top = np.array([0.0, 0.0, -2.0]) * self.scale
#         self.edges = [
#             ([self.rear_left, self.rear_right], 'red'),
#             ([self.rear_left, self.rear_bottom], 'red'),
#             ([self.rear_right, self.rear_bottom], 'red'),
#             ([self.nose, self.rear_left], 'blue'),
#             ([self.nose, self.rear_right], 'blue'),
#             ([self.nose, self.rear_bottom], 'blue'),
#             ([self.rear_left, self.tail_top], 'green'),
#             ([self.rear_right, self.tail_top], 'green'),
#         ]

#     # def ned_to_display(self, pos):
#     #     """
#     #     NED --> 前左上
#     #     """
#     #     x_n, y_e, z_d = pos
#     #     return np.array([x_n, -y_e, -z_d])

#     def set_state(self, pos_ned, rpy_ned, exp_pos_ned):

#         # exp_pos_f = self.ned_to_display(exp_pos_ned)
#         # x, y, z = self.ned_to_display(pos_ned)
        
#         x,y,z = pos_ned[0], pos_ned[1], pos_ned[2]
#         roll, pitch, yaw = rpy_ned
#         self.state = [x, y, z, roll, pitch, yaw]
#         self.history['x'].append(x)
#         self.history['y'].append(y)
#         self.history['z'].append(z)
#         self.history['roll'].append(roll)
#         self.history['pitch'].append(pitch)
#         self.history['yaw'].append(yaw)
        
#         self.history['exp_x'].append(exp_pos_ned[0])
#         self.history['exp_y'].append(exp_pos_ned[1])
#         self.history['exp_z'].append(exp_pos_ned[2])
    
#     def set_fuel_data(self, fuel_data):
#         """
#         设置燃油数据
#         """
#         self.fuel_data.append(fuel_data)
#     def set_equal_aspect_3d(self, ax):
#         #  """强制让 Axes3D 的 x, y, z 轴比例一致"""
#         x_limits = ax.get_xlim3d()
#         y_limits = ax.get_ylim3d()
#         z_limits = ax.get_zlim3d()

#         x_range = abs(x_limits[1] - x_limits[0])
#         y_range = abs(y_limits[1] - y_limits[0])
#         z_range = abs(z_limits[1] - z_limits[0])
#         max_range = max(x_range, y_range, z_range)

#         x_middle = np.mean(x_limits)
#         y_middle = np.mean(y_limits)
#         z_middle = np.mean(z_limits)

#         ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
#         ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
#         ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


#     def show(self):
#         # ------------------ Show Fig 1 ------------------ #
#         self.ax1.cla()
#         # self.ax1.set_xlabel('Front')
#         # self.ax1.set_ylabel('Left')
#         # self.ax1.set_zlabel('Up')
#         self.ax1.set_xlabel('North')
#         self.ax1.set_ylabel('East')
#         self.ax1.set_zlabel('Down')
#         x,y,z,roll,pitch,yaw = self.state

#         if self.follow:
#             self.ax1.set_xlim(x - self.xy_range/2, x + self.xy_range/2)
#             self.ax1.set_ylim(y - self.xy_range/2, y + self.xy_range/2)
#             self.ax1.set_zlim(z - self.z_range/2, z + self.z_range/2)
#         else:
#             self.ax1.set_xlim(-self.xy_range, self.xy_range)
#             self.ax1.set_ylim(-self.xy_range, self.xy_range)
#             self.ax1.set_zlim(-self.z_range, self.z_range)

#         if self.show_traj:
#             self.ax1.plot(self.history['x'], self.history['y'], self.history['z'], 'b--', linewidth=1)   

#         pos = np.array([x, y, z])

#         # 关键：NED中yaw是绕Down轴，前左上是绕Up轴，所以yaw取反
#         # r = R.from_euler('ZYX', [-yaw, pitch, roll])  
#         r = R.from_euler('ZYX', [yaw, pitch, roll])  # 保持 NED 坐标的真实方向

#         for edge, color in self.edges:
#             p1 = r.apply(edge[0]) + pos
#             p2 = r.apply(edge[1]) + pos
#             self.ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=2)

#        # ------------------ Show Fig 2 ------------------ #
#         self.ax2.cla()
#         self.ax2.set_title("Global View")
#         self.ax2.set_xlabel('North')
#         self.ax2.set_ylabel('East')
#         self.ax2.set_zlabel('Down')        
#         self.ax2.plot(self.history['x'], self.history['y'], self.history['z'], 'b--', linewidth=1)  
#         self.ax2.plot(self.history['exp_x'], self.history['exp_y'], self.history['exp_z'], 'r--', linewidth=1)    
#         self.ax2.scatter(x, y, z, c='r')
#         # self.set_equal_aspect_3d(self.ax2)
        
#         # ------------------ Show Fig 3 ------------------ #
#         self.pos_error_times.append(self.T * self.dt)
#         if self.first_frame:
#             # 删除第一个历史点
#             for key in self.history:
#                 if len(self.history[key]) > 0:
#                     self.history[key].pop(0)
#             if len(self.pos_error_times) > 0:
#                 self.pos_error_times.pop(0)
#             self.first_frame = False  # 不再删除
#         x_err = np.array(self.history['x']) - np.array(self.history['exp_x'])
#         y_err = np.array(self.history['y']) - np.array(self.history['exp_y'])
#         z_err = np.array(self.history['z']) - np.array(self.history['exp_z'])

#         self.ax3_1.cla()
#         self.ax3_1.plot(self.pos_error_times, x_err, label='X Error')
#         self.ax3_1.set_ylabel("X Error [m]")
#         self.ax3_1.legend()

#         self.ax3_2.cla()
#         self.ax3_2.plot(self.pos_error_times, y_err, label='Y Error')
#         self.ax3_2.set_ylabel("Y Error [m]")
#         self.ax3_2.legend()

#         self.ax3_3.cla()
#         self.ax3_3.plot(self.pos_error_times, z_err, label='Z Error')
#         self.ax3_3.set_ylabel("Z Error [m]")
#         self.ax3_3.set_xlabel("Time [s]")
#         self.ax3_3.legend()

#         self.T += 1
#         plt.pause(0.005)
        
        
#     # ------------------------Shows Final Path ------------------------
#     def show_final(self):
        
#         # ---------------------- 3D Position Controller Debug Info ----------------------  
#         self.ax2.cla()
#         self.ax2.set_title("Global View")
#         self.ax2.set_xlabel('North')
#         self.ax2.set_ylabel('East')
#         self.ax2.set_zlabel('Down')        
#         self.ax2.plot(self.history['x'], self.history['y'], self.history['z'], 'b--', linewidth=1)  
#         self.ax2.plot(self.history['exp_x'], self.history['exp_y'], self.history['exp_z'], 'r--', linewidth=1)    
#         # self.ax2.scatter(x, y, z, c='r')
#         # self.ax2.legend()
#         # print(self.history['x'], self.history['y'], self.history['z'])
#         # plt.pause(1)
        
#         # ---------------------- scalar error
#         times = self.data['times']

#         fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
#         fig.suptitle("Position Controller Debug Info", fontsize=16)

#         # 1. 位置误差
#         pos_errors = np.array(self.data['pos_errors'])
#         axs[0].plot(times, pos_errors[:, 0], label='North Error')
#         axs[0].plot(times, pos_errors[:, 1], label='East Error')
#         axs[0].plot(times, pos_errors[:, 2], label='Down Error')
#         axs[0].set_ylabel("Pos Error [m]")
#         axs[0].legend()
#         axs[0].grid(True)

#         # 2. 速度误差
#         vel_errors = np.array(self.data['vel_errors'])
#         axs[1].plot(times, vel_errors[:, 0], label='V_North Error')
#         axs[1].plot(times, vel_errors[:, 1], label='V_East Error')
#         axs[1].plot(times, vel_errors[:, 2], label='V_Down Error')
#         axs[1].set_ylabel("Vel Error [m/s]")
#         axs[1].legend()
#         axs[1].grid(True)

#         # 3. 控制加速度（NED）
#         controls = np.array(self.data['controls'])
#         axs[2].plot(times, controls[:, 0], label='Ax')
#         axs[2].plot(times, controls[:, 1], label='Ay')
#         axs[2].plot(times, controls[:, 2], label='Az')
#         axs[2].set_ylabel("Control Acc [m/s²]")
#         axs[2].legend()
#         axs[2].grid(True)

#         # 4. 油门 + RPY命令
#         rpy_cmd = np.array(self.data['rpy_cmd']) if 'rpy_cmd' in self.data else np.zeros((len(times), 3))
#         throttle = np.array(self.data['throttle_cmd'])
#         fuel = np.array(self.fuel_data)
#         fig2, axs2 = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
#         axs2[0].plot(times, rpy_cmd[:, 0], label='Roll [deg]')
#         axs2[0].plot(times, rpy_cmd[:, 1], label='Pitch [deg]')
#         axs2[0].plot(times, rpy_cmd[:, 2], label='Yaw [deg]')
#         axs2[0].set_ylabel("RPY Cmd [deg]")
#         axs2[0].legend()
#         axs2[0].grid(True)

#         axs2[1].plot(times, throttle, label='Throttle Cmd')
#         axs2[1].set_ylabel("Throttle [0-1]")
#         axs2[1].set_xlabel("Time [s]")
#         axs2[1].legend()
#         axs2[1].grid(True)
        
#         # axs2[2].plot(times, fuel, label='Fuel data')
#         # axs2[2].set_ylabel("fuel [0-1]")
#         # axs2[2].set_xlabel("Time [s]")
#         # axs2[2].legend()
#         # axs2[2].grid(True)
        
#         plt.tight_layout()
#         plt.show()
            
        
#     def set_ctrl_data(self, data):
#         self.data = data



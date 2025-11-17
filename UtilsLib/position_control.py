import numpy as np
import matplotlib.pyplot as plt
from UtilsLib.Tools import *
class PositionController:
    def __init__(self, conf):
                # 一些好的 参数
                #  kpx=0.125, kpy = 0.125, kvx=0.02, kvy = 0.2,
                #  HEIGHT_K_P=0.01, HEIGHT_K_I=0.005, 
                #  K_MOTOR=10.0, thrust_op=0.49,

                #  dt=0.01):
        self.conf = conf
        
        self.g_acc = 9.81
        
        dt = self.conf.dt
        
        kpx = self.conf.pid_pos_ctrl['kpx']
        kpy = self.conf.pid_pos_ctrl['kpy']
        kvx = self.conf.pid_pos_ctrl['kvx']
        kvy = self.conf.pid_pos_ctrl['kvy']
        HEIGHT_K_P = self.conf.pid_pos_ctrl['HEIGHT_K_P']
        HEIGHT_K_I = self.conf.pid_pos_ctrl['HEIGHT_K_I']
        K_MOTOR = self.conf.pid_pos_ctrl['K_MOTOR']
        thrust_op = self.conf.pid_pos_ctrl['thrust_op']

        self.min_roll = self.conf.pid_pos_ctrl['min_roll']
        self.max_roll = self.conf.pid_pos_ctrl['max_roll']
        self.min_pitch = self.conf.pid_pos_ctrl['min_pitch']
        self.max_pitch = self.conf.pid_pos_ctrl['max_pitch']
        
        self.HEIGHT_I_MAX = self.conf.pid_pos_ctrl['HEIGHT_I_MAX']
        
        self.kp = np.array([kpx, kpy, 0.0])  # 位置控制增益
        self.kv = np.array([kvx, kvy, 0.0])  # 速度控制增益

        self.HEIGHT_K_P = HEIGHT_K_P
        self.HEIGHT_K_I = HEIGHT_K_I
        self.K_MOTOR = K_MOTOR
        self.thrust_op = thrust_op
        self.dt = dt

        self.I_ez = 0.0  # 初始化积分项

        self.pos_errors = []
        self.vel_errors = []
        self.controls = []
        self.times = []
                
        self.T = 0
        
        self.acc_body = []  # Body系下的加速度分量
        self.rpy_cmd = []
        self.thrust_cmd = []
        
        self.data ={
            'pos_errors': self.pos_errors,
            'vel_errors': self.vel_errors,
            'controls': self.controls,
            'times': self.times,
            'acc_body':self.acc_body,
            'rpy_cmd':self.rpy_cmd,
            'throttle_cmd':self.thrust_cmd
        }
        
    def update(self, cur_pos_ned, cur_vel_ned, cur_rpy_ned,
                     exp_pos_ned, exp_vel_ned, exp_rpy_ned, 
                     exp_acc_ned):
        # 位置 & 速度误差
        pos_err = cur_pos_ned - exp_pos_ned
        vel_err = cur_vel_ned - exp_vel_ned
        g_acc = 9.81
        
        # 总控制加速度 (NED系)
        controls = - self.kp * pos_err - self.kv * vel_err + exp_acc_ned
        self.pos_errors.append(pos_err)
        self.vel_errors.append(vel_err)
        self.controls.append(controls)
        self.T += self.dt
        self.times.append(self.T)
        
        # Body系下的加速度分量
        v_north = cur_vel_ned[0]
        v_east = cur_vel_ned[1]
        # 只计算xy的 速度分量
        v_current = np.linalg.norm([v_north, v_east]) + 1e-6  # 防止除零

        # accx_body 方向
        accx_body = (controls[0] * v_north + controls[1] * v_east) / v_current / np.cos(cur_rpy_ned[1]) - g_acc * np.sin(cur_rpy_ned[1])
        # print("cos theta ",  np.cos(cur_rpy_ned[1]))
        # accy_body 方向
        accy_body = (-controls[0] * v_east + controls[1] * v_north) / v_current
        
        self.acc_body.append([accx_body, accy_body])
        
        # 根据 滚转角和横向加速度 限制 accy_body 的范围
        # accy_body = np.clip(accy_body, -g_acc * np.tan(np.radians(35)), g_acc * np.tan(np.radians(35)))

        # pitch控制
        self.I_ez -= pos_err[2] * self.HEIGHT_K_I * np.clip(0.015, 0, self.dt)
        self.I_ez = np.clip(self.I_ez, -self.HEIGHT_I_MAX, self.HEIGHT_I_MAX)
        pitch_target = -self.HEIGHT_K_P * pos_err[2] + self.I_ez
        
        pitch_target = np.clip(-pitch_target, self.min_pitch, self.max_pitch)
        # roll控制
        # roll_target = -np.clip(np.arctan(accy_body / self.g_acc), -0.7, 0.7)
        roll_target = np.clip(np.arctan(accy_body / self.g_acc), self.min_roll, self.max_roll)

        # yaw控制
        # =====  跟踪 将 yaw朝向 调整为 期望点
        # delta = exp_pos_ned[:2] - cur_pos_ned[:2]
        # normalize_angle_rad(delta)
        # yaw_target = np.arctan2(delta[1], delta[0])  # 朝向目标点
        
        # ===== 跟踪 并将 yaw朝向变成期望速度
        vx, vy = exp_vel_ned[:2]
        if np.linalg.norm([vx, vy]) > 1e-3:
            yaw_target = np.arctan2(vy, vx)
        else:
            # 如果速度非常小，退回原先的方向（如不变）
            yaw_target = cur_rpy_ned[2]  # 或者 yaw_target = 0

        # throttle控制
        real_thrust_op = self.thrust_op * np.linalg.norm(cur_vel_ned) / np.linalg.norm(exp_vel_ned)
        
        # throttle_cmd limited by conf.ctrl.thr_max and conf.ctrl.thr_min 
        throttle_cmd = accx_body / self.K_MOTOR + real_thrust_op
        
        # throttle_cmd = (np.linalg.norm(exp_vel_ned) - np.linalg.norm(cur_vel_ned) )* 0.1 + self.thrust_op
        # print(f"real_thrust_op: {real_thrust_op:.2f}, throttle_cmd: {throttle_cmd:.2f}")
        
        roll_target = np.degrees(roll_target)
        pitch_target = np.degrees(pitch_target)  # 转换为度数
        yaw_target = np.degrees(yaw_target)  # 转换为度数
        
        
        #-------------------------- Log Control data ------------------------------
        self.rpy_cmd.append([roll_target, pitch_target, yaw_target])
        self.thrust_cmd.append(throttle_cmd)
        
        self.data = {
            'pos_errors': self.pos_errors,
            'vel_errors': self.vel_errors,
            'controls': self.controls,
            'times': self.times,
            'acc_body': self.acc_body,
            'rpy_cmd': self.rpy_cmd,
            'throttle_cmd': self.thrust_cmd
        }
        
        # 输出控制指令(单位角度、归一化油门开度)
        return roll_target, pitch_target, yaw_target, throttle_cmd
    
    def get_log_data(self):
        return self.data
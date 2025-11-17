import time
from jsb_gym.TAU.aircraft2 import F16
# from jsb_gym.TAU.config import f16_dog_BVRGym
from jsb_gym.TAU.config import f16_traj_track
import numpy as np
from UtilsLib.fixwing_visualizer_3D import *
from UtilsLib.position_control import *
from UtilsLib.Tools import *

from UtilsLib.manual_trajectory import *

def run_simulation_loop():
    # 载入配置
    conf = f16_traj_track

    T_total, dt, looptimes= 0, conf.dt , 10
    A = 1000
    # realtime=True 开启实时可视化（3个窗口：局部姿态、全局航迹、位置误差）
    # record_debug=True 记录数据用于最后生成控制误差图
    viz = FixWingVisualizer(A=A, dt=dt, realtime=True, record_debug=True)
    
    # 初始化 F16 模型
    f16 = F16(conf=conf, FlightGear=False)
    init_lat, init_lon, init_alt = 32.0, 45.0, 7500
    init_vel = 220
    init_heading = 0
    # 重置飞机初始状态
    f16.reset(lat=init_lat, long=init_lon, alt=init_alt, vel=init_vel, heading=init_heading)

    nedTF = NEDTransformer()
    nedTF.reset(ref_lat=init_lat, ref_lon=init_lon, ref_alt=init_alt)

    # pos_ctrl = PositionController(use_l1 = False, L1_dist=100.0, dt = dt)
    pos_ctrl = PositionController(conf = conf)
    first_time_switch = True
    
    case = 'poly'  # 'Straight' or 'Curve'
    f16.fdm['propulsion/refuel'] = 1.0
    # last_loop_time = time.time()
    for i in range(60000):
        # now = time.time()
        # loop_dt = now - last_loop_time
        # last_loop_time = now
        # print(f"[{i}] Loop interval: {loop_dt:.6f} s")
        
        
        # 模拟获取当前状态
        cur_vel_ned = np.array([f16.get_v_north(),f16.get_v_east(), f16.get_v_down()])
        current_gps = (f16.get_lat_gc_deg(), f16.get_long_gc_deg(), f16.get_altitude())
        current_rpy = (f16.get_phi(in_deg=False), f16.get_theta(in_deg=False), f16.get_psi(in_deg=False))  # (roll, pitch, yaw) in radians

        # print("Current GPS:", current_gps)
        # print("Current RPY (rad):", current_rpy)
        print("Current Velocity:", f16.get_v_north(), f16.get_v_east(), f16.get_v_down())

        cur_pos_ned, cur_rpy_ned = nedTF.update(current_gps, current_rpy)  # 这里可以传入目标轨迹
        # print("Current Position NED:", cur_pos_ned) 
        print("Current RPY NED:", cur_rpy_ned)
        # if(f16.fdm['propulsion/total-fuel-lbs'] < 100):
            
        if i < 50:
            f16.step_BVR([1, 0, 0.3], 0)
            idx = i
            print(i)
            cur_vel = np.linalg.norm(cur_vel_ned)
            # print("cur_vel", cur_vel)
            exp_pos_ned=cur_pos_ned.copy()
        else:
            # f16.step_BVR([1 - (i - idx) * 0.01, 0, 0], 0)
            
            # cur_vel = np.linalg.norm(cur_vel_ned)
            # print("cur_vel", cur_vel)
            if first_time_switch:
                entry_pos = cur_pos_ned.copy()
                entry_vel = cur_vel_ned.copy()
                airspeed = init_vel  # m/s
                if(case == 'Curve'):
                    # t = np.linspace(0, 10*np.pi, 1000)
                    t = np.linspace(-np.pi/2, 2*np.pi - np.pi/2, 1000)

                    relative_waypoints = np.array([
                        10000 * np.cos(t),
                        10000 * np.sin(t),
                        0 * np.ones_like(t)
                    ]).T  # shape: (1000, 3)
                    

                    waypoints = relative_waypoints - relative_waypoints[0] + entry_pos  # shape: (1000, 3)
                    
                elif(case == 'Straight'):
                    
  
                    traj_duration = 12000  # 预期轨迹持续时间（秒）
                    total_dist = airspeed * traj_duration  # 比如 60 秒的轨迹：325*60 = 19500 m

                    num_points = 1000
                    entry_vel_dir = entry_vel / (np.linalg.norm(entry_vel) + 1e-6)
                    entry_vel_dir[2] = 0  # 保持水平飞行
                    distances = np.linspace(0, total_dist, num_points)
                    waypoints = np.array([entry_pos + d * entry_vel_dir for d in distances])
                elif(case == 'poly'):
                    entry_vel_dir = entry_vel / (np.linalg.norm(entry_vel) + 1e-6)
                    # trajectory = EightShapeTrajectory(
                    trajectory = ExpediteClimbAndTurnDescentTrajectory(
                        init_p=entry_pos,
                        init_vn = entry_vel[0],
                        init_ve = entry_vel[1],
                        leg_length_ref=10000,
                        h_ref = 400,
                        v_ref = 220
                        # init_vn=230.0, init_ve=230.0,
                    )
                    
                idx = i
                # traj = SmoothTrajectory(waypoints, airspeed= airspeed)
                traj = trajectory   
                first_time_switch = False
                f16.fdm.set_dt(dt/looptimes)
                exp_pos_ned_last = np.array([0, 0, 0])
            else:
                t = (i- idx - 1) * dt
                
                # exp_pos_ned, exp_vel_ned, exp_acc, exp_rpy_ned = traj.get_ref(t)  # 获取期望位置、速度、加速度和姿态
                exp_pos_ned, exp_vel_ned, exp_acc  = traj.get_minco_pva(t)# 获取期望位置
                exp_rpy_ned = np.array([0, 0, 0])  # 假设期望姿态为水平飞行
                exp_vel_dir = exp_vel_ned / (np.linalg.norm(exp_vel_ned) + 1e-6)  # 单位方向
                cur_vel_dir = cur_vel_ned / (np.linalg.norm(cur_vel_ned) + 1e-6)  # 单位方向
                # print("cur_vel_dir", cur_vel_dir, " exp_vel_dir", exp_vel_dir)
                # print("cur_vel_ned ",cur_vel_ned, " exp_vel_ned ", exp_vel_ned)
                
                # 这里去跟踪 exp_pos_ned, exp_rpy_ned
                roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd = pos_ctrl.update(cur_pos_ned, cur_vel_ned, cur_rpy_ned, 
                                                                                exp_pos_ned, exp_vel_ned, exp_rpy_ned,
                                                                                exp_acc )
                
                print("roll_cmd ", roll_cmd, " pitch_cmd ", pitch_cmd, " yaw_cmd ", yaw_cmd, " throttle_cmd ", throttle_cmd)
                # print(entry_pos)
                f16.set_roll_PID (roll_cmd)
                f16.set_pitch_PID(pitch_cmd)
                f16.set_yaw_PID(psi_ref = yaw_cmd)  # yaw_target - cur_yaw
                
                # print("yaw_error", yaw_err)
                f16.set_throttle(throttle_cmd)
                for i in range(looptimes):
                    f16.fdm.run()
                
        # 设置状态，realtime=True时会自动调用step()实时显示
        # others参数可以传入额外数据（如油量）
        viz.set_state(cur_pos_ned, cur_rpy_ned, exp_pos_ned, 
                      others={'fuel': f16.fdm['propulsion/total-fuel-lbs']})
    
        # print("airspeed: ",f16.get_true_airspeed())
        # pos_ctrl.show_err()
    
    # 仿真结束后，保存控制器日志数据
    viz.set_ctrl_data(pos_ctrl.get_log_data())
    
    # 显示最终的调试图（航迹、位置误差、速度误差、控制指令等）
    print("\n仿真完成！正在生成最终调试图...")
    viz.show_final()
    
def main():
    try:
        run_simulation_loop()  # 你主要的仿真/控制逻辑
        
    except KeyboardInterrupt:
        print("Exit requested, shutting down gracefully...")
        plt.close('all')
if __name__ == "__main__":
    main()
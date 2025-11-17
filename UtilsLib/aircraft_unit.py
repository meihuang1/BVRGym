# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from jsb_gym.TAU.aircraft2 import F16
from UtilsLib.fixwing_visualizer_3D import FixWingVisualizer
from UtilsLib.position_control import PositionController
from UtilsLib.manual_trajectory import ExpediteClimbAndTurnDescentTrajectory
from UtilsLib.Tools import NEDTransformer

@dataclass
class AircraftUnit:
    """把“一架飞机 + 控制 + 轨迹 + 可视化”封成一个对象"""
    f16: Optional[F16] = None
    pos_ctrl: Optional[PositionController] = None
    viz: Optional[FixWingVisualizer] = None

    traj: Optional[ExpediteClimbAndTurnDescentTrajectory] = None

    ned : Optional[NEDTransformer] = None
    
    first_switch: bool = True          # 首次生成轨迹的开关
    
    conf: Optional[dict] = None
    
    # ---------- 初始化函数 ----------
    
    def set_cfg(self, conf: dict) -> None:
        """设置配置"""
        self.conf = conf
        self.ned = NEDTransformer()
        self.ned.reset(ref_lat=conf.init_state['local_origin']['lat'],
                       ref_lon=conf.init_state['local_origin']['lon'],
                       ref_alt=conf.init_state['local_origin']['alt'])
    
    def init_f16(self, *, conf, lat, lon, alt, vel, heading) -> None:
        self.f16 = F16(conf=conf, FlightGear=False)
        self.f16.reset(lat=lat, long=lon, alt=alt, vel=vel, heading=heading)
        # 无限燃油
        self.f16.fdm['propulsion/refuel'] = 1.0
        
    def init_pos_ctrl(self, *, conf) -> None:
        self.pos_ctrl = PositionController(conf=conf)

    def init_viz(self, *, A: float = 1000, dt , realtime, record_debug) -> None:
        self.viz = FixWingVisualizer(A=A, dt = dt, realtime=realtime, record_debug=record_debug)

    def build_traj(self, entry_pos: np.ndarray, entry_vel: np.ndarray, 
                   leg_lenth_ref: float, h_ref: float,
                   v_ref: float = 220.0) -> None:
        """生成一次性轨迹，然后把 first_switch 关掉"""
        self.traj = ExpediteClimbAndTurnDescentTrajectory(
            init_p=entry_pos,
            init_vn=entry_vel[0],
            init_ve=entry_vel[1],
            leg_length_ref=leg_lenth_ref,
            h_ref=h_ref,
            v_ref=v_ref,
        )
        self.first_switch = False
        
    def get_gps(self) -> np.ndarray:
        """获取当前 GPS 坐标"""
        return np.array([self.f16.get_lat_gc_deg(),
                         self.f16.get_long_gc_deg(),
                         self.f16.get_altitude()])  
        
    def get_pos_ned(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            pos_ned : np.ndarray
                - Position in NED frame (north, east, down), shape (3,), unit: meters.
                
            rpy_ned : np.ndarray
                - Orientation (roll, pitch, yaw) in radians, shape (3,). unit: radians.
        """
        return self.ned.update(current_gps=self.get_gps(), current_rpy=self.get_rpy()) 
        
    def get_rpy(self) -> np.ndarray:
        """获取当前姿态角 (roll, pitch, yaw)"""
        return np.array([self.f16.get_phi(in_deg=False),
                         self.f16.get_theta(in_deg=False),
                         self.f16.get_psi(in_deg=False)])
    def get_vel_ned(self) -> np.ndarray:
        """获取当前速度 (North, East, Down)"""
        return np.array([self.f16.get_v_north(),
                         self.f16.get_v_east(),
                         self.f16.get_v_down()])
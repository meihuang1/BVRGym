import numpy as np
import pyproj

class NEDTransformer:
    """
    GPS转NED,统一在NED坐标系下工作
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
        """单点转换: GPS -> NED"""
        x, y, z = self.ecef_transformer.transform(lon, lat, alt)
        cur_ecef = np.array([x, y, z])
        delta_ecef = cur_ecef - self.ref_ecef
        pos_ned = self.ned_matrix @ delta_ecef
        
        # 这里 z轴 使用了 ref_alt - alt 作为D 不用转化后的结果
        pos_ned[2] = self.ref_alt - alt
        
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
        核心统一接口: 
        - current_gps: (lat, lon, alt)
        - current_rpy: (roll, pitch, yaw)  【假定已在NED frame下】
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            pos_ned : np.ndarray
                Position in NED frame, shape (3,)
            rpy_ned : np.ndarray
                Orientation in NED frame, shape (3,)
        """

  
        pos_ned = self.gps_to_ned(*current_gps)

        return pos_ned, current_rpy
    
def normalize_angle_rad(angle):
    """将角度归一化到 [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi
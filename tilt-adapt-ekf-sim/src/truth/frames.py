"""
坐标系与重力模型

约定：
- 导航系 (n): NED (North-East-Down)
- 机体系 (b): FRD (Forward-Right-Down)
- 重力方向: 沿导航系 Z 轴正方向（指向地心）
"""

import numpy as np

# 标准重力加速度 (m/s^2)
GRAVITY_STANDARD = 9.80665


def gravity_n(g: float = GRAVITY_STANDARD) -> np.ndarray:
    """
    导航系中的重力向量 (NED 坐标系)
    
    重力沿 Z 轴正方向（指向地心）
    
    Args:
        g: 重力加速度大小 (m/s^2)，默认为标准重力
    
    Returns:
        g_n = [0, 0, g] (3,)
    """
    return np.array([0.0, 0.0, g], dtype=np.float64)


def gravity_b(q_nb: np.ndarray, g: float = GRAVITY_STANDARD) -> np.ndarray:
    """
    机体系中的重力向量
    
    g_b = R_bn @ g_n
    
    Args:
        q_nb: 姿态四元数 [w, x, y, z] (4,)
        g: 重力加速度大小 (m/s^2)
    
    Returns:
        g_b (3,)
    """
    from ..common.math3d import quat_to_R_bn
    
    g_n = gravity_n(g)
    R_bn = quat_to_R_bn(q_nb)
    return R_bn @ g_n


def accel_measurement_static(q_nb: np.ndarray, g: float = GRAVITY_STANDARD) -> np.ndarray:
    """
    静止时加速度计的理想测量值
    
    加速度计测量的是比力 (specific force)，即 -g_b
    
    Args:
        q_nb: 姿态四元数 [w, x, y, z] (4,)
        g: 重力加速度大小 (m/s^2)
    
    Returns:
        a_meas = -g_b (3,)
    """
    return -gravity_b(q_nb, g)

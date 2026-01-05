"""
三维数学库

包含四元数、旋转矩阵、欧拉角等基本运算。

约定：
- 四元数: q = [w, x, y, z]，Hamilton 约定
- 欧拉角: ZYX 顺序 (yaw-pitch-roll)
- q_nb: 从机体系 (b) 到导航系 (n) 的旋转
- 角度单位: 弧度 (rad)
"""

import numpy as np
from typing import Tuple


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    欧拉角转四元数 (ZYX 顺序)
    
    Args:
        roll: 横滚角 (rad)
        pitch: 俯仰角 (rad)
        yaw: 偏航角 (rad)
    
    Returns:
        q = [w, x, y, z] (4,)
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z], dtype=np.float64)


def quat_to_rpy(q: np.ndarray) -> Tuple[float, float, float]:
    """
    四元数转欧拉角 (ZYX 顺序)
    
    Args:
        q: [w, x, y, z] (4,)
    
    Returns:
        (roll, pitch, yaw) in radians
    """
    w, x, y, z = q
    
    # Roll (φ)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (θ) - 使用 clip 防止 asin 输入超出 [-1, 1]
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    
    # Yaw (ψ)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return (float(roll), float(pitch), float(yaw))


def quat_to_R_nb(q: np.ndarray) -> np.ndarray:
    """
    四元数转旋转矩阵 R_nb (机体系到导航系)
    
    Args:
        q: [w, x, y, z] (4,)
    
    Returns:
        R_nb (3, 3)
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)


def quat_to_R_bn(q: np.ndarray) -> np.ndarray:
    """
    四元数转旋转矩阵 R_bn (导航系到机体系)
    
    Args:
        q: [w, x, y, z] (4,)
    
    Returns:
        R_bn (3, 3) = R_nb^T
    """
    return quat_to_R_nb(q).T


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    四元数归一化
    
    Args:
        q: [w, x, y, z] (4,)
    
    Returns:
        归一化后的四元数 (4,)
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    四元数乘法 (Hamilton 约定)
    
    Args:
        q1: [w, x, y, z] (4,)
        q2: [w, x, y, z] (4,)
    
    Returns:
        q1 ⊗ q2 (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    四元数共轭
    
    Args:
        q: [w, x, y, z] (4,)
    
    Returns:
        q* = [w, -x, -y, -z] (4,)
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    使用四元数旋转向量
    
    v_rotated = q ⊗ [0, v] ⊗ q*
    
    Args:
        q: [w, x, y, z] (4,)
        v: [x, y, z] (3,)
    
    Returns:
        旋转后的向量 (3,)
    """
    # 使用旋转矩阵更高效
    R = quat_to_R_nb(q)
    return R @ v


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    向量的反对称矩阵 (skew-symmetric matrix)
    
    Args:
        v: [x, y, z] (3,)
    
    Returns:
        [v]× (3, 3)
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float64)


def deg2rad(deg: float) -> float:
    """度转弧度"""
    return deg * np.pi / 180.0


def rad2deg(rad: float) -> float:
    """弧度转度"""
    return rad * 180.0 / np.pi


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    从轴角表示创建四元数
    
    Args:
        axis: 旋转轴 (3,)，需要是单位向量
        angle: 旋转角度 (rad)
    
    Returns:
        q = [w, x, y, z] (4,)
    """
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64)


def quat_identity() -> np.ndarray:
    """返回单位四元数"""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

"""
IMU 前向模型

将真值轨迹转换为 IMU 观测数据

Step 11 增强误差模型：
- bias random walk (sigma_rw)
- 温漂 bias(T) (k1, T0)
- scale/misalignment
- 饱和/量化
"""

import numpy as np
from typing import Dict, Any

from ..common.math3d import quat_to_R_bn
from ..truth.frames import gravity_n
from .error_models import (
    add_constant_bias,
    add_white_noise,
    add_bias_random_walk,
    apply_temperature_drift,
    apply_scale_misalignment,
    apply_saturation,
    apply_quantization,
)


def forward_imu(
    truth: Dict[str, Any],
    sensor_params: Dict[str, Any],
    seed: int,
    g: float = 9.80665
) -> Dict[str, Any]:
    """
    IMU 前向模型：将真值转换为 IMU 观测
    
    测量模型（完整误差链）：
    1. 理想测量：gyro_ideal = omega_b, acc_ideal = R_bn @ (a_lin_n + g_n)
    2. 比例因子/安装偏差：x = (I + S + A) @ x_ideal
    3. 偏置：x = x + bias(t)
       - bias(t) = bias0 + random_walk(t) + temp_drift(T)
    4. 白噪声：x = x + noise
    5. 饱和/量化：x = quantize(saturate(x))
    
    Args:
        truth: 真值字典
            - q_nb: (N, 4) 姿态四元数
            - omega_b: (N, 3) 角速度 rad/s
            - a_lin_n: (N, 3) 非重力加速度 m/s^2
            - temp: (N,) 温度 Celsius
        sensor_params: 传感器参数（支持开关控制）
            - acc/gyro:
              - bias0: [3] 初始偏置
              - sigma_white: float 白噪声标准差
              - bias_rw: {enabled, sigma_rw} 偏置随机游走
              - temp_drift: {enabled, k1, T0} 温漂
              - scale_misalign: {enabled, scale_error, misalignment} 比例因子/安装偏差
              - saturation: {enabled, range} 饱和
              - quantization: {enabled, bits, range} 量化
        seed: 随机种子
        g: 重力加速度 m/s^2
    
    Returns:
        meas: 测量字典
            - acc: (N, 3) 加速度计测量 m/s^2
            - gyro: (N, 3) 陀螺仪测量 rad/s
            - acc_bias_true: (N, 3) 真实加速度偏置（含RW+温漂）
            - gyro_bias_true: (N, 3) 真实陀螺偏置（含RW+温漂）
    """
    rng = np.random.default_rng(seed)
    
    q_nb = truth["q_nb"]
    omega_b = truth["omega_b"]
    a_lin_n = truth["a_lin_n"]
    temp = truth.get("temp", None)
    
    n_samples = len(q_nb)
    dt = 1.0 / truth.get("fs", 100.0) if "fs" in truth else 0.01
    
    # 获取传感器参数
    acc_params = sensor_params.get("acc", {})
    gyro_params = sensor_params.get("gyro", {})
    
    # 重力向量（导航系）
    g_n = gravity_n(g)
    
    # ========== 1. 计算理想测量 ==========
    acc_ideal = np.zeros((n_samples, 3), dtype=np.float64)
    for i in range(n_samples):
        R_bn = quat_to_R_bn(q_nb[i])
        a_total_n = a_lin_n[i] + g_n
        acc_ideal[i] = R_bn @ a_total_n
    
    gyro_ideal = omega_b.copy()
    
    # ========== 2. 应用比例因子/安装偏差 ==========
    acc_out = _apply_scale_misalign(acc_ideal, acc_params, rng)
    gyro_out = _apply_scale_misalign(gyro_ideal, gyro_params, rng)
    
    # ========== 3. 计算并应用偏置 ==========
    acc_bias_true = _compute_bias(n_samples, dt, temp, acc_params, rng)
    gyro_bias_true = _compute_bias(n_samples, dt, temp, gyro_params, rng)
    
    acc_out = acc_out + acc_bias_true
    gyro_out = gyro_out + gyro_bias_true
    
    # ========== 4. 添加白噪声 ==========
    acc_sigma = acc_params.get("sigma_white", 0.0)
    gyro_sigma = gyro_params.get("sigma_white", 0.0)
    
    if acc_sigma > 0:
        acc_out = add_white_noise(acc_out, acc_sigma, rng)
    if gyro_sigma > 0:
        gyro_out = add_white_noise(gyro_out, gyro_sigma, rng)
    
    # ========== 5. 饱和/量化 ==========
    acc_out = _apply_saturation_quantization(acc_out, acc_params)
    gyro_out = _apply_saturation_quantization(gyro_out, gyro_params)
    
    return {
        "acc": acc_out,
        "gyro": gyro_out,
        "acc_bias_true": acc_bias_true,
        "gyro_bias_true": gyro_bias_true,
    }


def _compute_bias(
    n_samples: int,
    dt: float,
    temp: np.ndarray,
    params: Dict[str, Any],
    rng: np.random.Generator
) -> np.ndarray:
    """
    计算偏置序列：bias0 + random_walk + temp_drift
    """
    bias0 = np.array(params.get("bias0", [0.0, 0.0, 0.0]), dtype=np.float64)
    
    # 初始化为常值偏置
    bias = np.tile(bias0, (n_samples, 1))
    
    # 偏置随机游走
    bias_rw_cfg = params.get("bias_rw", {})
    if bias_rw_cfg.get("enabled", False):
        sigma_rw = bias_rw_cfg.get("sigma_rw", 0.0)
        if sigma_rw > 0:
            bias_rw = add_bias_random_walk(n_samples, np.zeros(3), sigma_rw, dt, rng)
            bias = bias + bias_rw
    
    # 温漂
    temp_drift_cfg = params.get("temp_drift", {})
    if temp_drift_cfg.get("enabled", False) and temp is not None:
        k1 = temp_drift_cfg.get("k1", 0.0)
        T0 = temp_drift_cfg.get("T0", 25.0)
        if isinstance(k1, list):
            k1 = np.array(k1, dtype=np.float64)
        # 温漂：bias_drift = k1 * (T - T0)
        delta_T = temp - T0
        if np.isscalar(k1):
            bias = bias + k1 * delta_T[:, None]
        else:
            bias = bias + k1[None, :] * delta_T[:, None]
    
    return bias


def _apply_scale_misalign(
    x: np.ndarray,
    params: Dict[str, Any],
    rng: np.random.Generator
) -> np.ndarray:
    """
    应用比例因子和安装偏差
    """
    cfg = params.get("scale_misalign", {})
    if not cfg.get("enabled", False):
        return x.copy()
    
    scale_error = np.array(cfg.get("scale_error", [0.0, 0.0, 0.0]), dtype=np.float64)
    misalignment = np.array(cfg.get("misalignment", [0.0, 0.0, 0.0]), dtype=np.float64)
    
    return apply_scale_misalignment(x, scale_error, misalignment)


def _apply_saturation_quantization(
    x: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    应用饱和和量化
    """
    out = x.copy()
    
    # 饱和
    sat_cfg = params.get("saturation", {})
    if sat_cfg.get("enabled", False):
        range_val = sat_cfg.get("range", np.inf)
        out = apply_saturation(out, range_val)
    
    # 量化
    quant_cfg = params.get("quantization", {})
    if quant_cfg.get("enabled", False):
        bits = quant_cfg.get("bits", 16)
        range_val = quant_cfg.get("range", 16.0)
        out = apply_quantization(out, bits, range_val)
    
    return out

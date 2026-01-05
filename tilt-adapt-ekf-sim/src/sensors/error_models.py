"""
传感器误差模型

包含各种误差注入函数：
- 常值偏置
- 白噪声
- 偏置随机游走（后续扩展）
- 温漂（后续扩展）
- 比例因子/安装偏差（后续扩展）
- 饱和/量化（后续扩展）
"""

import numpy as np
from typing import Optional


def add_constant_bias(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    添加常值偏置
    
    Args:
        x: (N, 3) 输入数据
        b: (3,) 偏置向量
    
    Returns:
        x + b (N, 3)
    """
    return x + b[None, :]


def add_white_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    添加高斯白噪声
    
    Args:
        x: (N, 3) 输入数据
        sigma: 噪声标准差
        rng: 随机数生成器
    
    Returns:
        x + noise (N, 3)
    """
    return x + rng.normal(0.0, sigma, size=x.shape)


def add_bias_random_walk(
    n_samples: int,
    initial_bias: np.ndarray,
    sigma_rw: float,
    dt: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    生成偏置随机游走序列
    
    Args:
        n_samples: 样本数
        initial_bias: (3,) 初始偏置
        sigma_rw: 随机游走噪声密度 (单位/sqrt(s))
        dt: 采样间隔 (s)
        rng: 随机数生成器
    
    Returns:
        bias: (N, 3) 偏置序列
    """
    bias = np.zeros((n_samples, 3), dtype=np.float64)
    bias[0] = initial_bias
    
    # 随机游走：b[k+1] = b[k] + w[k], w ~ N(0, sigma_rw * sqrt(dt))
    sigma_step = sigma_rw * np.sqrt(dt)
    
    for i in range(1, n_samples):
        bias[i] = bias[i-1] + rng.normal(0.0, sigma_step, size=3)
    
    return bias


def apply_temperature_drift(
    bias: np.ndarray,
    temperature: np.ndarray,
    temp_coeff: float,
    ref_temp: float
) -> np.ndarray:
    """
    应用温漂模型
    
    bias_out = bias + temp_coeff * (temperature - ref_temp)
    
    Args:
        bias: (N, 3) 偏置序列
        temperature: (N,) 温度序列
        temp_coeff: 温度系数 (单位/°C)
        ref_temp: 参考温度 (°C)
    
    Returns:
        bias_out: (N, 3) 温漂后的偏置
    """
    delta_temp = temperature - ref_temp
    return bias + temp_coeff * delta_temp[:, None]


def apply_scale_misalignment(
    x: np.ndarray,
    scale_error: np.ndarray,
    misalignment: np.ndarray
) -> np.ndarray:
    """
    应用比例因子和安装偏差
    
    x_out = (I + diag(scale_error) + skew(misalignment)) @ x
    
    Args:
        x: (N, 3) 输入数据
        scale_error: (3,) 比例因子误差 (ppm -> 需要除以 1e6)
        misalignment: (3,) 安装偏差角度 (rad)
    
    Returns:
        x_out: (N, 3)
    """
    # 构建变换矩阵
    # M = I + diag(scale) + skew(misalign)
    I = np.eye(3)
    S = np.diag(scale_error)
    
    # 反对称矩阵
    mx, my, mz = misalignment
    A = np.array([
        [0, -mz, my],
        [mz, 0, -mx],
        [-my, mx, 0]
    ])
    
    M = I + S + A
    
    return (M @ x.T).T


def apply_saturation(x: np.ndarray, range_val: float) -> np.ndarray:
    """
    应用饱和限幅
    
    Args:
        x: (N, 3) 输入数据
        range_val: 量程（正负对称）
    
    Returns:
        x_clipped: (N, 3)
    """
    return np.clip(x, -range_val, range_val)


def apply_quantization(x: np.ndarray, resolution_bits: int, range_val: float) -> np.ndarray:
    """
    应用量化
    
    Args:
        x: (N, 3) 输入数据
        resolution_bits: 分辨率位数
        range_val: 量程
    
    Returns:
        x_quantized: (N, 3)
    """
    # 量化步长
    n_levels = 2 ** resolution_bits
    step = 2 * range_val / n_levels
    
    # 量化
    return np.round(x / step) * step

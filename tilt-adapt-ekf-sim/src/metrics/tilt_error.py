"""
姿态误差指标计算

计算 roll/pitch 估计误差的各种统计指标

注意：指标计算支持 burn-in（预热期），前 burn_in_s 秒的数据不计入统计。
这是因为滤波器需要一定时间收敛，论文中通常也会说明这一点。
"""

import numpy as np
from typing import Dict, Any, Optional


def wrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    """
    将角度 wrap 到 [-180, 180) 范围
    
    Args:
        angle_deg: 角度（度）
    
    Returns:
        wrapped 角度（度）
    """
    return ((angle_deg + 180) % 360) - 180


def compute_tilt_metrics(
    truth: Dict[str, Any],
    est: Dict[str, Any],
    skip_samples: int = 0,
    burn_in_s: float = 0.0,
    fs: float = None,
    wrap_error: bool = True
) -> Dict[str, float]:
    """
    计算姿态误差指标
    
    Args:
        truth: 真值字典
            - rpy_deg: (N, 3) 真值欧拉角 [roll, pitch, yaw] (deg)
            或
            - roll: (N,) 真值 roll (rad)
            - pitch: (N,) 真值 pitch (rad)
        est: 估计字典
            - roll: (N,) 估计 roll (rad)
            - pitch: (N,) 估计 pitch (rad)
        skip_samples: 跳过前 N 个样本（用于跳过收敛段）
        burn_in_s: burn-in 时间（秒），前 burn_in_s 秒不计入统计
                   如果同时指定 skip_samples 和 burn_in_s，取较大值
        fs: 采样率（Hz），用于计算 burn_in_s 对应的样本数
        wrap_error: 是否对误差进行 wrap（避免角度跳变）
    
    Returns:
        metrics: 指标字典
            - rmse_roll: roll RMSE (deg)
            - rmse_pitch: pitch RMSE (deg)
            - peak_roll: roll 峰值误差 (deg)
            - peak_pitch: pitch 峰值误差 (deg)
            - mean_roll: roll 均值误差 (deg)
            - mean_pitch: pitch 均值误差 (deg)
            - std_roll: roll 误差标准差 (deg)
            - std_pitch: pitch 误差标准差 (deg)
            - burn_in_samples: 实际跳过的样本数
    
    Note:
        burn-in 是滤波器评估中的常见做法，因为滤波器需要时间收敛。
        论文中通常会说明 burn-in 时间，例如 "前 0.5s 不计入统计"。
    """
    # 计算实际跳过的样本数
    actual_skip = skip_samples
    if burn_in_s > 0 and fs is not None:
        burn_in_samples = int(burn_in_s * fs)
        actual_skip = max(skip_samples, burn_in_samples)
    
    # 获取真值（支持两种格式）
    if "rpy_deg" in truth:
        roll_true_deg = truth["rpy_deg"][actual_skip:, 0]
        pitch_true_deg = truth["rpy_deg"][actual_skip:, 1]
    else:
        roll_true_deg = np.rad2deg(truth["roll"][actual_skip:])
        pitch_true_deg = np.rad2deg(truth["pitch"][actual_skip:])
    
    # 获取估计值（转换为度）
    roll_est_deg = np.rad2deg(est["roll"][actual_skip:])
    pitch_est_deg = np.rad2deg(est["pitch"][actual_skip:])
    
    # 计算误差
    roll_err = roll_est_deg - roll_true_deg
    pitch_err = pitch_est_deg - pitch_true_deg
    
    # 可选：wrap 误差到 [-180, 180)
    if wrap_error:
        roll_err = wrap_deg(roll_err)
        pitch_err = wrap_deg(pitch_err)
    
    # 计算指标
    # 分离稳态偏置（mean）和随机误差（std）
    # RMSE = sqrt(mean^2 + std^2)，可以分解为系统性误差和随机误差
    mean_roll = float(np.mean(roll_err))
    mean_pitch = float(np.mean(pitch_err))
    std_roll = float(np.std(roll_err))
    std_pitch = float(np.std(pitch_err))
    
    metrics = {
        # RMSE（综合误差）
        "rmse_roll": float(np.sqrt(np.mean(roll_err**2))),
        "rmse_pitch": float(np.sqrt(np.mean(pitch_err**2))),
        
        # 峰值误差
        "peak_roll": float(np.max(np.abs(roll_err))),
        "peak_pitch": float(np.max(np.abs(pitch_err))),
        
        # 稳态偏置（系统性误差）
        "bias_roll": mean_roll,
        "bias_pitch": mean_pitch,
        
        # 随机波动（噪声）
        "noise_roll": std_roll,
        "noise_pitch": std_pitch,
        
        # 兼容旧接口
        "mean_roll": mean_roll,
        "mean_pitch": mean_pitch,
        "std_roll": std_roll,
        "std_pitch": std_pitch,
        
        # burn-in 信息
        "burn_in_samples": actual_skip,
        "burn_in_s": burn_in_s if burn_in_s > 0 else (actual_skip / fs if fs else 0),
    }
    
    return metrics


def compute_convergence_time(
    error: np.ndarray,
    threshold: float,
    dt: float,
    window: int = 10
) -> Optional[float]:
    """
    计算收敛时间
    
    收敛定义：误差首次进入阈值范围并保持 window 个样本
    
    Args:
        error: (N,) 误差序列
        threshold: 收敛阈值
        dt: 采样间隔 (s)
        window: 保持窗口大小
    
    Returns:
        收敛时间 (s)，如果未收敛返回 None
    """
    n = len(error)
    
    for i in range(n - window):
        # 检查从 i 开始的 window 个样本是否都在阈值内
        if np.all(np.abs(error[i:i+window]) < threshold):
            return i * dt
    
    return None


def print_tilt_metrics(metrics: Dict[str, float], name: str = "", fs: float = None) -> None:
    """
    打印姿态误差指标
    
    Args:
        metrics: 指标字典
        name: 滤波器名称
        fs: 采样率（用于显示 burn-in 时间）
    """
    print(f"\n{'='*50}")
    if name:
        print(f"姿态误差指标 - {name}")
    else:
        print("姿态误差指标")
    print(f"{'='*50}")
    
    # 显示 burn-in 信息
    if "burn_in_samples" in metrics and metrics["burn_in_samples"] > 0:
        burn_in_samples = metrics["burn_in_samples"]
        burn_in_s = metrics.get("burn_in_s", burn_in_samples / fs if fs else 0)
        print(f"  Burn-in: {burn_in_samples} samples ({burn_in_s:.2f}s)")
    
    # 综合误差
    print(f"\n  [综合误差 RMSE]")
    print(f"    Roll:  {metrics['rmse_roll']:.4f}°")
    print(f"    Pitch: {metrics['rmse_pitch']:.4f}°")
    
    # 稳态偏置（系统性误差）
    print(f"\n  [稳态偏置 Bias] (系统性)")
    bias_roll = metrics.get('bias_roll', metrics.get('mean_roll', 0))
    bias_pitch = metrics.get('bias_pitch', metrics.get('mean_pitch', 0))
    print(f"    Roll:  {bias_roll:+.4f}°")
    print(f"    Pitch: {bias_pitch:+.4f}°")
    
    # 随机波动（噪声）
    print(f"\n  [随机波动 Noise] (1σ)")
    noise_roll = metrics.get('noise_roll', metrics.get('std_roll', 0))
    noise_pitch = metrics.get('noise_pitch', metrics.get('std_pitch', 0))
    print(f"    Roll:  {noise_roll:.4f}°")
    print(f"    Pitch: {noise_pitch:.4f}°")
    
    # 峰值误差
    print(f"\n  [峰值误差 Peak]")
    print(f"    Roll:  {metrics['peak_roll']:.4f}°")
    print(f"    Pitch: {metrics['peak_pitch']:.4f}°")
    
    print(f"{'='*50}")

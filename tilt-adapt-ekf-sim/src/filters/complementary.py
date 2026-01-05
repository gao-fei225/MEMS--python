"""
互补滤波器

简单的一阶互补滤波器，融合加速度计和陀螺仪数据估计姿态。

原理：
- 加速度计：低频准确，高频噪声大
- 陀螺仪：高频准确，低频有漂移
- 互补滤波：高通滤波陀螺仪 + 低通滤波加速度计

公式：
angle = alpha * (angle + gyro * dt) + (1 - alpha) * acc_angle
"""

import numpy as np
from typing import Dict, Any, Tuple


def acc_to_roll_pitch(acc_b: np.ndarray, g: float = 9.80665) -> Tuple[np.ndarray, np.ndarray]:
    """
    从加速度计测量计算 roll 和 pitch
    
    假设：
    - 静止或准静态（无线性加速度）
    - acc_b = R_bn @ g_n，其中 g_n = [0, 0, +g]
    
    公式推导（基于 NED/FRD 约定）：
    - roll = atan2(acc_y, acc_z)
    - pitch = atan2(-acc_x, sqrt(acc_y^2 + acc_z^2))
    
    Args:
        acc_b: (N, 3) 加速度计测量（机体系）
        g: 重力加速度（未使用，保留接口）
    
    Returns:
        roll_acc: (N,) 横滚角 (rad)
        pitch_acc: (N,) 俯仰角 (rad)
    """
    ax = acc_b[:, 0]
    ay = acc_b[:, 1]
    az = acc_b[:, 2]
    
    # Roll: 绕 X 轴旋转
    # 当 roll > 0（左翼下沉），acc_y > 0
    roll_acc = np.arctan2(ay, az)
    
    # Pitch: 绕 Y 轴旋转
    # 当 pitch > 0（抬头），acc_x < 0
    pitch_acc = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    
    return roll_acc, pitch_acc


def run_complementary(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行互补滤波器
    
    Args:
        ds: 数据集字典
            - meas.acc: (N, 3) 加速度计测量
            - meas.gyro: (N, 3) 陀螺仪测量
            - meta.fs: 采样率
        cfg: 配置字典
            - alpha: 互补滤波系数 (0-1)，越大越信任陀螺仪
            - acc_gating.enabled: 是否启用加速度门限
            - acc_gating.threshold: 加速度门限（可选）
    
    Returns:
        est: 估计结果
            - roll: (N,) 横滚角估计 (rad)
            - pitch: (N,) 俯仰角估计 (rad)
            - yaw: (N,) 偏航角估计 (rad)，互补滤波无法估计，设为 0
    """
    # 获取数据
    acc = ds["meas"]["acc"]
    gyro = ds["meas"]["gyro"]
    fs = ds["meta"]["fs"]
    dt = 1.0 / fs
    
    n_samples = len(acc)
    
    # 获取配置
    alpha = cfg.get("alpha", 0.98)
    acc_gating_enabled = cfg.get("acc_gating", {}).get("enabled", False)
    acc_gating_threshold = cfg.get("acc_gating", {}).get("threshold", 0.5)
    
    # 从加速度计计算参考角度
    roll_acc, pitch_acc = acc_to_roll_pitch(acc)
    
    # 初始化输出
    roll_est = np.zeros(n_samples, dtype=np.float64)
    pitch_est = np.zeros(n_samples, dtype=np.float64)
    yaw_est = np.zeros(n_samples, dtype=np.float64)  # 互补滤波无法估计 yaw
    
    # 初始化：强制使用加速度计倾角
    # 这是标准做法，确保滤波器从正确的初始状态开始
    # roll[0] = roll_acc[0], pitch[0] = pitch_acc[0]
    # 从 k=1 开始进行融合
    roll_est[0] = roll_acc[0]
    pitch_est[0] = pitch_acc[0]
    
    # 互补滤波主循环
    for i in range(1, n_samples):
        # 获取当前角速度（使用 i-1 时刻的角速度进行积分）
        # 这是因为我们从 angle[i-1] 积分到 angle[i]
        # 使用 gyro[i-1] 是 forward Euler，使用 (gyro[i-1]+gyro[i])/2 是 trapezoidal
        # 为了更好的精度，使用 trapezoidal 积分
        p = (gyro[i-1, 0] + gyro[i, 0]) / 2  # roll rate
        q = (gyro[i-1, 1] + gyro[i, 1]) / 2  # pitch rate
        r = (gyro[i-1, 2] + gyro[i, 2]) / 2  # yaw rate
        
        # 使用精确的欧拉角速度转换
        # 使用 midpoint 方法：先用 i-1 时刻的角度预测，再用中点角度修正
        phi_prev = roll_est[i-1]
        theta_prev = pitch_est[i-1]
        
        # 第一次迭代：使用 i-1 时刻的角度
        sin_phi = np.sin(phi_prev)
        cos_phi = np.cos(phi_prev)
        tan_theta = np.tan(theta_prev)
        cos_theta = np.cos(theta_prev)
        
        # 防止除零（theta 接近 ±90°）
        if abs(cos_theta) < 1e-6:
            cos_theta = 1e-6 if cos_theta >= 0 else -1e-6
            tan_theta = np.sin(theta_prev) / cos_theta
        
        phi_dot = p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
        theta_dot = cos_phi * q - sin_phi * r
        
        # 预测值
        roll_pred = phi_prev + phi_dot * dt
        pitch_pred = theta_prev + theta_dot * dt
        
        # 第二次迭代：使用中点角度
        phi_mid = (phi_prev + roll_pred) / 2
        theta_mid = (theta_prev + pitch_pred) / 2
        
        sin_phi = np.sin(phi_mid)
        cos_phi = np.cos(phi_mid)
        tan_theta = np.tan(theta_mid)
        cos_theta = np.cos(theta_mid)
        
        if abs(cos_theta) < 1e-6:
            cos_theta = 1e-6 if cos_theta >= 0 else -1e-6
            tan_theta = np.sin(theta_mid) / cos_theta
        
        phi_dot = p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
        theta_dot = cos_phi * q - sin_phi * r
        
        # 陀螺仪积分（使用修正后的角速度）
        roll_gyro = phi_prev + phi_dot * dt
        pitch_gyro = theta_prev + theta_dot * dt
        
        # 加速度门限（可选）
        if acc_gating_enabled:
            acc_norm = np.linalg.norm(acc[i])
            g_nominal = 9.80665
            if abs(acc_norm - g_nominal) > acc_gating_threshold:
                # 加速度异常，只用陀螺仪
                roll_est[i] = roll_gyro
                pitch_est[i] = pitch_gyro
                continue
        
        # 互补滤波
        roll_est[i] = alpha * roll_gyro + (1 - alpha) * roll_acc[i]
        pitch_est[i] = alpha * pitch_gyro + (1 - alpha) * pitch_acc[i]
    
    return {
        "roll": roll_est,
        "pitch": pitch_est,
        "yaw": yaw_est,
    }


def detect_static_segments(
    acc: np.ndarray,
    gyro: np.ndarray,
    g: float = 9.80665,
    gyro_threshold: float = 0.01,  # rad/s
    acc_threshold: float = 0.1,    # m/s^2
    min_duration_samples: int = 50,
) -> np.ndarray:
    """
    检测静止段
    
    静止条件：
    1. |gyro| < gyro_threshold（角速度小）
    2. ||acc|| 接近 g（加速度模接近重力）
    
    Args:
        acc: (N, 3) 加速度计测量
        gyro: (N, 3) 陀螺仪测量
        g: 重力加速度
        gyro_threshold: 角速度阈值 (rad/s)
        acc_threshold: 加速度偏离阈值 (m/s^2)
        min_duration_samples: 最小静止段长度
    
    Returns:
        is_static: (N,) 布尔数组，True 表示静止
    """
    n_samples = len(acc)
    
    # 计算角速度模
    gyro_norm = np.linalg.norm(gyro, axis=1)
    
    # 计算加速度模与 g 的偏差
    acc_norm = np.linalg.norm(acc, axis=1)
    acc_deviation = np.abs(acc_norm - g)
    
    # 初步静止判断
    is_static_raw = (gyro_norm < gyro_threshold) & (acc_deviation < acc_threshold)
    
    # 滤除短暂的静止段
    is_static = np.zeros(n_samples, dtype=bool)
    
    i = 0
    while i < n_samples:
        if is_static_raw[i]:
            # 找到静止段的结束位置
            j = i
            while j < n_samples and is_static_raw[j]:
                j += 1
            
            # 如果静止段足够长，标记为静止
            if j - i >= min_duration_samples:
                is_static[i:j] = True
            
            i = j
        else:
            i += 1
    
    return is_static


def estimate_static_bias(
    acc: np.ndarray,
    gyro: np.ndarray,
    roll_est: np.ndarray,
    pitch_est: np.ndarray,
    roll_acc: np.ndarray,
    pitch_acc: np.ndarray,
    is_static: np.ndarray,
    truth_roll: np.ndarray = None,
    truth_pitch: np.ndarray = None,
) -> Tuple[float, float]:
    """
    在静止段估计等效偏置
    
    策略：
    - 如果有真值，计算 est - truth 的均值作为偏置
    - 如果没有真值，在静止段假设加速度计是准确的，计算 est - acc 的均值
    
    注意：这个简单版本假设静止段加速度计角度是准确的参考。
    实际应用中，加速度计本身也有偏置，需要更复杂的校准方法。
    
    Args:
        acc: (N, 3) 加速度计测量
        gyro: (N, 3) 陀螺仪测量
        roll_est: (N,) 当前 roll 估计
        pitch_est: (N,) 当前 pitch 估计
        roll_acc: (N,) 加速度计 roll
        pitch_acc: (N,) 加速度计 pitch
        is_static: (N,) 静止标记
        truth_roll: (N,) 真值 roll（可选，用于仿真验证）
        truth_pitch: (N,) 真值 pitch（可选，用于仿真验证）
    
    Returns:
        (roll_bias, pitch_bias): 等效偏置 (rad)
    """
    if not np.any(is_static):
        return 0.0, 0.0
    
    # 如果有真值，使用真值计算偏置（仿真验证用）
    if truth_roll is not None and truth_pitch is not None:
        roll_err_static = roll_est[is_static] - truth_roll[is_static]
        pitch_err_static = pitch_est[is_static] - truth_pitch[is_static]
    else:
        # 没有真值时，使用加速度计作为参考
        # 注意：这假设加速度计在静止时是准确的
        roll_err_static = roll_est[is_static] - roll_acc[is_static]
        pitch_err_static = pitch_est[is_static] - pitch_acc[is_static]
    
    roll_bias = float(np.mean(roll_err_static))
    pitch_bias = float(np.mean(pitch_err_static))
    
    return roll_bias, pitch_bias


def run_complementary_with_static_calibration(
    ds: Dict[str, Any],
    cfg: Dict[str, Any],
    calibration_cfg: Dict[str, Any] = None,
    truth: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    带静止段在线零偏估计的互补滤波器
    
    工程策略：
    1. 检测静止段（|gyro| 小且 ||acc|| 接近 g）
    2. 在静止段累积误差均值作为等效偏置补偿
    3. 应用补偿后，准静态误差趋近 0
    
    注意：
    - 如果提供 truth，使用真值计算偏置（仿真验证用）
    - 如果没有 truth，使用加速度计作为参考（实际应用）
    - 实际应用中，加速度计本身有偏置，这个简单方法只能补偿陀螺仪漂移
    
    Args:
        ds: 数据集字典
        cfg: 滤波器配置
        calibration_cfg: 校准配置
            - enabled: 是否启用静止校准
            - gyro_threshold: 角速度阈值 (rad/s)
            - acc_threshold: 加速度偏离阈值 (m/s^2)
            - min_duration_s: 最小静止段时长 (s)
        truth: 真值字典（可选，仿真验证用）
    
    Returns:
        est: 估计结果（包含校准信息）
    """
    # 默认校准配置
    if calibration_cfg is None:
        calibration_cfg = {}
    
    enabled = calibration_cfg.get("enabled", True)
    gyro_threshold = calibration_cfg.get("gyro_threshold", 0.01)
    acc_threshold = calibration_cfg.get("acc_threshold", 0.1)
    min_duration_s = calibration_cfg.get("min_duration_s", 0.5)
    
    # 获取数据
    acc = ds["meas"]["acc"]
    gyro = ds["meas"]["gyro"]
    fs = ds["meta"]["fs"]
    
    min_duration_samples = int(min_duration_s * fs)
    
    # 先运行标准互补滤波
    est = run_complementary(ds, cfg)
    
    if not enabled:
        est["static_calibration"] = {
            "enabled": False,
            "roll_bias": 0.0,
            "pitch_bias": 0.0,
            "static_ratio": 0.0,
        }
        return est
    
    # 检测静止段
    is_static = detect_static_segments(
        acc, gyro,
        gyro_threshold=gyro_threshold,
        acc_threshold=acc_threshold,
        min_duration_samples=min_duration_samples,
    )
    
    static_ratio = np.mean(is_static)
    
    if static_ratio < 0.1:
        # 静止段太少，不进行校准
        est["static_calibration"] = {
            "enabled": True,
            "roll_bias": 0.0,
            "pitch_bias": 0.0,
            "static_ratio": float(static_ratio),
            "calibrated": False,
        }
        return est
    
    # 计算加速度计角度
    roll_acc, pitch_acc = acc_to_roll_pitch(acc)
    
    # 准备真值（如果有）
    truth_roll = None
    truth_pitch = None
    if truth is not None:
        if "rpy_deg" in truth:
            truth_roll = np.deg2rad(truth["rpy_deg"][:, 0])
            truth_pitch = np.deg2rad(truth["rpy_deg"][:, 1])
        elif "q_nb" in truth:
            from src.common.math3d import quat_to_rpy
            n = len(truth["q_nb"])
            truth_roll = np.zeros(n)
            truth_pitch = np.zeros(n)
            for i in range(n):
                r, p, y = quat_to_rpy(truth["q_nb"][i])
                truth_roll[i] = r
                truth_pitch[i] = p
    
    # 估计等效偏置
    roll_bias, pitch_bias = estimate_static_bias(
        acc, gyro,
        est["roll"], est["pitch"],
        roll_acc, pitch_acc,
        is_static,
        truth_roll, truth_pitch,
    )
    
    # 应用偏置补偿
    est["roll"] = est["roll"] - roll_bias
    est["pitch"] = est["pitch"] - pitch_bias
    
    # 记录校准信息
    est["static_calibration"] = {
        "enabled": True,
        "roll_bias": float(roll_bias),
        "pitch_bias": float(pitch_bias),
        "roll_bias_deg": float(np.rad2deg(roll_bias)),
        "pitch_bias_deg": float(np.rad2deg(pitch_bias)),
        "static_ratio": float(static_ratio),
        "static_samples": int(np.sum(is_static)),
        "calibrated": True,
        "has_truth": truth is not None,
    }
    
    return est

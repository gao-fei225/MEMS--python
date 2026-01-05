"""
工况库

支持的工况类型：
- quasi_static: 准静态（固定姿态）
- swing: 摆动（正弦姿态变化）
- accel: 加减速（线性加速度）
- turn: 转弯（恒定角速度）
- vibration: 振动
- shock: 冲击
"""

import numpy as np
from typing import Dict, Any

from ..common.math3d import rpy_to_quat, deg2rad


def generate_quasi_static(
    fs: float,
    duration_s: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    temp_C: float,
    seed: int
) -> Dict[str, Any]:
    """
    生成准静态工况真值
    
    姿态固定，角速度为零，无非重力加速度
    
    Args:
        fs: 采样率 (Hz)
        duration_s: 时长 (s)
        roll_deg: 横滚角 (deg)
        pitch_deg: 俯仰角 (deg)
        yaw_deg: 偏航角 (deg)
        temp_C: 温度 (Celsius)
        seed: 随机种子（此工况不使用，保留接口一致性）
    
    Returns:
        truth dict:
            t: (N,) 时间戳
            dt: float 采样间隔
            fs: float 采样率
            q_nb: (N, 4) 姿态四元数（常值）
            omega_b: (N, 3) 角速度（全零）
            a_lin_n: (N, 3) 非重力加速度（全零）
            temp: (N,) 温度（常值）
    """
    # 计算样本数
    n_samples = int(np.ceil(fs * duration_s)) + 1
    dt = 1.0 / fs
    
    # 时间戳
    t = np.arange(n_samples, dtype=np.float64) / fs
    
    # 姿态四元数（常值）
    roll = deg2rad(roll_deg)
    pitch = deg2rad(pitch_deg)
    yaw = deg2rad(yaw_deg)
    q = rpy_to_quat(roll, pitch, yaw)
    q_nb = np.tile(q, (n_samples, 1))
    
    # 角速度（全零）
    omega_b = np.zeros((n_samples, 3), dtype=np.float64)
    
    # 非重力加速度（全零）
    a_lin_n = np.zeros((n_samples, 3), dtype=np.float64)
    
    # 温度（常值）
    temp = np.full(n_samples, temp_C, dtype=np.float64)
    
    return {
        "t": t,
        "dt": dt,
        "fs": fs,
        "q_nb": q_nb,
        "omega_b": omega_b,
        "a_lin_n": a_lin_n,
        "temp": temp,
    }


def generate_swing(
    fs: float,
    duration_s: float,
    roll_amp_deg: float,
    pitch_amp_deg: float,
    roll_freq_hz: float,
    pitch_freq_hz: float,
    roll_phase_deg: float,
    pitch_phase_deg: float,
    yaw_deg: float,
    temp_C: float,
    seed: int
) -> Dict[str, Any]:
    """
    生成摆动工况真值
    
    姿态按正弦规律变化，角速度为姿态的导数
    
    Args:
        fs: 采样率 (Hz)
        duration_s: 时长 (s)
        roll_amp_deg: 横滚幅度 (deg)
        pitch_amp_deg: 俯仰幅度 (deg)
        roll_freq_hz: 横滚频率 (Hz)
        pitch_freq_hz: 俯仰频率 (Hz)
        roll_phase_deg: 横滚相位 (deg)
        pitch_phase_deg: 俯仰相位 (deg)
        yaw_deg: 偏航角（固定）(deg)
        temp_C: 温度 (Celsius)
        seed: 随机种子（此工况不使用）
    
    Returns:
        truth dict (包含 dt, fs, rpy_deg 用于调试)
    """
    # 计算样本数
    n_samples = int(np.ceil(fs * duration_s)) + 1
    dt = 1.0 / fs
    
    # 时间戳
    t = np.arange(n_samples, dtype=np.float64) / fs
    
    # 转换为弧度
    roll_amp = deg2rad(roll_amp_deg)
    pitch_amp = deg2rad(pitch_amp_deg)
    roll_phase = deg2rad(roll_phase_deg)
    pitch_phase = deg2rad(pitch_phase_deg)
    yaw = deg2rad(yaw_deg)
    
    # 计算欧拉角序列
    roll_seq = roll_amp * np.sin(2 * np.pi * roll_freq_hz * t + roll_phase)
    pitch_seq = pitch_amp * np.sin(2 * np.pi * pitch_freq_hz * t + pitch_phase)
    yaw_seq = np.full(n_samples, yaw)
    
    # 保存欧拉角（度）用于调试
    rpy_deg = np.column_stack([
        roll_seq * 180.0 / np.pi,
        pitch_seq * 180.0 / np.pi,
        yaw_seq * 180.0 / np.pi
    ])
    
    # 计算四元数序列
    q_nb = np.zeros((n_samples, 4), dtype=np.float64)
    for i in range(n_samples):
        q_nb[i] = rpy_to_quat(roll_seq[i], pitch_seq[i], yaw_seq[i])
    
    # 计算欧拉角速度（解析导数）
    phi_dot = roll_amp * 2 * np.pi * roll_freq_hz * np.cos(2 * np.pi * roll_freq_hz * t + roll_phase)
    theta_dot = pitch_amp * 2 * np.pi * pitch_freq_hz * np.cos(2 * np.pi * pitch_freq_hz * t + pitch_phase)
    psi_dot = np.zeros(n_samples, dtype=np.float64)
    
    # 将欧拉角速度转换为机体角速度
    # [p]   [1    0        -sin(θ)    ] [φ_dot]
    # [q] = [0  cos(φ)   sin(φ)cos(θ)] [θ_dot]
    # [r]   [0 -sin(φ)   cos(φ)cos(θ)] [ψ_dot]
    omega_b = np.zeros((n_samples, 3), dtype=np.float64)
    
    for i in range(n_samples):
        phi = roll_seq[i]
        theta = pitch_seq[i]
        
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # 机体角速度
        omega_b[i, 0] = phi_dot[i] - sin_theta * psi_dot[i]  # p
        omega_b[i, 1] = cos_phi * theta_dot[i] + sin_phi * cos_theta * psi_dot[i]  # q
        omega_b[i, 2] = -sin_phi * theta_dot[i] + cos_phi * cos_theta * psi_dot[i]  # r
    
    # 非重力加速度（全零，假设无平移运动）
    a_lin_n = np.zeros((n_samples, 3), dtype=np.float64)
    
    # 温度（常值）
    temp = np.full(n_samples, temp_C, dtype=np.float64)
    
    return {
        "t": t,
        "dt": dt,
        "fs": fs,
        "q_nb": q_nb,
        "omega_b": omega_b,
        "a_lin_n": a_lin_n,
        "temp": temp,
        "rpy_deg": rpy_deg,  # 调试用
    }


def generate_accel(
    fs: float,
    duration_s: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    accel_type: str,  # "step", "ramp", "sine"
    accel_axis: str,  # "x", "y", "z" (导航系)
    accel_peak: float,  # m/s^2
    accel_freq_hz: float = 0.1,  # 仅 sine 模式使用
    accel_start_s: float = 5.0,  # 加速开始时间
    accel_duration_s: float = 10.0,  # 加速持续时间
    temp_C: float = 25.0,
    seed: int = 1
) -> Dict[str, Any]:
    """
    生成加减速工况真值
    
    姿态近水平固定，a_lin_n 做阶跃/斜坡/正弦变化
    用于测试 EKF 在动态加速度下的量测失配
    
    Args:
        fs: 采样率 (Hz)
        duration_s: 总时长 (s)
        roll_deg: 横滚角 (deg)
        pitch_deg: 俯仰角 (deg)
        yaw_deg: 偏航角 (deg)
        accel_type: 加速度类型 "step"/"ramp"/"sine"
        accel_axis: 加速度方向 "x"/"y"/"z" (导航系)
        accel_peak: 峰值加速度 (m/s^2)
        accel_freq_hz: 正弦频率 (Hz)，仅 sine 模式
        accel_start_s: 加速开始时间 (s)
        accel_duration_s: 加速持续时间 (s)
        temp_C: 温度 (Celsius)
        seed: 随机种子
    
    Returns:
        truth dict
    """
    n_samples = int(np.ceil(fs * duration_s)) + 1
    dt = 1.0 / fs
    t = np.arange(n_samples, dtype=np.float64) / fs
    
    # 姿态四元数（常值）
    roll = deg2rad(roll_deg)
    pitch = deg2rad(pitch_deg)
    yaw = deg2rad(yaw_deg)
    q = rpy_to_quat(roll, pitch, yaw)
    q_nb = np.tile(q, (n_samples, 1))
    
    # 角速度（全零）
    omega_b = np.zeros((n_samples, 3), dtype=np.float64)
    
    # 非重力加速度
    a_lin_n = np.zeros((n_samples, 3), dtype=np.float64)
    
    # 确定加速度轴索引
    axis_map = {"x": 0, "y": 1, "z": 2}
    if accel_axis.lower() not in axis_map:
        raise ValueError(f"未知加速度轴: {accel_axis}，支持: x, y, z")
    axis_idx = axis_map[accel_axis.lower()]
    
    # 计算加速度时间窗口
    accel_end_s = accel_start_s + accel_duration_s
    
    for i in range(n_samples):
        ti = t[i]
        
        if ti < accel_start_s or ti > accel_end_s:
            # 静止段
            continue
        
        # 加速段内的相对时间
        t_rel = ti - accel_start_s
        
        if accel_type == "step":
            # 阶跃：立即达到峰值
            a_lin_n[i, axis_idx] = accel_peak
            
        elif accel_type == "ramp":
            # 斜坡：线性增加到峰值，然后线性减少
            half_dur = accel_duration_s / 2
            if t_rel < half_dur:
                # 上升段
                a_lin_n[i, axis_idx] = accel_peak * (t_rel / half_dur)
            else:
                # 下降段
                a_lin_n[i, axis_idx] = accel_peak * (1 - (t_rel - half_dur) / half_dur)
                
        elif accel_type == "sine":
            # 正弦：周期性加速度
            a_lin_n[i, axis_idx] = accel_peak * np.sin(2 * np.pi * accel_freq_hz * t_rel)
            
        else:
            raise ValueError(f"未知加速度类型: {accel_type}，支持: step, ramp, sine")
    
    # 温度（常值）
    temp = np.full(n_samples, temp_C, dtype=np.float64)
    
    # 保存欧拉角（度）用于调试
    rpy_deg = np.column_stack([
        np.full(n_samples, roll_deg),
        np.full(n_samples, pitch_deg),
        np.full(n_samples, yaw_deg)
    ])
    
    return {
        "t": t,
        "dt": dt,
        "fs": fs,
        "q_nb": q_nb,
        "omega_b": omega_b,
        "a_lin_n": a_lin_n,
        "temp": temp,
        "rpy_deg": rpy_deg,
    }


# 工况工厂函数
SCENARIO_GENERATORS = {
    "quasi_static": generate_quasi_static,
    "swing": generate_swing,
    "accel": generate_accel,
}


def generate_turn(
    fs: float,
    duration_s: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_rate_dps: float,  # deg/s
    turn_radius_m: float,  # 转弯半径 (m)
    turn_start_s: float = 5.0,
    turn_duration_s: float = 20.0,
    temp_C: float = 25.0,
    seed: int = 1
) -> Dict[str, Any]:
    """
    生成转弯工况真值
    
    yaw 以恒定角速度变化，同时产生向心加速度
    向心加速度 a_c = v^2/r = (ω*r)^2/r = ω^2*r
    
    Args:
        fs: 采样率 (Hz)
        duration_s: 总时长 (s)
        roll_deg: 横滚角 (deg)
        pitch_deg: 俯仰角 (deg)
        yaw_rate_dps: 偏航角速度 (deg/s)
        turn_radius_m: 转弯半径 (m)
        turn_start_s: 转弯开始时间 (s)
        turn_duration_s: 转弯持续时间 (s)
        temp_C: 温度 (Celsius)
        seed: 随机种子
    
    Returns:
        truth dict
    """
    n_samples = int(np.ceil(fs * duration_s)) + 1
    dt = 1.0 / fs
    t = np.arange(n_samples, dtype=np.float64) / fs
    
    # 转换为弧度
    roll = deg2rad(roll_deg)
    pitch = deg2rad(pitch_deg)
    yaw_rate = deg2rad(yaw_rate_dps)  # rad/s
    
    # 计算向心加速度幅值
    # a_c = ω^2 * r (当 v = ω*r 时)
    a_centripetal = yaw_rate**2 * turn_radius_m
    
    # 初始化数组
    q_nb = np.zeros((n_samples, 4), dtype=np.float64)
    omega_b = np.zeros((n_samples, 3), dtype=np.float64)
    a_lin_n = np.zeros((n_samples, 3), dtype=np.float64)
    yaw_seq = np.zeros(n_samples, dtype=np.float64)
    
    turn_end_s = turn_start_s + turn_duration_s
    yaw_0 = 0.0  # 初始 yaw
    
    for i in range(n_samples):
        ti = t[i]
        
        if ti < turn_start_s:
            # 转弯前：静止
            yaw_i = yaw_0
            omega_yaw = 0.0
            a_c = 0.0
        elif ti <= turn_end_s:
            # 转弯中
            t_rel = ti - turn_start_s
            yaw_i = yaw_0 + yaw_rate * t_rel
            omega_yaw = yaw_rate
            a_c = a_centripetal
        else:
            # 转弯后：静止
            yaw_i = yaw_0 + yaw_rate * turn_duration_s
            omega_yaw = 0.0
            a_c = 0.0
        
        yaw_seq[i] = yaw_i
        
        # 四元数
        q_nb[i] = rpy_to_quat(roll, pitch, yaw_i)
        
        # 机体角速度（yaw 变化时 r = ψ_dot * cos(φ) * cos(θ)）
        # 简化：假设 roll/pitch 很小
        omega_b[i, 0] = 0.0  # p
        omega_b[i, 1] = 0.0  # q
        omega_b[i, 2] = omega_yaw  # r ≈ ψ_dot (小角度)
        
        # 向心加速度（导航系，指向圆心）
        # 假设沿 y 轴（侧向）
        a_lin_n[i, 1] = a_c * np.sign(yaw_rate)  # 正 yaw_rate 时向左
    
    # 温度
    temp = np.full(n_samples, temp_C, dtype=np.float64)
    
    # 保存欧拉角（度）
    rpy_deg = np.column_stack([
        np.full(n_samples, roll_deg),
        np.full(n_samples, pitch_deg),
        np.rad2deg(yaw_seq)
    ])
    
    return {
        "t": t,
        "dt": dt,
        "fs": fs,
        "q_nb": q_nb,
        "omega_b": omega_b,
        "a_lin_n": a_lin_n,
        "temp": temp,
        "rpy_deg": rpy_deg,
    }


def generate_vibration(
    fs: float,
    duration_s: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    vib_rms: float,  # m/s^2 RMS
    vib_bandwidth_hz: float,  # 带宽 (Hz)
    vib_center_hz: float = 0.0,  # 中心频率 (Hz)，0 表示低通
    temp_C: float = 25.0,
    seed: int = 1
) -> Dict[str, Any]:
    """
    生成振动工况真值
    
    姿态固定，a_lin_n 为带限随机噪声
    
    Args:
        fs: 采样率 (Hz)
        duration_s: 总时长 (s)
        roll_deg: 横滚角 (deg)
        pitch_deg: 俯仰角 (deg)
        yaw_deg: 偏航角 (deg)
        vib_rms: 振动 RMS 幅值 (m/s^2)
        vib_bandwidth_hz: 振动带宽 (Hz)
        vib_center_hz: 振动中心频率 (Hz)
        temp_C: 温度 (Celsius)
        seed: 随机种子
    
    Returns:
        truth dict
    """
    np.random.seed(seed)
    
    n_samples = int(np.ceil(fs * duration_s)) + 1
    dt = 1.0 / fs
    t = np.arange(n_samples, dtype=np.float64) / fs
    
    # 姿态四元数（常值）
    roll = deg2rad(roll_deg)
    pitch = deg2rad(pitch_deg)
    yaw = deg2rad(yaw_deg)
    q = rpy_to_quat(roll, pitch, yaw)
    q_nb = np.tile(q, (n_samples, 1))
    
    # 角速度（全零）
    omega_b = np.zeros((n_samples, 3), dtype=np.float64)
    
    # 生成带限随机振动
    # 方法：生成白噪声，然后用 FFT 滤波
    from scipy import signal
    
    # 生成白噪声
    white_noise = np.random.randn(n_samples, 3)
    
    # 设计带通/低通滤波器
    nyq = fs / 2
    if vib_center_hz > 0:
        # 带通滤波器
        low = max((vib_center_hz - vib_bandwidth_hz / 2) / nyq, 0.01)
        high = min((vib_center_hz + vib_bandwidth_hz / 2) / nyq, 0.99)
        b, a = signal.butter(4, [low, high], btype='band')
    else:
        # 低通滤波器
        cutoff = min(vib_bandwidth_hz / nyq, 0.99)
        b, a = signal.butter(4, cutoff, btype='low')
    
    # 滤波
    a_lin_n = np.zeros((n_samples, 3), dtype=np.float64)
    for axis in range(3):
        filtered = signal.filtfilt(b, a, white_noise[:, axis])
        # 归一化到目标 RMS
        current_rms = np.sqrt(np.mean(filtered**2))
        if current_rms > 1e-10:
            a_lin_n[:, axis] = filtered * (vib_rms / current_rms)
    
    # 温度
    temp = np.full(n_samples, temp_C, dtype=np.float64)
    
    # 保存欧拉角（度）
    rpy_deg = np.column_stack([
        np.full(n_samples, roll_deg),
        np.full(n_samples, pitch_deg),
        np.full(n_samples, yaw_deg)
    ])
    
    return {
        "t": t,
        "dt": dt,
        "fs": fs,
        "q_nb": q_nb,
        "omega_b": omega_b,
        "a_lin_n": a_lin_n,
        "temp": temp,
        "rpy_deg": rpy_deg,
    }


def generate_shock(
    fs: float,
    duration_s: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    shock_peak: float,  # m/s^2 峰值
    shock_width_s: float,  # 脉冲宽度 (s)
    shock_times: list,  # 冲击发生时间列表 (s)
    shock_axis: str = "z",  # 冲击方向
    temp_C: float = 25.0,
    seed: int = 1
) -> Dict[str, Any]:
    """
    生成冲击工况真值
    
    姿态固定，a_lin_n 在指定时刻产生脉冲
    
    Args:
        fs: 采样率 (Hz)
        duration_s: 总时长 (s)
        roll_deg: 横滚角 (deg)
        pitch_deg: 俯仰角 (deg)
        yaw_deg: 偏航角 (deg)
        shock_peak: 冲击峰值 (m/s^2)
        shock_width_s: 脉冲宽度 (s)
        shock_times: 冲击发生时间列表 (s)
        shock_axis: 冲击方向 "x"/"y"/"z"
        temp_C: 温度 (Celsius)
        seed: 随机种子
    
    Returns:
        truth dict
    """
    n_samples = int(np.ceil(fs * duration_s)) + 1
    dt = 1.0 / fs
    t = np.arange(n_samples, dtype=np.float64) / fs
    
    # 姿态四元数（常值）
    roll = deg2rad(roll_deg)
    pitch = deg2rad(pitch_deg)
    yaw = deg2rad(yaw_deg)
    q = rpy_to_quat(roll, pitch, yaw)
    q_nb = np.tile(q, (n_samples, 1))
    
    # 角速度（全零）
    omega_b = np.zeros((n_samples, 3), dtype=np.float64)
    
    # 确定冲击轴
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map.get(shock_axis.lower(), 2)
    
    # 生成冲击脉冲
    a_lin_n = np.zeros((n_samples, 3), dtype=np.float64)
    
    half_width = shock_width_s / 2
    
    for shock_t in shock_times:
        # 找到冲击时间窗口内的样本
        mask = (t >= shock_t - half_width) & (t <= shock_t + half_width)
        
        # 半正弦脉冲形状
        for i in np.where(mask)[0]:
            t_rel = t[i] - shock_t
            # 半正弦：peak * cos(π * t_rel / width)
            a_lin_n[i, axis_idx] = shock_peak * np.cos(np.pi * t_rel / shock_width_s)
    
    # 温度
    temp = np.full(n_samples, temp_C, dtype=np.float64)
    
    # 保存欧拉角（度）
    rpy_deg = np.column_stack([
        np.full(n_samples, roll_deg),
        np.full(n_samples, pitch_deg),
        np.full(n_samples, yaw_deg)
    ])
    
    return {
        "t": t,
        "dt": dt,
        "fs": fs,
        "q_nb": q_nb,
        "omega_b": omega_b,
        "a_lin_n": a_lin_n,
        "temp": temp,
        "rpy_deg": rpy_deg,
    }


# 更新工况工厂
SCENARIO_GENERATORS["turn"] = generate_turn
SCENARIO_GENERATORS["vibration"] = generate_vibration
SCENARIO_GENERATORS["shock"] = generate_shock


def generate_roll_wrap_test(
    fs: float,
    duration_s: float,
    roll_start_deg: float,
    roll_end_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    temp_C: float,
    seed: int
) -> Dict[str, Any]:
    """
    生成 roll 角跨越 ±180° 的测试工况
    
    用于测试误差 wrap 正确性
    
    Args:
        fs: 采样率 (Hz)
        duration_s: 时长 (s)
        roll_start_deg: 起始 roll 角 (deg)
        roll_end_deg: 结束 roll 角 (deg)
        pitch_deg: 固定 pitch 角 (deg)
        yaw_deg: 固定 yaw 角 (deg)
        temp_C: 温度 (Celsius)
        seed: 随机种子
    
    Returns:
        truth dict
    """
    n_samples = int(np.ceil(fs * duration_s)) + 1
    dt = 1.0 / fs
    t = np.arange(n_samples, dtype=np.float64) / fs
    
    # roll 线性变化
    roll_seq_deg = np.linspace(roll_start_deg, roll_end_deg, n_samples)
    roll_seq = deg2rad(roll_seq_deg)
    
    pitch = deg2rad(pitch_deg)
    yaw = deg2rad(yaw_deg)
    
    pitch_seq = np.full(n_samples, pitch)
    yaw_seq = np.full(n_samples, yaw)
    
    # 保存欧拉角（度）
    rpy_deg = np.column_stack([roll_seq_deg, np.full(n_samples, pitch_deg), np.full(n_samples, yaw_deg)])
    
    # 计算四元数序列
    q_nb = np.zeros((n_samples, 4), dtype=np.float64)
    for i in range(n_samples):
        q_nb[i] = rpy_to_quat(roll_seq[i], pitch_seq[i], yaw_seq[i])
    
    # 计算 roll 角速度（线性变化的导数是常数）
    roll_rate = deg2rad(roll_end_deg - roll_start_deg) / duration_s
    
    # 将欧拉角速度转换为机体角速度
    omega_b = np.zeros((n_samples, 3), dtype=np.float64)
    for i in range(n_samples):
        phi = roll_seq[i]
        theta = pitch_seq[i]
        
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # phi_dot = roll_rate, theta_dot = 0, psi_dot = 0
        omega_b[i, 0] = roll_rate  # p
        omega_b[i, 1] = 0.0  # q
        omega_b[i, 2] = 0.0  # r
    
    a_lin_n = np.zeros((n_samples, 3), dtype=np.float64)
    temp = np.full(n_samples, temp_C, dtype=np.float64)
    
    return {
        "t": t,
        "dt": dt,
        "fs": fs,
        "q_nb": q_nb,
        "omega_b": omega_b,
        "a_lin_n": a_lin_n,
        "temp": temp,
        "rpy_deg": rpy_deg,
    }


def generate_scenario(scenario_type: str, **kwargs) -> Dict[str, Any]:
    """
    根据工况类型生成真值
    
    Args:
        scenario_type: 工况类型
        **kwargs: 工况参数
    
    Returns:
        truth dict
    """
    if scenario_type not in SCENARIO_GENERATORS:
        raise ValueError(f"未知工况类型: {scenario_type}，支持的类型: {list(SCENARIO_GENERATORS.keys())}")
    
    return SCENARIO_GENERATORS[scenario_type](**kwargs)

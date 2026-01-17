"""
自适应 EKF (Adaptive Extended Kalman Filter) - 工业级版本

核心功能：
1. 双通道检测：方向偏差 + 幅值偏差
2. NIS + M-Estimation 连续权重函数
3. 振动/机动解耦检测
4. ZARU (Zero Angular Rate Update) 零角速度修正
5. LPF 数字低通滤波预处理
"""

import numpy as np
from typing import Dict, Any, Tuple
from collections import deque
from scipy import signal

from ..common.math3d import (
    quat_normalize,
    quat_multiply,
    quat_to_rpy,
    quat_to_R_bn,
    rpy_to_quat,
    skew_symmetric,
)

GRAVITY = 9.80665
CHI2_3_95 = 7.815


# ========== LPF 低通滤波器 ==========
def apply_lpf(data: np.ndarray, fs: float = 100.0, cutoff: float = 20.0, 
              use_filtfilt: bool = False) -> np.ndarray:
    """
    数字低通滤波器 (2阶 Butterworth)
    
    Args:
        data: 输入数据 (N, 3)
        fs: 采样频率 (Hz)
        cutoff: 截止频率 (Hz)
        use_filtfilt: True=零相位(仿真), False=单向(实物)
    
    Returns:
        滤波后的数据
    """
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    # 防止截止频率超过奈奎斯特频率
    normalized_cutoff = min(normalized_cutoff, 0.99)
    
    b, a = signal.butter(2, normalized_cutoff, btype='low', analog=False)
    
    if use_filtfilt:
        # 零相位滤波 (仿真用，无延迟)
        return signal.filtfilt(b, a, data, axis=0)
    else:
        # 单向滤波 (实物用，有延迟)
        return signal.lfilter(b, a, data, axis=0)


def quat_omega_matrix(omega: np.ndarray) -> np.ndarray:
    p, q, r = omega
    return np.array([
        [0, -p, -q, -r],
        [p,  0,  r, -q],
        [q, -r,  0,  p],
        [r,  q, -p,  0]
    ], dtype=np.float64)


def propagate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    Omega = quat_omega_matrix(omega)
    q_new = q + 0.5 * Omega @ q * dt
    return quat_normalize(q_new)


class EKFAdaptive:
    def __init__(self, cfg: Dict[str, Any]):
        self.Q_gyro = cfg.get("Q_gyro", 1e-5)
        self.Q_bias = cfg.get("Q_bias", 1e-8)
        self.R0 = cfg.get("R0", 3.5e-6)
        
        innov_cfg = cfg.get("innovation_stat", {})
        self.window_W = innov_cfg.get("window_W", 50)
        self.nis_high = innov_cfg.get("nis_high", 10.0)
        self.nis_low = innov_cfg.get("nis_low", 2.0)
        self.nis_ewma_alpha = innov_cfg.get("ewma_alpha", 0.1)

        adapt_cfg = cfg.get("adaptation", {})
        self.r_up = adapt_cfg.get("r_up", 1.2)
        self.r_down = adapt_cfg.get("r_down", 0.98)
        self.lambda_max = adapt_cfg.get("lambda_max", 100.0)
        self.lambda_min = adapt_cfg.get("lambda_min", 1.0)
        self.soft_saturation = adapt_cfg.get("soft_saturation", True)
        self.lambda_soft_max = adapt_cfg.get("lambda_soft_max", 20.0)
        
        # Plan A: λ EWMA 平滑和延迟响应
        self.ewma_lambda_alpha = adapt_cfg.get("ewma_lambda_alpha", 0.0)
        self.delay_up_count = adapt_cfg.get("delay_up_count", 0)
        self.delay_down_count = adapt_cfg.get("delay_down_count", 0)
        self.lambda_raw = 1.0
        self._up_counter = 0
        self._down_counter = 0
        
        # Plan D: 基于 sigmoid 的平滑 λ 映射
        self.use_sigmoid_mapping = adapt_cfg.get("use_sigmoid_mapping", False)
        self.sigmoid_center = adapt_cfg.get("sigmoid_center", 10.0)  # NIS 中心点
        self.sigmoid_scale = adapt_cfg.get("sigmoid_scale", 0.1)  # 斜率控制
        
        # Plan E: 模仿固定 EKF 的 inflate_R 机制
        self.use_inflate_mapping = adapt_cfg.get("use_inflate_mapping", False)
        self.inflate_decay_rate = adapt_cfg.get("inflate_decay_rate", 0.8)  # 下降衰减率
        self.inflate_rise_smooth = adapt_cfg.get("inflate_rise_smooth", 1.0)  # 上升平滑因子 (1.0=直接跟随)
        
        # Plan F: 幅值感知的自适应策略
        self.use_magnitude_aware = adapt_cfg.get("use_magnitude_aware", False)
        
        # Plan G: 动态感知策略（针对转弯/加减速优化）
        self.use_dynamic_aware = adapt_cfg.get("use_dynamic_aware", False)
        self.mag_threshold = adapt_cfg.get("mag_threshold", 0.3)  # 幅值偏差阈值 (m/s²)
        self.mag_lambda_gain = adapt_cfg.get("mag_lambda_gain", 5.0)  # 幅值偏差对 λ 的增益
        self.gyro_threshold = adapt_cfg.get("gyro_threshold", 0.1)  # 角速度阈值 (rad/s)
        self._mag_error_ewma = 0.0  # 幅值偏差的 EWMA
        self._gyro_norm_ewma = 0.0  # 角速度幅值的 EWMA
        self._dynamic_alpha = adapt_cfg.get("dynamic_alpha", 0.1)  # EWMA 平滑因子
        
        # Plan G+: 加速度矢量平滑（解决振动场景的噪声整流效应）
        self._acc_vec_ewma = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)
        self._acc_vec_alpha = adapt_cfg.get("acc_vec_alpha", 0.1)  # 矢量滤波系数
        
        # Plan G++: 振动检测（区分振动和机动）
        self._acc_var_ewma = 0.0  # 加速度方差的 EWMA（用于检测振动）
        self._vib_threshold = adapt_cfg.get("vib_threshold", 0.1)  # 振动检测阈值
        
        # Plan H: 滑动窗口方差-均值检测（终极版）
        self._acc_window_size = adapt_cfg.get("acc_window_size", 20)  # 滑动窗口大小
        self._acc_buffer = deque(maxlen=self._acc_window_size)  # 加速度缓冲区
        self._vib_var_threshold = adapt_cfg.get("vib_var_threshold", 0.1)  # 振动方差阈值
        self._maneuver_mean_threshold = adapt_cfg.get("maneuver_mean_threshold", 0.2)  # 机动均值阈值
        self._lambda_vibration = adapt_cfg.get("lambda_vibration", 100.0)  # 振动时的λ（模仿A4）
        
        dual_cfg = cfg.get("dual_channel", {})
        self.use_dual_channel = dual_cfg.get("enabled", True)
        self.mag_weight = dual_cfg.get("mag_weight", 1.0)
        self.mag_sigma = dual_cfg.get("mag_sigma", 0.5)
        self.combine_mode = dual_cfg.get("combine_mode", "max")
        self.vibration_aware = dual_cfg.get("vibration_aware", False)
        
        self.use_direction_meas = cfg.get("use_direction_meas", True)
        
        init_P_att = cfg.get("init_P_att", (5 * np.pi / 180)**2)
        init_P_bias = cfg.get("init_P_bias", (0.01)**2)
        
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.b_g = np.zeros(3, dtype=np.float64)
        self.P = np.diag([init_P_att]*3 + [init_P_bias]*3)
        
        self.g_n = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)
        self.g_n_unit = self.g_n / np.linalg.norm(self.g_n)
        
        self.lambda_k = 1.0
        self.R_acc = self.R0
        self.nis_ewma = 3.0
        self.nis_window = deque(maxlen=self.window_W)
        
        # ========== ZARU (Zero Angular Rate Update) 配置 ==========
        zaru_cfg = cfg.get("zaru", {})
        self.use_zaru = zaru_cfg.get("enabled", True)
        self.zaru_acc_std_threshold = zaru_cfg.get("acc_std_threshold", 0.01)  # 加速度方差阈值
        self.zaru_gyro_threshold = zaru_cfg.get("gyro_threshold", 0.02)  # 陀螺仪阈值 (rad/s)
        self.zaru_r_scale = zaru_cfg.get("r_scale", 0.01)  # 静止时 R 缩放因子
        self.zaru_q_att_scale = zaru_cfg.get("q_att_scale", 0.001)  # 静止时角度 Q 缩放
        self._is_static = False  # 静止状态标志
        self._static_counter = 0  # 静止计数器
        self._static_confirm_count = zaru_cfg.get("confirm_count", 10)  # 确认静止需要的帧数
        
        # ========== 磁力计融合配置 ==========
        mag_cfg = cfg.get("magnetometer", {})
        self.use_mag = mag_cfg.get("enabled", False)
        self.R_mag = mag_cfg.get("R_mag", 0.1)  # 磁力计测量噪声
        self.mag_ref = None  # 参考磁场方向（在导航坐标系下）
        self.mag_declination = mag_cfg.get("declination", 0.0)  # 磁偏角（弧度）
        
        # 磁场异常检测参数
        self.mag_norm_ref = mag_cfg.get("norm_ref", 50.0)  # 参考磁场强度 (μT)
        self.mag_norm_threshold = mag_cfg.get("norm_threshold", 0.3)  # 模值偏差阈值 (30%)
        self._mag_norm_ewma = 0.0  # 磁场模值的 EWMA
        self._mag_dip_ref = 0.0  # 参考磁倾角
        self._mag_initialized = False
        self._mag_reject_counter = 0  # 连续拒绝计数器
        
        # 持续性磁场异常检测（新增）
        self._mag_persistent_anomaly = False  # 持续性异常标志
        self._mag_anomaly_counter = 0  # 异常帧计数
        self._mag_normal_counter = 0  # 正常帧计数
        self._mag_persistent_threshold = 200  # 持续异常阈值（约 0.7 秒，从 500 降到 200）
        
        # ========== VQF 风格捷联预滤波 ==========
        strap_cfg = cfg.get("strapdown_prefilter", {})
        self.use_strap_prefilter = strap_cfg.get("enabled", False)
        self.strap_cutoff = strap_cfg.get("cutoff", 0.5)  # 低通滤波截止频率 (Hz)
        self.strap_alpha = strap_cfg.get("alpha", 0.01)  # EWMA 系数 (备用)
        
        # 捷联四元数（纯陀螺仪积分，不校正）
        self.q_strap = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # 惯性系下的加速度 EWMA（低通滤波后的重力估计）
        self._acc_inertial_ewma = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)
        # 第二级 EWMA（更慢的滤波器）
        self._acc_inertial_ewma2 = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)
        # 高频加速度能量（用于自适应 R）
        self._acc_hf_energy = 0.0
        # 自适应重力估计（融合两级滤波器）
        self._gravity_estimate = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)

    def predict(self, gyro: np.ndarray, dt: float) -> None:
        omega = gyro - self.b_g
        self.q = propagate_quaternion(self.q, omega, dt)
        
        # VQF 风格：并行积分捷联四元数（不使用 bias 校正）
        if self.use_strap_prefilter:
            self.q_strap = propagate_quaternion(self.q_strap, gyro, dt)
        
        omega_skew = skew_symmetric(omega)
        F = np.eye(6, dtype=np.float64)
        F[0:3, 0:3] = np.eye(3) - omega_skew * dt
        F[0:3, 3:6] = -np.eye(3) * dt
        
        # ZARU: 静止时降低角度 Q，让 Bias 更快收敛
        Q = np.zeros((6, 6), dtype=np.float64)
        if self._is_static and self.use_zaru:
            # 静止时：角度预测几乎没噪声，Bias 快速收敛
            Q[0:3, 0:3] = np.eye(3) * self.Q_gyro * dt * self.zaru_q_att_scale
            # 增强：静止时大幅增大 Bias 的过程噪声，极速收敛（VQF 风格）
            Q[3:6, 3:6] = np.eye(3) * self.Q_bias * dt * 50.0  # 从 10.0 提升到 50.0
        else:
            Q[0:3, 0:3] = np.eye(3) * self.Q_gyro * dt
            Q[3:6, 3:6] = np.eye(3) * self.Q_bias * dt
        
        self.P = F @ self.P @ F.T + Q
    
    def _strapdown_prefilter(self, acc: np.ndarray, gyro: np.ndarray = None) -> np.ndarray:
        """
        VQF 风格捷联惯性系预滤波（增强版）
        
        核心改进：
        1. 将加速度旋转到惯性系（使用捷联四元数）
        2. 在惯性系下强低通滤波（去除线性加速度，保留重力）
        3. 提供纯净的重力参考，用于倾角估计和磁场门控
        
        Returns:
            伪重力观测值（在机体坐标系下）
        """
        # 捷联四元数对应的旋转矩阵
        R_strap = quat_to_R_bn(self.q_strap)
        
        # 将加速度旋转到惯性系
        acc_inertial = R_strap.T @ acc
        
        # 在惯性系下强低通滤波（EWMA，alpha 越小滤波越强）
        # VQF 使用 2-3Hz 截止频率，这里用 alpha=0.005 模拟
        alpha = self.strap_alpha
        self._acc_inertial_ewma = alpha * acc_inertial + (1 - alpha) * self._acc_inertial_ewma
        
        # 归一化得到纯净重力方向（用于磁场门控）
        gravity_norm = np.linalg.norm(self._acc_inertial_ewma)
        if gravity_norm > 1e-6:
            self._gravity_estimate = self._acc_inertial_ewma / gravity_norm * GRAVITY
        else:
            self._gravity_estimate = np.array([0.0, 0.0, GRAVITY])
        
        # 计算高频能量（用于自适应 R）
        acc_hf = acc_inertial - self._acc_inertial_ewma
        self._acc_hf_energy = 0.1 * np.linalg.norm(acc_hf)**2 + 0.9 * self._acc_hf_energy
        
        # 将滤波后的加速度旋转回机体系
        acc_filtered = R_strap @ self._acc_inertial_ewma
        
        return acc_filtered
    
    def get_gravity_reference(self) -> np.ndarray:
        """
        获取惯性系下的纯净重力参考（用于磁场门控）
        
        Returns:
            惯性系下的重力向量
        """
        if hasattr(self, '_gravity_estimate'):
            return self._gravity_estimate
        else:
            return np.array([0.0, 0.0, GRAVITY])
    
    def _compute_magnitude_nis(self, acc: np.ndarray) -> float:
        acc_norm = np.linalg.norm(acc)
        mag_error = acc_norm - GRAVITY
        nis_mag = (mag_error / self.mag_sigma) ** 2
        return float(nis_mag)
    
    def _combine_nis(self, nis_dir: float, nis_mag: float) -> float:
        if self.combine_mode == "max":
            return max(nis_dir, nis_mag * self.mag_weight)
        elif self.combine_mode == "sum":
            return nis_dir + nis_mag * self.mag_weight
        else:
            return max(nis_dir, nis_mag * self.mag_weight)

    def update(self, acc: np.ndarray, gyro: np.ndarray = None) -> Tuple[np.ndarray, float, float, float, float, float]:
        # VQF 风格预滤波
        if self.use_strap_prefilter:
            acc_obs = self._strapdown_prefilter(acc)
        else:
            acc_obs = acc
        
        R_bn = quat_to_R_bn(self.q)
        
        # 使用原始加速度计算幅值 NIS（用于自适应）
        nis_mag = self._compute_magnitude_nis(acc) if self.use_dual_channel else 0.0
        
        if self.use_direction_meas:
            # 使用预滤波后的加速度进行方向观测
            acc_obs_norm = np.linalg.norm(acc_obs)
            if acc_obs_norm < 1e-6:
                return np.zeros(3), 0.0, 0.0, self.lambda_k, 0.0, 0.0
            
            z = acc_obs / acc_obs_norm
            h = R_bn @ self.g_n_unit
            v = z - h
            
            H = np.zeros((3, 6), dtype=np.float64)
            H[0:3, 0:3] = skew_symmetric(h)
        else:
            acc_pred = R_bn @ self.g_n
            v = acc_obs - acc_pred
            
            H = np.zeros((3, 6), dtype=np.float64)
            H[0:3, 0:3] = skew_symmetric(acc_pred)
        
        R0_mat = np.eye(3) * self.R0
        S0 = H @ self.P @ H.T + R0_mat
        S0_inv = np.linalg.inv(S0)
        NIS_dir = float(v.T @ S0_inv @ v)
        
        if self.use_dual_channel:
            NIS_combined = self._combine_nis(NIS_dir, nis_mag)
        else:
            NIS_combined = NIS_dir
        
        self.nis_ewma = self.nis_ewma_alpha * NIS_combined + (1 - self.nis_ewma_alpha) * self.nis_ewma
        self.nis_window.append(NIS_combined)
        
        # 选择自适应策略（使用原始加速度进行检测）
        if self.use_dynamic_aware and gyro is not None:
            self._adapt_lambda_dynamic_aware(NIS_dir, acc, gyro)
        elif self.vibration_aware and self.use_dual_channel:
            self._adapt_lambda_vibration_aware(NIS_dir, nis_mag)
        elif self.use_inflate_mapping:
            self._adapt_lambda_inflate(NIS_combined)
        elif self.use_sigmoid_mapping:
            self._adapt_lambda_sigmoid(NIS_combined)
        else:
            self._adapt_lambda(self.nis_ewma)
        
        R_adaptive = np.eye(3) * self.R_acc
        S = H @ self.P @ H.T + R_adaptive
        S_inv = np.linalg.inv(S)
        NIS_adaptive = float(v.T @ S_inv @ v)
        
        K = self.P @ H.T @ S_inv
        dx = K @ v
        
        dtheta = dx[0:3]
        dtheta_norm = np.linalg.norm(dtheta)
        if dtheta_norm > 1e-10:
            dq = np.array([
                np.cos(dtheta_norm / 2),
                dtheta[0] / dtheta_norm * np.sin(dtheta_norm / 2),
                dtheta[1] / dtheta_norm * np.sin(dtheta_norm / 2),
                dtheta[2] / dtheta_norm * np.sin(dtheta_norm / 2),
            ])
            self.q = quat_multiply(self.q, dq)
            self.q = quat_normalize(self.q)
        
        self.b_g = self.b_g + dx[3:6]
        
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_adaptive @ K.T
        
        return v, NIS_dir, NIS_adaptive, self.lambda_k, float(nis_mag), float(NIS_combined)

    def update_mag(self, mag: np.ndarray, acc: np.ndarray = None) -> None:
        """
        磁力计更新 - VQF 风格磁场门控（增强版）
        
        核心改进：
        1. 双重检测：模长 + 磁倾角（使用 EKF 预测姿态，不依赖原始加速度）
        2. 倾角-航向解耦：磁力计更新只修正航向角，不影响倾角和零偏
        3. 极端保守门控：长时间拒绝机制
        """
        if not self.use_mag:
            return
        
        mag_norm = np.linalg.norm(mag)
        if mag_norm < 1e-6:
            return
        
        # ========== VQF 风格磁场异常检测 ==========
        # 初始化参考磁场强度和磁倾角
        if not self._mag_initialized:
            self._mag_norm_ewma = mag_norm
            self.mag_norm_ref = mag_norm
            
            # 使用 EKF 预测的姿态计算初始磁倾角
            R_bn = quat_to_R_bn(self.q)
            h_init = R_bn.T @ (mag / mag_norm)
            self._mag_dip_ref = np.arctan2(h_init[2], np.sqrt(h_init[0]**2 + h_init[1]**2))
            
            self._mag_initialized = True
            self._mag_reject_counter = 0
        
        # 更新磁场模值的 EWMA
        self._mag_norm_ewma = 0.05 * mag_norm + 0.95 * self._mag_norm_ewma
        
        # ========== 检测加速度可靠性（使用滤波后的重力估计）==========
        acc_reliable = True
        if acc is not None:
            # 关键改进：使用惯性系滤波后的重力估计，而非原始加速度
            if self.use_strap_prefilter and hasattr(self, '_gravity_estimate'):
                gravity_estimate_norm = np.linalg.norm(self._gravity_estimate)
                acc_deviation = abs(gravity_estimate_norm - GRAVITY)
            else:
                acc_norm = np.linalg.norm(acc)
                acc_deviation = abs(acc_norm - GRAVITY)
            
            if acc_deviation > 3.0:
                acc_reliable = False
        
        # ========== VQF 风格双重检测（使用惯性系重力参考）==========
        # 检测 1：模长偏差
        norm_deviation = abs(mag_norm - self.mag_norm_ref) / self.mag_norm_ref
        norm_deviation_smooth = abs(self._mag_norm_ewma - self.mag_norm_ref) / self.mag_norm_ref
        effective_norm_dev = max(norm_deviation, norm_deviation_smooth * 0.8)
        
        # 检测 2：磁倾角偏差（使用 EKF 预测姿态）
        R_bn = quat_to_R_bn(self.q)
        
        # 将磁力计转到惯性系
        mag_inertial = R_bn.T @ (mag / mag_norm)
        
        # 计算磁倾角（与初始化时保持一致）
        measured_dip = np.arctan2(mag_inertial[2], np.sqrt(mag_inertial[0]**2 + mag_inertial[1]**2))
        dip_deviation = abs(measured_dip - self._mag_dip_ref)
        
        # ========== 极端保守的门控决策 + 持续性异常检测 ==========
        # 检测当前帧是否异常（严格阈值，保护极端场景）
        is_severe_anomaly = (effective_norm_dev > 0.20 or dip_deviation > np.deg2rad(20))  # 极端异常
        is_anomaly = (effective_norm_dev > 0.12 or dip_deviation > np.deg2rad(12))  # 一般异常
        
        # 更新持续性异常状态
        if is_severe_anomaly:
            self._mag_anomaly_counter += 2  # 极端异常加速计数
            self._mag_normal_counter = 0
            
            # 如果连续异常超过阈值，标记为持续性异常（不可逆）
            if self._mag_anomaly_counter > self._mag_persistent_threshold:
                self._mag_persistent_anomaly = True
        elif is_anomaly:
            self._mag_anomaly_counter += 1
            self._mag_normal_counter = 0
            
            if self._mag_anomaly_counter > self._mag_persistent_threshold:
                self._mag_persistent_anomaly = True
        else:
            self._mag_normal_counter += 1
            # 只有在未触发持续性异常时才重置计数器
            if not self._mag_persistent_anomaly:
                self._mag_anomaly_counter = max(0, self._mag_anomaly_counter - 1)  # 缓慢衰减
        
        # 如果检测到持续性异常（如场景 33），完全禁用磁力计（不可逆）
        if self._mag_persistent_anomaly:
            return
        
        # 瞬时异常门控
        if is_anomaly:
            self._mag_reject_counter += 1
            return
        
        # 连续拒绝计数器
        if self._mag_reject_counter > 0:
            if self._mag_reject_counter < 30:
                self._mag_reject_counter += 1
                return
            else:
                self._mag_reject_counter = 0
                mag_trust = 0.4
        else:
            # 分级信任度
            if effective_norm_dev > 0.10 or dip_deviation > np.deg2rad(10):
                mag_trust = 0.5
            elif effective_norm_dev > 0.08 or dip_deviation > np.deg2rad(8):
                mag_trust = 0.7
            elif not acc_reliable:
                mag_trust = 0.6
            else:
                mag_trust = 1.0
        
        # ========== 磁力计融合（EKF 更新 + 软解耦）==========
        m = mag / mag_norm
        
        # 重新计算 h（惯性系下的磁场方向）
        h = mag_inertial
        
        # 计算参考磁场方向（只保留水平分量）
        bx = np.sqrt(h[0]**2 + h[1]**2)
        bz = h[2]
        
        # 预期的磁场方向
        b_ref = np.array([bx, 0.0, bz])
        m_pred = R_bn @ b_ref
        
        # 观测残差（3D 向量）
        v_mag = m - m_pred
        
        # 观测矩阵 H（3x6）：磁力计观测对状态的偏导数
        # 状态：[θ_roll, θ_pitch, θ_yaw, b_x, b_y, b_z]
        H_mag = np.zeros((3, 6), dtype=np.float64)
        H_mag[0:3, 0:3] = skew_symmetric(m_pred)  # 对姿态角的偏导数
        # 磁力计不观测陀螺仪零偏，所以 H_mag[:, 3:6] = 0
        
        # 关键改进：软解耦 - 屏蔽 Roll/Pitch 的更新
        # 将 H 矩阵中对应 Roll/Pitch 的列置零，只保留 Yaw
        # 这样 Kalman 增益 K 就不会更新 Roll/Pitch
        H_mag_decoupled = H_mag.copy()
        H_mag_decoupled[:, 0] = 0.0  # 屏蔽 Roll
        H_mag_decoupled[:, 1] = 0.0  # 屏蔽 Pitch
        # 保留 H_mag_decoupled[:, 2] (Yaw)
        # H_mag_decoupled[:, 3:6] 已经是 0（不更新 Bias）
        
        # 测量噪声协方差（根据信任度调整）
        R_mag_adaptive = np.eye(3) * self.R_mag / mag_trust
        
        # 计算 Kalman 增益
        S_mag = H_mag_decoupled @ self.P @ H_mag_decoupled.T + R_mag_adaptive
        
        # 检查 S_mag 是否可逆
        try:
            S_mag_inv = np.linalg.inv(S_mag)
        except np.linalg.LinAlgError:
            # 矩阵奇异，跳过更新
            return
        
        K_mag = self.P @ H_mag_decoupled.T @ S_mag_inv
        
        # 状态更新
        dx_mag = K_mag @ v_mag
        
        # 应用姿态修正（只有 Yaw 会被更新）
        dtheta_mag = dx_mag[0:3]
        dtheta_mag_norm = np.linalg.norm(dtheta_mag)
        
        if dtheta_mag_norm > 1e-10:
            # 限制修正幅度
            if dtheta_mag_norm > 0.1:  # 约 5.7°
                dtheta_mag = dtheta_mag / dtheta_mag_norm * 0.1
                dtheta_mag_norm = 0.1
            
            dq_mag = np.array([
                np.cos(dtheta_mag_norm / 2),
                dtheta_mag[0] / dtheta_mag_norm * np.sin(dtheta_mag_norm / 2),
                dtheta_mag[1] / dtheta_mag_norm * np.sin(dtheta_mag_norm / 2),
                dtheta_mag[2] / dtheta_mag_norm * np.sin(dtheta_mag_norm / 2),
            ])
            self.q = quat_multiply(self.q, dq_mag)
            self.q = quat_normalize(self.q)
        
        # Bias 更新（由于 H_mag_decoupled[:, 3:6] = 0，这里不会更新）
        self.b_g = self.b_g + dx_mag[3:6]
        
        # 协方差更新（Joseph 形式，数值稳定）
        I_KH_mag = np.eye(6) - K_mag @ H_mag_decoupled
        self.P = I_KH_mag @ self.P @ I_KH_mag.T + K_mag @ R_mag_adaptive @ K_mag.T

    def _adapt_lambda(self, nis_smoothed: float) -> None:
        """自适应 λ 更新律 - Plan A 改进版"""
        lambda_target = self.lambda_raw
        
        if nis_smoothed > self.nis_high:
            self._up_counter += 1
            self._down_counter = 0
            
            if self._up_counter >= max(1, self.delay_up_count):
                if self.soft_saturation and lambda_target > self.lambda_soft_max:
                    saturation_factor = 1.0 - np.tanh(
                        (lambda_target - self.lambda_soft_max) / 
                        (self.lambda_max - self.lambda_soft_max)
                    )
                    effective_r_up = 1.0 + (self.r_up - 1.0) * saturation_factor
                else:
                    effective_r_up = self.r_up
                lambda_target = min(lambda_target * effective_r_up, self.lambda_max)
                
        elif nis_smoothed < self.nis_low:
            self._down_counter += 1
            self._up_counter = 0
            
            if self._down_counter >= max(1, self.delay_down_count):
                lambda_target = max(lambda_target * self.r_down, self.lambda_min)
        else:
            self._up_counter = 0
            self._down_counter = 0
            decay_rate = 0.995
            lambda_target = max(lambda_target * decay_rate, self.lambda_min)
        
        self.lambda_raw = lambda_target
        
        if self.ewma_lambda_alpha > 0:
            self.lambda_k = (self.ewma_lambda_alpha * lambda_target + 
                           (1 - self.ewma_lambda_alpha) * self.lambda_k)
        else:
            self.lambda_k = lambda_target
        
        self.lambda_k = np.clip(self.lambda_k, self.lambda_min, self.lambda_max)
        self.R_acc = self.R0 * self.lambda_k
    
    def _adapt_lambda_sigmoid(self, nis: float) -> None:
        """
        Plan D: 基于 sigmoid 的平滑 λ 映射
        
        直接将 NIS 映射到 λ，使用 sigmoid 函数实现平滑过渡：
        λ = λ_min + (λ_max - λ_min) * sigmoid((NIS - center) * scale)
        
        优点：
        1. λ 与 NIS 直接正相关
        2. 平滑过渡，避免阶跃响应
        3. 自然饱和，不会过度膨胀
        """
        # sigmoid 映射：NIS -> [0, 1]
        x = (nis - self.sigmoid_center) * self.sigmoid_scale
        sigmoid_val = 1.0 / (1.0 + np.exp(-x))
        
        # 映射到 [λ_min, λ_max]
        lambda_target = self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid_val
        
        self.lambda_raw = lambda_target
        
        # 应用 EWMA 平滑
        if self.ewma_lambda_alpha > 0:
            self.lambda_k = (self.ewma_lambda_alpha * lambda_target + 
                           (1 - self.ewma_lambda_alpha) * self.lambda_k)
        else:
            self.lambda_k = lambda_target
        
        self.lambda_k = np.clip(self.lambda_k, self.lambda_min, self.lambda_max)
        self.R_acc = self.R0 * self.lambda_k
    
    def _adapt_lambda_inflate(self, nis: float) -> None:
        """
        Plan E: 模仿固定 EKF 的 inflate_R 机制，但更智能
        
        核心思想：λ = max(1, NIS / threshold)
        - 当 NIS <= threshold 时，λ = 1（完全信任观测）
        - 当 NIS > threshold 时，λ = NIS / threshold（与固定 EKF 一致）
        
        改进：
        1. 非对称响应：上升直接跟随，下降可选衰减
        2. 可配置的衰减率和平滑因子
        3. 当 NIS 低于阈值时，快速恢复到 λ=1（避免不必要的衰减）
        
        关键：为了保证一致性，λ 必须足够大使得自适应 NIS ≈ 3
        自适应 NIS = 原始 NIS / λ，所以 λ ≈ 原始 NIS / 3
        """
        threshold = self.nis_high  # 使用 nis_high 作为阈值
        
        if nis <= threshold:
            lambda_target = self.lambda_min
        else:
            # 与固定 EKF 的 inflate_R 一致
            # 为了使自适应 NIS ≈ 3，需要 λ = NIS / 3
            # 但使用 threshold 作为分母可以提供更保守的估计
            lambda_target = nis / threshold
        
        # 限制最大值
        lambda_target = min(lambda_target, self.lambda_max)
        
        self.lambda_raw = lambda_target
        
        # 非对称响应：上升直接跟随，下降可选衰减
        if lambda_target > self.lambda_k:
            # 上升：直接跟随（快速响应）
            self.lambda_k = lambda_target
        else:
            # 下降逻辑改进：
            # 1. 如果 NIS 低于阈值，快速恢复到 lambda_min（避免不必要的衰减）
            # 2. 如果 NIS 高于阈值但在下降，使用衰减
            if nis <= threshold:
                # NIS 正常，快速恢复到 lambda_min
                # 使用更快的衰减率，确保快速恢复
                fast_decay = max(self.inflate_decay_rate, 0.8)
                self.lambda_k = max(lambda_target, self.lambda_k * fast_decay)
            else:
                # NIS 仍然高于阈值，使用配置的衰减率
                if self.inflate_decay_rate < 1.0:
                    self.lambda_k = max(lambda_target, self.lambda_k * self.inflate_decay_rate)
                else:
                    self.lambda_k = lambda_target
        
        self.lambda_k = np.clip(self.lambda_k, self.lambda_min, self.lambda_max)
        self.R_acc = self.R0 * self.lambda_k
    
    def _adapt_lambda_magnitude_aware(self, nis_dir: float, acc: np.ndarray) -> None:
        """
        Plan F: 幅值感知的自适应策略
        
        核心思想：当加速度幅值偏离 g 时，说明存在线性加速度，
        此时加速度方向不再指向真实重力方向，应该降低对观测的信任。
        
        λ = max(λ_dir, λ_mag)
        - λ_dir: 基于方向 NIS 的 inflate 映射
        - λ_mag: 基于幅值偏差的额外膨胀
        """
        threshold = self.nis_high
        
        # 方向通道：与 inflate 一致
        if nis_dir <= threshold:
            lambda_dir = self.lambda_min
        else:
            lambda_dir = nis_dir / threshold
        
        # 幅值通道：检测线性加速度
        acc_norm = np.linalg.norm(acc)
        mag_error = abs(acc_norm - GRAVITY)
        
        # 幅值偏差超过阈值时，增加 λ
        # 使用更敏感的阈值：0.1 m/s² 对应约 0.6° 的方向偏差
        mag_threshold = self.mag_sigma  # 默认 0.5 m/s²
        if mag_error <= mag_threshold:
            lambda_mag = self.lambda_min
        else:
            # 幅值偏差越大，λ 越大
            # 1 m/s² 偏差 → λ ≈ 2, 5 m/s² 偏差 → λ ≈ 10
            lambda_mag = 1.0 + (mag_error / mag_threshold) ** 2
        
        # 取两个通道的最大值
        lambda_target = max(lambda_dir, lambda_mag * self.mag_weight)
        lambda_target = min(lambda_target, self.lambda_max)
        
        self.lambda_raw = lambda_target
        
        # 非对称响应
        if lambda_target > self.lambda_k:
            if self.inflate_rise_smooth < 1.0:
                self.lambda_k = self.inflate_rise_smooth * lambda_target + (1 - self.inflate_rise_smooth) * self.lambda_k
            else:
                self.lambda_k = lambda_target
        else:
            self.lambda_k = max(lambda_target, self.lambda_k * self.inflate_decay_rate)
        
        self.lambda_k = np.clip(self.lambda_k, self.lambda_min, self.lambda_max)
        self.R_acc = self.R0 * self.lambda_k
    
    def _adapt_lambda_vibration_aware(self, nis_dir: float, nis_mag: float) -> None:
        """Plan C: 振动感知的自适应策略"""
        is_pure_vibration = (nis_dir > self.nis_high) and (nis_mag < 5.0)
        is_shock = (nis_mag > 100.0)
        is_stable = (nis_dir < self.nis_low) and (nis_mag < 2.0)
        
        lambda_target = self.lambda_raw
        
        if is_shock:
            lambda_target = min(lambda_target * self.r_up, self.lambda_max)
            self._up_counter = 0
            self._down_counter = 0
        elif is_pure_vibration:
            target_lambda = self.lambda_soft_max
            if lambda_target < target_lambda * 0.9:
                lambda_target = min(lambda_target * 1.15, target_lambda)
            elif lambda_target > target_lambda * 1.1:
                lambda_target = max(lambda_target * 0.85, target_lambda)
        elif is_stable:
            self._down_counter += 1
            if self._down_counter >= max(1, self.delay_down_count):
                lambda_target = max(lambda_target * self.r_down, self.lambda_min)
        else:
            decay_rate = 0.998
            lambda_target = max(lambda_target * decay_rate, self.lambda_min)
        
        self.lambda_raw = lambda_target
        
        if self.ewma_lambda_alpha > 0:
            self.lambda_k = (self.ewma_lambda_alpha * lambda_target + 
                           (1 - self.ewma_lambda_alpha) * self.lambda_k)
        else:
            self.lambda_k = lambda_target
        
        self.lambda_k = np.clip(self.lambda_k, self.lambda_min, self.lambda_max)
        self.R_acc = self.R0 * self.lambda_k

    def _adapt_lambda_dynamic_aware(self, nis_dir: float, acc: np.ndarray, gyro: np.ndarray) -> None:
        """
        Plan V: 激进门控策略
        
        核心改进：
        1. 加速度门控：当 |acc| 偏离 g 超过阈值时，几乎完全不信任加速度计
        2. 使用更激进的指数响应
        3. 振动检测仅在幅值偏差小时触发
        """
        # ========== 1. 更新缓冲区和统计量 ==========
        self._acc_buffer.append(acc.copy())
        
        self._acc_vec_ewma = (self._acc_vec_alpha * acc + 
                             (1 - self._acc_vec_alpha) * self._acc_vec_ewma)
        
        # 计算滑动窗口方差（振动检测）
        acc_std = 0.0
        if len(self._acc_buffer) >= 5:
            acc_array = np.array(self._acc_buffer)
            acc_std = np.max(np.std(acc_array, axis=0))
        
        # 角速度统计
        gyro_norm = np.linalg.norm(gyro)
        self._gyro_norm_ewma = (self._dynamic_alpha * gyro_norm + 
                               (1 - self._dynamic_alpha) * self._gyro_norm_ewma)
        
        # 瞬时幅值偏差
        acc_norm = np.linalg.norm(acc)
        mag_error = abs(acc_norm - GRAVITY)
        
        # 平滑幅值偏差
        acc_norm_smoothed = np.linalg.norm(self._acc_vec_ewma)
        mag_error_smoothed = abs(acc_norm_smoothed - GRAVITY)
        
        # 更新幅值偏差的 EWMA
        self._mag_error_ewma = (self._dynamic_alpha * mag_error + 
                               (1 - self._dynamic_alpha) * self._mag_error_ewma)
        
        # ========== 2. 运动模式分类 ==========
        is_static = (acc_std < self.zaru_acc_std_threshold and 
                    gyro_norm < self.zaru_gyro_threshold)
        
        # 振动检测：高方差但幅值偏差小（真正的振动，不是线性加速度）
        is_vibration = (acc_std > self._vib_var_threshold and 
                       mag_error < 1.5 and 
                       self._mag_error_ewma < 1.0)
        
        # ========== 3. 自适应 λ 计算 ==========
        lambda_factor = self.lambda_min
        
        # A. 静止模式
        if is_static:
            self._static_counter += 1
            if self._static_counter >= self._static_confirm_count:
                self._is_static = True
                if self.use_zaru:
                    lambda_factor = self.zaru_r_scale
        else:
            self._static_counter = 0
            self._is_static = False
        
        if not self._is_static:
            # 使用瞬时和平滑值的最大值
            effective_mag_error = max(mag_error, self._mag_error_ewma * 0.8)
            
            # B. 加速度门控 - 核心策略
            if effective_mag_error > self.mag_threshold:
                excess = effective_mag_error / self.mag_threshold
                
                # 激进门控：偏差越大，λ 越大
                if effective_mag_error > 5.0:
                    # 大偏差：几乎完全不信任加速度计
                    lambda_factor = self.lambda_max * 0.8
                elif effective_mag_error > 2.0:
                    # 中等偏差：大幅降低信任
                    lambda_factor = self.lambda_min + self.mag_lambda_gain * (excess ** 2.0)
                elif effective_mag_error > 1.0:
                    # 小偏差：适度降低信任
                    lambda_factor = self.lambda_min + self.mag_lambda_gain * (excess ** 1.5)
                else:
                    # 很小偏差：线性响应
                    lambda_factor = self.lambda_min + self.mag_lambda_gain * excess
            
            # C. 振动模式（仅当幅值偏差小时）
            elif is_vibration:
                lambda_factor = self._lambda_vibration
            
            # D. 常规模式
            else:
                if nis_dir > self.nis_high:
                    lambda_factor = nis_dir / self.nis_high
                
                # 平滑幅值偏差的补充检测
                if mag_error_smoothed > self.mag_threshold * 0.5:
                    excess_smooth = mag_error_smoothed / self.mag_threshold
                    lambda_smooth = self.lambda_min + self.mag_lambda_gain * 0.5 * excess_smooth
                    lambda_factor = max(lambda_factor, lambda_smooth)
        
        # ========== 4. 限幅和平滑 ==========
        # 高角速度时限制 λ 上限，防止陀螺仪积分误差累积
        effective_lambda_max = self.lambda_max
        if gyro_norm > 0.6:  # > 34.4°/s（平衡的触发阈值）
            # 高角速度时，需要更多地信任加速度计来修正陀螺仪漂移
            gyro_factor = min((gyro_norm / 0.6) ** 1.3, 10.0)  # 平衡的衰减
            effective_lambda_max = max(self.lambda_max / gyro_factor, 20.0)  # 最小上限 20
        
        lambda_target = np.clip(lambda_factor, self.lambda_min, effective_lambda_max)
        self.lambda_raw = lambda_target
        
        # 非对称响应：上升快，下降慢
        if lambda_target > self.lambda_k:
            self.lambda_k = lambda_target
        else:
            self.lambda_k = max(lambda_target, self.lambda_k * self.inflate_decay_rate)
        
        self.lambda_k = np.clip(self.lambda_k, self.lambda_min, effective_lambda_max)
        self.R_acc = self.R0 * self.lambda_k

    def get_window_stats(self) -> Dict[str, float]:
        if len(self.nis_window) == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        nis_arr = np.array(self.nis_window)
        return {"mean": float(np.mean(nis_arr)), "std": float(np.std(nis_arr)), "max": float(np.max(nis_arr))}
    
    def get_attitude(self) -> Tuple[float, float, float]:
        return quat_to_rpy(self.q)
    
    def get_bias(self) -> np.ndarray:
        return self.b_g.copy()
    
    def get_covariance(self) -> np.ndarray:
        return self.P.copy()
    
    def get_lambda(self) -> float:
        return self.lambda_k
    
    def get_R_acc(self) -> float:
        return self.R_acc
    
    def get_nis_ewma(self) -> float:
        return self.nis_ewma


def run_ekf_adaptive(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    acc_raw = ds["meas"]["acc"]
    gyro_raw = ds["meas"]["gyro"]
    fs = ds["meta"]["fs"]
    dt = 1.0 / fs
    
    # ========== LPF 预处理 ==========
    lpf_cfg = cfg.get("lpf", {})
    use_lpf = lpf_cfg.get("enabled", False)
    
    if use_lpf:
        acc_cutoff = lpf_cfg.get("acc_cutoff", 15.0)  # 加速度计截止频率
        gyro_cutoff = lpf_cfg.get("gyro_cutoff", 30.0)  # 陀螺仪截止频率
        use_filtfilt = lpf_cfg.get("use_filtfilt", False)  # False=模拟实物延迟
        
        acc = apply_lpf(acc_raw, fs, acc_cutoff, use_filtfilt)
        gyro = apply_lpf(gyro_raw, fs, gyro_cutoff, use_filtfilt)
    else:
        acc = acc_raw
        gyro = gyro_raw
    
    n_samples = len(acc)
    ekf = EKFAdaptive(cfg)
    
    from ..filters.complementary import acc_to_roll_pitch
    roll_init, pitch_init = acc_to_roll_pitch(acc[0:1])
    ekf.q = rpy_to_quat(roll_init[0], pitch_init[0], 0.0)
    
    roll_est = np.zeros(n_samples, dtype=np.float64)
    pitch_est = np.zeros(n_samples, dtype=np.float64)
    yaw_est = np.zeros(n_samples, dtype=np.float64)
    bias_gyro = np.zeros((n_samples, 3), dtype=np.float64)
    
    innovation = np.zeros((n_samples, 3), dtype=np.float64)
    nis = np.zeros(n_samples, dtype=np.float64)
    nis_raw = np.zeros(n_samples, dtype=np.float64)
    nis_mag = np.zeros(n_samples, dtype=np.float64)
    nis_combined = np.zeros(n_samples, dtype=np.float64)
    lambda_k = np.zeros(n_samples, dtype=np.float64)
    R_acc = np.zeros(n_samples, dtype=np.float64)
    P_diag = np.zeros((n_samples, 6), dtype=np.float64)
    window_mean_nis = np.zeros(n_samples, dtype=np.float64)
    nis_ewma = np.zeros(n_samples, dtype=np.float64)
    lambda_raw = np.zeros(n_samples, dtype=np.float64)
    
    roll, pitch, yaw = ekf.get_attitude()
    roll_est[0] = roll
    pitch_est[0] = pitch
    yaw_est[0] = yaw
    bias_gyro[0] = ekf.get_bias()
    lambda_k[0] = ekf.get_lambda()
    lambda_raw[0] = ekf.lambda_raw
    R_acc[0] = ekf.get_R_acc()
    P_diag[0] = np.diag(ekf.get_covariance())
    nis_ewma[0] = ekf.get_nis_ewma()
    
    for i in range(1, n_samples):
        ekf.predict(gyro[i], dt)
        v, nis_dir_k, nis_k, lam_k, nis_mag_k, nis_comb_k = ekf.update(acc[i], gyro[i])
        
        roll, pitch, yaw = ekf.get_attitude()
        roll_est[i] = roll
        pitch_est[i] = pitch
        yaw_est[i] = yaw
        bias_gyro[i] = ekf.get_bias()
        
        innovation[i] = v
        nis[i] = nis_k
        nis_raw[i] = nis_dir_k
        nis_mag[i] = nis_mag_k
        nis_combined[i] = nis_comb_k
        lambda_k[i] = lam_k
        R_acc[i] = ekf.get_R_acc()
        P_diag[i] = np.diag(ekf.get_covariance())
        nis_ewma[i] = ekf.get_nis_ewma()
        
        stats = ekf.get_window_stats()
        window_mean_nis[i] = stats["mean"]
        lambda_raw[i] = ekf.lambda_raw
    
    return {
        "roll": roll_est,
        "pitch": pitch_est,
        "yaw": yaw_est,
        "bias_gyro": bias_gyro,
        "debug": {
            "innovation": innovation,
            "nis": nis,
            "nis_raw": nis_raw,
            "nis_mag": nis_mag,
            "nis_combined": nis_combined,
            "lambda_k": lambda_k,
            "R_acc": R_acc,
            "P_diag": P_diag,
            "window_mean_nis": window_mean_nis,
            "nis_ewma": nis_ewma,
            "lambda_raw": lambda_raw,
        },
    }

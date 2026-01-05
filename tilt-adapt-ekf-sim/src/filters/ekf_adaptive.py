"""
自适应 EKF (Adaptive Extended Kalman Filter) - 双通道版

Step 12：基于新息统计的自适应更新律

核心改进：
1. 双通道检测：方向偏差 + 幅值偏差
2. 方向通道：检测姿态变化引起的方向偏差
3. 幅值通道：检测非重力加速度（冲击/振动）
4. 综合 NIS = max(NIS_dir, NIS_mag) 或加权组合
5. 使用基准 R0 计算 NIS（避免负反馈循环）
6. Plan A: λ EWMA 平滑和延迟响应
7. Plan C: 振动感知策略
"""

import numpy as np
from typing import Dict, Any, Tuple
from collections import deque

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

    def predict(self, gyro: np.ndarray, dt: float) -> None:
        omega = gyro - self.b_g
        self.q = propagate_quaternion(self.q, omega, dt)
        
        omega_skew = skew_symmetric(omega)
        F = np.eye(6, dtype=np.float64)
        F[0:3, 0:3] = np.eye(3) - omega_skew * dt
        F[0:3, 3:6] = -np.eye(3) * dt
        
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:3, 0:3] = np.eye(3) * self.Q_gyro * dt
        Q[3:6, 3:6] = np.eye(3) * self.Q_bias * dt
        
        self.P = F @ self.P @ F.T + Q
    
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
        R_bn = quat_to_R_bn(self.q)
        
        nis_mag = self._compute_magnitude_nis(acc) if self.use_dual_channel else 0.0
        
        if self.use_direction_meas:
            acc_norm = np.linalg.norm(acc)
            if acc_norm < 1e-6:
                return np.zeros(3), 0.0, 0.0, self.lambda_k, 0.0, 0.0
            
            z = acc / acc_norm
            h = R_bn @ self.g_n_unit
            v = z - h
            
            H = np.zeros((3, 6), dtype=np.float64)
            H[0:3, 0:3] = skew_symmetric(h)
        else:
            acc_pred = R_bn @ self.g_n
            v = acc - acc_pred
            
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
        
        # 选择自适应策略
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
        1. 非对称响应：上升可平滑，下降更快
        2. 可配置的衰减率和平滑因子
        """
        threshold = self.nis_high  # 使用 nis_high 作为阈值
        
        if nis <= threshold:
            lambda_target = self.lambda_min
        else:
            # 与固定 EKF 的 inflate_R 一致
            lambda_target = nis / threshold
        
        # 限制最大值
        lambda_target = min(lambda_target, self.lambda_max)
        
        self.lambda_raw = lambda_target
        
        # 非对称响应：上升可平滑，下降更快
        if lambda_target > self.lambda_k:
            # 上升：可选平滑
            if self.inflate_rise_smooth < 1.0:
                self.lambda_k = self.inflate_rise_smooth * lambda_target + (1 - self.inflate_rise_smooth) * self.lambda_k
            else:
                self.lambda_k = lambda_target
        else:
            # 下降：更快恢复（比固定 EKF 更激进地信任观测）
            self.lambda_k = max(lambda_target, self.lambda_k * self.inflate_decay_rate)
        
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
        Plan G: 动态感知策略（针对转弯/加减速优化）
        
        核心思想：
        1. 检测持续的线性加速度（通过加速度幅值偏离 g）
        2. 检测角速度（转弯时角速度较大）
        3. 当检测到动态运动时，更激进地降低对加速度观测的信任
        
        λ = max(λ_nis, λ_mag, λ_gyro)
        - λ_nis: 基于 NIS 的 inflate 映射
        - λ_mag: 基于加速度幅值偏差
        - λ_gyro: 基于角速度幅值（可选）
        """
        threshold = self.nis_high
        
        # 1. NIS 通道：与 inflate 一致
        if nis_dir <= threshold:
            lambda_nis = self.lambda_min
        else:
            lambda_nis = nis_dir / threshold
        
        # 2. 幅值通道：检测线性加速度
        acc_norm = np.linalg.norm(acc)
        mag_error = abs(acc_norm - GRAVITY)
        
        # 使用 EWMA 平滑幅值偏差，避免瞬时噪声
        self._mag_error_ewma = (self._dynamic_alpha * mag_error + 
                               (1 - self._dynamic_alpha) * self._mag_error_ewma)
        
        # 幅值偏差映射到 λ
        if self._mag_error_ewma <= self.mag_threshold:
            lambda_mag = self.lambda_min
        else:
            # 使用平方关系：偏差越大，λ 增长越快
            # 0.5 m/s² → λ ≈ 3.8, 1.0 m/s² → λ ≈ 12.1
            excess = (self._mag_error_ewma - self.mag_threshold) / self.mag_threshold
            lambda_mag = self.lambda_min + self.mag_lambda_gain * (excess ** 1.5)
        
        # 3. 角速度通道：检测转弯（可选）
        gyro_norm = np.linalg.norm(gyro)
        self._gyro_norm_ewma = (self._dynamic_alpha * gyro_norm + 
                               (1 - self._dynamic_alpha) * self._gyro_norm_ewma)
        
        # 当角速度较大时，说明在转弯，此时加速度包含向心加速度
        if self._gyro_norm_ewma > self.gyro_threshold:
            # 角速度越大，向心加速度越大，应该更不信任加速度观测
            # 向心加速度 = ω² * r，但我们不知道 r，所以用 ω 作为代理
            gyro_factor = 1.0 + (self._gyro_norm_ewma / self.gyro_threshold - 1.0) ** 2
            lambda_mag = lambda_mag * gyro_factor
        
        # 取最大值
        lambda_target = max(lambda_nis, lambda_mag)
        lambda_target = min(lambda_target, self.lambda_max)
        
        self.lambda_raw = lambda_target
        
        # 非对称响应：快速上升，较快下降
        if lambda_target > self.lambda_k:
            if self.inflate_rise_smooth < 1.0:
                self.lambda_k = self.inflate_rise_smooth * lambda_target + (1 - self.inflate_rise_smooth) * self.lambda_k
            else:
                self.lambda_k = lambda_target
        else:
            # 下降时使用更快的衰减
            self.lambda_k = max(lambda_target, self.lambda_k * self.inflate_decay_rate)
        
        self.lambda_k = np.clip(self.lambda_k, self.lambda_min, self.lambda_max)
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
    acc = ds["meas"]["acc"]
    gyro = ds["meas"]["gyro"]
    fs = ds["meta"]["fs"]
    dt = 1.0 / fs
    
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

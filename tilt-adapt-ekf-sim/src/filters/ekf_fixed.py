"""
固定噪声 EKF (Extended Kalman Filter)

Baseline 2：使用固定的过程噪声和测量噪声协方差矩阵。

状态向量 (7维):
- q: 姿态四元数 [w, x, y, z] (4)
- b_g: 陀螺仪偏置 [bx, by, bz] (3)

预测模型:
- 姿态: 四元数积分 (gyro - bias)
- 偏置: 随机游走

测量模型:
- 方向量测：使用单位向量 acc/||acc|| 约束重力方向
- 比原始 acc 量测更鲁棒，不受 ||acc|| ≠ g 影响
- NIS 门控：超过阈值时降低更新权重

输出:
- roll, pitch: 姿态估计
- debug: innovation, NIS 等调试信息
"""

import numpy as np
from typing import Dict, Any, Tuple

from ..common.math3d import (
    quat_normalize,
    quat_multiply,
    quat_to_rpy,
    quat_to_R_nb,
    quat_to_R_bn,
    rpy_to_quat,
    skew_symmetric,
)


# ============================================================
# 常量
# ============================================================

GRAVITY = 9.80665  # m/s^2
CHI2_3_95 = 7.815  # χ²(3) 的 95% 分位数


# ============================================================
# 四元数运动学
# ============================================================

def quat_omega_matrix(omega: np.ndarray) -> np.ndarray:
    """
    构建四元数运动学矩阵 Ω(ω)
    
    q_dot = 0.5 * Ω(ω) @ q
    
    Args:
        omega: 角速度 [p, q, r] (rad/s)
    
    Returns:
        Ω (4, 4)
    """
    p, q, r = omega
    return np.array([
        [0, -p, -q, -r],
        [p,  0,  r, -q],
        [q, -r,  0,  p],
        [r,  q, -p,  0]
    ], dtype=np.float64)


def propagate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    四元数积分（一阶近似）
    
    q_{k+1} = q_k + 0.5 * Ω(ω) @ q_k * dt
    
    Args:
        q: 当前四元数 [w, x, y, z]
        omega: 角速度 [p, q, r] (rad/s)
        dt: 时间步长 (s)
    
    Returns:
        更新后的四元数（归一化）
    """
    Omega = quat_omega_matrix(omega)
    q_new = q + 0.5 * Omega @ q * dt
    return quat_normalize(q_new)


# ============================================================
# EKF 核心
# ============================================================

class EKFFixed:
    """
    固定噪声 EKF
    
    状态: x = [q(4), b_g(3)]^T (7维)
    
    但为了避免四元数约束问题，使用误差状态 EKF (ESKF):
    - 误差状态: δx = [δθ(3), δb_g(3)]^T (6维)
    - δθ 是小角度旋转向量
    
    改进：
    - 方向量测：使用单位向量而非原始加速度
    - NIS 门控：超过阈值时降低更新权重
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        初始化 EKF
        
        Args:
            cfg: 配置字典
                - Q_gyro: 陀螺仪噪声功率谱密度 (rad/s)^2/Hz
                - Q_bias: 偏置随机游走功率谱密度 (rad/s)^2/Hz
                - R_acc: 加速度计测量噪声方差 (m/s^2)^2 或方向噪声方差
                - init_P_att: 初始姿态不确定度 (rad)^2
                - init_P_bias: 初始偏置不确定度 (rad/s)^2
                - use_direction_meas: 是否使用方向量测（默认 True）
                - nis_gating: NIS 门控配置
        """
        # 过程噪声参数
        self.Q_gyro = cfg.get("Q_gyro", 1e-6)  # (rad/s)^2/Hz
        self.Q_bias = cfg.get("Q_bias", 1e-10)  # (rad/s)^2/Hz
        
        # 测量噪声参数
        self.R_acc = cfg.get("R_acc", 0.1**2)  # (m/s^2)^2 或方向噪声
        
        # 方向量测开关
        self.use_direction_meas = cfg.get("use_direction_meas", True)
        
        # NIS 门控配置
        nis_gating = cfg.get("nis_gating", {})
        self.nis_gating_enabled = nis_gating.get("enabled", True)
        self.nis_threshold = nis_gating.get("threshold", CHI2_3_95)
        self.nis_gating_mode = nis_gating.get("mode", "inflate_R")  # "skip" 或 "inflate_R"
        
        # 初始不确定度
        init_P_att = cfg.get("init_P_att", (5 * np.pi / 180)**2)  # 5° 初始不确定度
        init_P_bias = cfg.get("init_P_bias", (0.01)**2)  # 0.01 rad/s 初始偏置不确定度
        
        # 状态初始化
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 单位四元数
        self.b_g = np.zeros(3, dtype=np.float64)  # 陀螺仪偏置
        
        # 误差状态协方差 (6x6): [δθ(3), δb_g(3)]
        self.P = np.diag([
            init_P_att, init_P_att, init_P_att,
            init_P_bias, init_P_bias, init_P_bias
        ])
        
        # 重力向量（导航系）
        self.g_n = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)
        self.g_n_unit = self.g_n / np.linalg.norm(self.g_n)  # 单位重力向量
    
    def predict(self, gyro: np.ndarray, dt: float) -> None:
        """
        预测步骤
        
        Args:
            gyro: 陀螺仪测量 [p, q, r] (rad/s)
            dt: 时间步长 (s)
        """
        # 校正后的角速度
        omega = gyro - self.b_g
        
        # 四元数积分
        self.q = propagate_quaternion(self.q, omega, dt)
        
        # 状态转移矩阵 F (6x6)
        omega_skew = skew_symmetric(omega)
        
        F = np.eye(6, dtype=np.float64)
        F[0:3, 0:3] = np.eye(3) - omega_skew * dt  # 姿态误差传播
        F[0:3, 3:6] = -np.eye(3) * dt  # 偏置对姿态的影响
        
        # 过程噪声协方差 Q
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:3, 0:3] = np.eye(3) * self.Q_gyro * dt  # 陀螺仪噪声
        Q[3:6, 3:6] = np.eye(3) * self.Q_bias * dt  # 偏置随机游走
        
        # 协方差预测
        self.P = F @ self.P @ F.T + Q
    
    def update(self, acc: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        更新步骤（加速度计测量）
        
        使用方向量测：z = acc/||acc||, h(x) = R_bn @ g_n / ||g_n||
        
        Args:
            acc: 加速度计测量 [ax, ay, az] (m/s^2)
        
        Returns:
            (innovation, NIS, gated): 新息向量、归一化新息平方、是否被门控
        """
        R_bn = quat_to_R_bn(self.q)
        
        if self.use_direction_meas:
            # 方向量测：使用单位向量
            acc_norm = np.linalg.norm(acc)
            if acc_norm < 1e-6:
                # 加速度太小，跳过更新
                return np.zeros(3), 0.0, True
            
            z = acc / acc_norm  # 测量的单位向量
            h = R_bn @ self.g_n_unit  # 预测的单位向量
            
            # 新息
            v = z - h
            
            # 测量雅可比 H (3x6)
            # h(x) = R_bn @ g_n_unit
            # ∂h/∂δθ = [R_bn @ g_n_unit]× = [h]×
            H = np.zeros((3, 6), dtype=np.float64)
            H[0:3, 0:3] = skew_symmetric(h)
        else:
            # 原始加速度量测
            acc_pred = R_bn @ self.g_n
            v = acc - acc_pred
            
            H = np.zeros((3, 6), dtype=np.float64)
            H[0:3, 0:3] = skew_symmetric(acc_pred)
        
        # 测量噪声协方差 R
        R = np.eye(3) * self.R_acc
        
        # 新息协方差 S
        S = H @ self.P @ H.T + R
        
        # 计算 NIS (Normalized Innovation Squared)
        S_inv = np.linalg.inv(S)
        NIS = float(v.T @ S_inv @ v)
        
        # NIS 门控
        gated = False
        if self.nis_gating_enabled and NIS > self.nis_threshold:
            gated = True
            if self.nis_gating_mode == "skip":
                # 跳过更新，仅预测
                return v, NIS, gated
            elif self.nis_gating_mode == "inflate_R":
                # 膨胀 R，降低更新权重
                inflate_factor = NIS / self.nis_threshold
                R = R * inflate_factor
                S = H @ self.P @ H.T + R
                S_inv = np.linalg.inv(S)
        
        # 卡尔曼增益 K
        K = self.P @ H.T @ S_inv
        
        # 误差状态更新
        dx = K @ v  # (6,)
        
        # 姿态更新（小角度旋转）
        dtheta = dx[0:3]
        dtheta_norm = np.linalg.norm(dtheta)
        if dtheta_norm > 1e-10:
            # 小角度旋转四元数
            dq = np.array([
                np.cos(dtheta_norm / 2),
                dtheta[0] / dtheta_norm * np.sin(dtheta_norm / 2),
                dtheta[1] / dtheta_norm * np.sin(dtheta_norm / 2),
                dtheta[2] / dtheta_norm * np.sin(dtheta_norm / 2),
            ])
            self.q = quat_multiply(self.q, dq)
            self.q = quat_normalize(self.q)
        
        # 偏置更新
        self.b_g = self.b_g + dx[3:6]
        
        # 协方差更新 (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        return v, NIS, gated
    
    def get_attitude(self) -> Tuple[float, float, float]:
        """
        获取当前姿态估计
        
        Returns:
            (roll, pitch, yaw) in radians
        """
        return quat_to_rpy(self.q)
    
    def get_bias(self) -> np.ndarray:
        """
        获取当前偏置估计
        
        Returns:
            b_g (3,) in rad/s
        """
        return self.b_g.copy()
    
    def get_covariance(self) -> np.ndarray:
        """
        获取当前协方差矩阵
        
        Returns:
            P (6, 6)
        """
        return self.P.copy()


# ============================================================
# 主接口
# ============================================================

def run_ekf_fixed(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行固定噪声 EKF
    
    Args:
        ds: 数据集字典
            - meas.acc: (N, 3) 加速度计测量
            - meas.gyro: (N, 3) 陀螺仪测量
            - meta.fs: 采样率
        cfg: 配置字典
            - Q_gyro: 陀螺仪噪声功率谱密度
            - Q_bias: 偏置随机游走功率谱密度
            - R_acc: 加速度计测量噪声方差
            - init_P_att: 初始姿态不确定度
            - init_P_bias: 初始偏置不确定度
            - use_direction_meas: 是否使用方向量测
            - nis_gating: NIS 门控配置
    
    Returns:
        est: 估计结果
            - roll: (N,) 横滚角估计 (rad)
            - pitch: (N,) 俯仰角估计 (rad)
            - yaw: (N,) 偏航角估计 (rad)
            - bias_gyro: (N, 3) 陀螺仪偏置估计 (rad/s)
            - debug: 调试信息
                - innovation: (N, 3) 新息向量
                - nis: (N,) 归一化新息平方
                - P_diag: (N, 6) 协方差对角线
                - gated: (N,) 是否被门控
    """
    # 获取数据
    acc = ds["meas"]["acc"]
    gyro = ds["meas"]["gyro"]
    fs = ds["meta"]["fs"]
    dt = 1.0 / fs
    
    n_samples = len(acc)
    
    # 初始化 EKF
    ekf = EKFFixed(cfg)
    
    # 使用第一帧加速度计初始化姿态
    from ..filters.complementary import acc_to_roll_pitch
    roll_init, pitch_init = acc_to_roll_pitch(acc[0:1])
    ekf.q = rpy_to_quat(roll_init[0], pitch_init[0], 0.0)
    
    # 输出数组
    roll_est = np.zeros(n_samples, dtype=np.float64)
    pitch_est = np.zeros(n_samples, dtype=np.float64)
    yaw_est = np.zeros(n_samples, dtype=np.float64)
    bias_gyro = np.zeros((n_samples, 3), dtype=np.float64)
    
    # 调试信息
    innovation = np.zeros((n_samples, 3), dtype=np.float64)
    nis = np.zeros(n_samples, dtype=np.float64)
    P_diag = np.zeros((n_samples, 6), dtype=np.float64)
    gated = np.zeros(n_samples, dtype=bool)
    
    # 第一帧
    roll, pitch, yaw = ekf.get_attitude()
    roll_est[0] = roll
    pitch_est[0] = pitch
    yaw_est[0] = yaw
    bias_gyro[0] = ekf.get_bias()
    P_diag[0] = np.diag(ekf.get_covariance())
    
    # 主循环
    for i in range(1, n_samples):
        # 预测
        ekf.predict(gyro[i], dt)
        
        # 更新
        v, nis_k, gated_k = ekf.update(acc[i])
        
        # 保存结果
        roll, pitch, yaw = ekf.get_attitude()
        roll_est[i] = roll
        pitch_est[i] = pitch
        yaw_est[i] = yaw
        bias_gyro[i] = ekf.get_bias()
        
        # 调试信息
        innovation[i] = v
        nis[i] = nis_k
        P_diag[i] = np.diag(ekf.get_covariance())
        gated[i] = gated_k
    
    return {
        "roll": roll_est,
        "pitch": pitch_est,
        "yaw": yaw_est,
        "bias_gyro": bias_gyro,
        "debug": {
            "innovation": innovation,
            "nis": nis,
            "P_diag": P_diag,
            "gated": gated,
        },
    }

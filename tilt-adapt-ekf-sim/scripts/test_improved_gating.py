#!/usr/bin/env python3
"""
测试改进的门控策略

核心改进：
1. 幅值门控：||a||-g > th_mag 时降权
2. 角速率门控：||ω|| > th_gyro 时降权
3. λ 平滑 + 滞回
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.truth.scenarios import generate_vibration, generate_shock, generate_swing, generate_turn, generate_accel
from src.sensors.imu_model import forward_imu
from src.datasets.validate import validate_dataset
from src.filters.ekf_fixed import run_ekf_fixed
from src.common.math3d import quat_to_rpy, quat_normalize, quat_multiply, quat_to_R_bn, rpy_to_quat, skew_symmetric

GRAVITY = 9.80665


def create_dataset(truth, sensor_params, seed=42):
    meas = forward_imu(truth, sensor_params, seed=seed)
    ds = {
        "t": truth["t"],
        "truth": {"q_nb": truth["q_nb"], "omega_b": truth["omega_b"], "a_lin_n": truth["a_lin_n"], "temp": truth["temp"]},
        "meas": {"gyro": meas["gyro"], "acc": meas["acc"]},
        "meta": {"fs": truth["fs"], "seed": seed, "scenario_name": "test", "sensor_params": sensor_params},
    }
    validate_dataset(ds)
    return ds


def get_truth_rpy(truth):
    n = len(truth["q_nb"])
    roll_true = np.zeros(n)
    pitch_true = np.zeros(n)
    for i in range(n):
        r, p, y = quat_to_rpy(truth["q_nb"][i])
        roll_true[i] = r
        pitch_true[i] = p
    return roll_true, pitch_true


def quat_omega_matrix(omega):
    p, q, r = omega
    return np.array([
        [0, -p, -q, -r],
        [p,  0,  r, -q],
        [q, -r,  0,  p],
        [r,  q, -p,  0]
    ], dtype=np.float64)


def propagate_quaternion(q, omega, dt):
    Omega = quat_omega_matrix(omega)
    q_new = q + 0.5 * Omega @ q * dt
    return quat_normalize(q_new)


class EKFImproved:
    """
    改进版 EKF，加入两级门控
    """
    def __init__(self, cfg):
        self.Q_gyro = cfg.get("Q_gyro", 1e-5)
        self.Q_bias = cfg.get("Q_bias", 1e-8)
        self.R0 = cfg.get("R0", 2e-6)
        
        # 门控参数
        self.th_mag = cfg.get("th_mag", 0.3)  # ||a||-g 阈值 (m/s²)
        self.th_gyro = cfg.get("th_gyro", 0.1)  # ||ω|| 阈值 (rad/s, ~6°/s)
        self.lambda_gated = cfg.get("lambda_gated", 1000.0)  # 门控时的 λ
        
        # NIS 自适应参数
        self.nis_high = cfg.get("nis_high", 15.0)
        self.lambda_max = cfg.get("lambda_max", 100.0)
        self.lambda_min = cfg.get("lambda_min", 1.0)
        
        # λ 平滑参数
        self.lambda_smooth_alpha = cfg.get("lambda_smooth_alpha", 0.1)  # EWMA 系数
        
        # 状态初始化
        init_P_att = cfg.get("init_P_att", (5 * np.pi / 180)**2)
        init_P_bias = cfg.get("init_P_bias", (0.01)**2)
        
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.b_g = np.zeros(3, dtype=np.float64)
        self.P = np.diag([init_P_att]*3 + [init_P_bias]*3)
        
        self.g_n = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)
        self.g_n_unit = self.g_n / np.linalg.norm(self.g_n)
        
        self.lambda_k = 1.0
        self.R_acc = self.R0
        
        # EWMA 状态
        self._mag_error_ewma = 0.0
        self._gyro_norm_ewma = 0.0
        self._ewma_alpha = 0.1
        
        # 调试信息
        self.is_gated = False
        self.gate_reason = ""

    def predict(self, gyro, dt):
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

    def update(self, acc, gyro):
        R_bn = quat_to_R_bn(self.q)
        
        # 计算门控指标
        acc_norm = np.linalg.norm(acc)
        mag_error = abs(acc_norm - GRAVITY)
        gyro_norm = np.linalg.norm(gyro)
        
        # EWMA 平滑
        self._mag_error_ewma = self._ewma_alpha * mag_error + (1 - self._ewma_alpha) * self._mag_error_ewma
        self._gyro_norm_ewma = self._ewma_alpha * gyro_norm + (1 - self._ewma_alpha) * self._gyro_norm_ewma
        
        # 方向观测
        z = acc / acc_norm if acc_norm > 1e-6 else np.array([0, 0, 1])
        h = R_bn @ self.g_n_unit
        v = z - h
        
        H = np.zeros((3, 6), dtype=np.float64)
        H[0:3, 0:3] = skew_symmetric(h)
        
        # 计算 NIS (用 R0)
        R0_mat = np.eye(3) * self.R0
        S0 = H @ self.P @ H.T + R0_mat
        S0_inv = np.linalg.inv(S0)
        NIS_dir = float(v.T @ S0_inv @ v)
        
        # 两级门控 - 改进：
        # 1. 对于 turn-like（||a||-g 高 AND ||ω|| 高）：使用极大 λ
        # 2. 对于其他情况：使用 NIS 自适应
        self.is_gated = False
        self.gate_reason = ""
        
        # 只有同时满足 ||a||-g 高 AND ||ω|| 高才门控
        if self._mag_error_ewma > self.th_mag and self._gyro_norm_ewma > self.th_gyro:
            self.is_gated = True
            self.gate_reason = "mag+gyro"
        
        # 计算 λ
        if self.is_gated:
            # 门控时使用极大 λ
            lambda_target = self.lambda_gated
        else:
            # 正常 NIS 自适应
            if NIS_dir <= self.nis_high:
                lambda_target = self.lambda_min
            else:
                lambda_target = NIS_dir / self.nis_high
                lambda_target = min(lambda_target, self.lambda_max)
        
        # λ 平滑
        self.lambda_k = self.lambda_smooth_alpha * lambda_target + (1 - self.lambda_smooth_alpha) * self.lambda_k
        self.lambda_k = np.clip(self.lambda_k, self.lambda_min, self.lambda_gated)
        
        self.R_acc = self.R0 * self.lambda_k
        
        # 更新
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
        
        return v, NIS_dir, NIS_adaptive, self.lambda_k, mag_error, gyro_norm

    def get_attitude(self):
        return quat_to_rpy(self.q)

    def get_bias(self):
        return self.b_g.copy()


def run_ekf_improved(ds, cfg):
    """运行改进版 EKF"""
    acc = ds["meas"]["acc"]
    gyro = ds["meas"]["gyro"]
    fs = ds["meta"]["fs"]
    dt = 1.0 / fs
    
    n_samples = len(acc)
    ekf = EKFImproved(cfg)
    
    # 初始化
    from src.filters.complementary import acc_to_roll_pitch
    roll_init, pitch_init = acc_to_roll_pitch(acc[0:1])
    ekf.q = rpy_to_quat(roll_init[0], pitch_init[0], 0.0)
    
    roll_est = np.zeros(n_samples)
    pitch_est = np.zeros(n_samples)
    bias_gyro = np.zeros((n_samples, 3))
    lambda_k = np.zeros(n_samples)
    nis_raw = np.zeros(n_samples)
    nis_adaptive = np.zeros(n_samples)
    is_gated = np.zeros(n_samples, dtype=bool)
    
    roll, pitch, _ = ekf.get_attitude()
    roll_est[0] = roll
    pitch_est[0] = pitch
    bias_gyro[0] = ekf.get_bias()
    lambda_k[0] = ekf.lambda_k
    
    for i in range(1, n_samples):
        ekf.predict(gyro[i], dt)
        v, nis_dir, nis_adapt, lam, mag_err, gyro_norm = ekf.update(acc[i], gyro[i])
        
        roll, pitch, _ = ekf.get_attitude()
        roll_est[i] = roll
        pitch_est[i] = pitch
        bias_gyro[i] = ekf.get_bias()
        lambda_k[i] = lam
        nis_raw[i] = nis_dir
        nis_adaptive[i] = nis_adapt
        is_gated[i] = ekf.is_gated
    
    return {
        "roll": roll_est,
        "pitch": pitch_est,
        "bias_gyro": bias_gyro,
        "debug": {
            "lambda_k": lambda_k,
            "nis_raw": nis_raw,
            "nis": nis_adaptive,
            "is_gated": is_gated,
        },
    }


def main():
    print("=" * 70)
    print("测试改进的门控策略")
    print("=" * 70)
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    # 生成数据集
    datasets = {}
    
    truth = generate_vibration(fs=100, duration_s=30, roll_deg=0, pitch_deg=0, yaw_deg=0,
                               vib_rms=0.5, vib_bandwidth_hz=10.0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["vibration"] = create_dataset(truth, sensor_params)
    
    truth = generate_shock(fs=100, duration_s=20, roll_deg=0, pitch_deg=0, yaw_deg=0,
                          shock_peak=50.0, shock_width_s=0.05, shock_times=[5.0, 10.0, 15.0], temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["shock"] = create_dataset(truth, sensor_params)
    
    truth = generate_swing(fs=100, duration_s=30, roll_amp_deg=15.0, pitch_amp_deg=10.0,
                          roll_freq_hz=0.5, pitch_freq_hz=0.3, roll_phase_deg=0, pitch_phase_deg=90,
                          yaw_deg=0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["swing"] = create_dataset(truth, sensor_params)
    
    truth = generate_turn(fs=100, duration_s=40, roll_deg=0, pitch_deg=0,
                         yaw_rate_dps=30.0, turn_radius_m=10.0, turn_start_s=5.0, turn_duration_s=30.0,
                         temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["turn"] = create_dataset(truth, sensor_params)
    
    truth = generate_accel(fs=100, duration_s=30, roll_deg=0, pitch_deg=0, yaw_deg=0,
                          accel_type="ramp", accel_axis="x", accel_peak=5.0, accel_start_s=5.0,
                          accel_duration_s=20.0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["accel"] = create_dataset(truth, sensor_params)
    
    # 固定 EKF 配置
    cfg_fixed = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 2e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    
    # 测试不同门控配置
    configs = {
        "no_gating": {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 2e-6,
            "th_mag": 100.0,  # 很高，不触发
            "th_gyro": 100.0,  # 很高，不触发
            "lambda_gated": 1000.0,
            "nis_high": 15.0,
            "lambda_max": 100.0,
            "lambda_smooth_alpha": 0.1,
        },
        "combined_gating": {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 2e-6,
            "th_mag": 0.2,  # 0.2 m/s²
            "th_gyro": np.deg2rad(10),  # 10°/s
            "lambda_gated": 1000.0,
            "nis_high": 15.0,
            "lambda_max": 100.0,
            "lambda_smooth_alpha": 0.1,
        },
        "combined_gating_sensitive": {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 2e-6,
            "th_mag": 0.15,  # 更敏感
            "th_gyro": np.deg2rad(5),  # 更敏感
            "lambda_gated": 1000.0,
            "nis_high": 15.0,
            "lambda_max": 100.0,
            "lambda_smooth_alpha": 0.1,
        },
    }
    
    print(f"\n{'配置':<20} | {'场景':<12} | {'固定':<10} | {'自适应':<10} | {'改善':<10} | {'门控比例':<10}")
    print("-" * 85)
    
    for cfg_name, cfg in configs.items():
        for scenario_name, ds in datasets.items():
            roll_true, pitch_true = get_truth_rpy(ds["truth"])
            
            # 固定 EKF
            est_fixed = run_ekf_fixed(ds, cfg_fixed)
            roll_err = np.rad2deg(est_fixed["roll"] - roll_true)
            pitch_err = np.rad2deg(est_fixed["pitch"] - pitch_true)
            rmse_fixed = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
            
            # 改进 EKF
            est = run_ekf_improved(ds, cfg)
            roll_err = np.rad2deg(est["roll"] - roll_true)
            pitch_err = np.rad2deg(est["pitch"] - pitch_true)
            rmse_adapt = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
            
            improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
            gated_ratio = np.mean(est["debug"]["is_gated"]) * 100
            
            print(f"{cfg_name:<20} | {scenario_name:<12} | {rmse_fixed:<10.3f} | {rmse_adapt:<10.3f} | {improvement:+9.1f}% | {gated_ratio:>8.1f}%")
        print("-" * 85)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

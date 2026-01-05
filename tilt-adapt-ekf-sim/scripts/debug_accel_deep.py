#!/usr/bin/env python3
"""
深入分析 accel 场景：理解为什么两种 EKF 表现相似
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.truth.scenarios import generate_accel
from src.sensors.imu_model import forward_imu
from src.datasets.validate import validate_dataset
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.filters.ekf_fixed import run_ekf_fixed
from src.common.math3d import quat_to_rpy

GRAVITY = 9.80665

def get_truth_rpy(truth):
    n = len(truth["q_nb"])
    roll_true = np.zeros(n)
    pitch_true = np.zeros(n)
    for i in range(n):
        r, p, y = quat_to_rpy(truth["q_nb"][i])
        roll_true[i] = r
        pitch_true[i] = p
    return roll_true, pitch_true

def create_dataset(truth, sensor_params, seed=42):
    meas = forward_imu(truth, sensor_params, seed=seed)
    ds = {
        "t": truth["t"],
        "truth": {
            "q_nb": truth["q_nb"],
            "omega_b": truth["omega_b"],
            "a_lin_n": truth["a_lin_n"],
            "temp": truth["temp"],
        },
        "meas": {"gyro": meas["gyro"], "acc": meas["acc"]},
        "meta": {"fs": truth["fs"], "seed": seed, "scenario_name": "accel", "sensor_params": sensor_params},
    }
    validate_dataset(ds)
    return ds

def main():
    print("="*70)
    print("深入分析 accel 场景")
    print("="*70)
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    truth = generate_accel(
        fs=100, duration_s=30,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        accel_type="ramp", accel_axis="x",
        accel_peak=5.0, accel_start_s=5.0, accel_duration_s=20.0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    ds = create_dataset(truth, sensor_params)
    
    # 分析加速度测量
    acc = ds["meas"]["acc"]
    a_lin_n = truth["a_lin_n"]
    t = ds["t"]
    
    print("\n1. 加速度分析:")
    
    # 在加速段中间取一个点
    mid_idx = int(15.0 * 100)  # t=15s
    acc_mid = acc[mid_idx]
    a_lin_mid = a_lin_n[mid_idx]
    
    print(f"   t=15s 时:")
    print(f"   线性加速度 (真值): [{a_lin_mid[0]:.3f}, {a_lin_mid[1]:.3f}, {a_lin_mid[2]:.3f}] m/s²")
    print(f"   加速度计测量: [{acc_mid[0]:.3f}, {acc_mid[1]:.3f}, {acc_mid[2]:.3f}] m/s²")
    
    # 计算方向偏差
    acc_norm = np.linalg.norm(acc_mid)
    acc_dir = acc_mid / acc_norm
    gravity_dir = np.array([0, 0, 1])  # 真实重力方向
    
    # 方向偏差角度
    cos_angle = np.dot(acc_dir, gravity_dir)
    angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
    
    print(f"   加速度幅值: {acc_norm:.3f} m/s² (理论: {np.sqrt(GRAVITY**2 + a_lin_mid[0]**2):.3f})")
    print(f"   方向偏差: {angle_deg:.2f}°")
    
    # 理论分析
    print("\n2. 理论分析:")
    print(f"   5 m/s² 线性加速度导致的理论方向偏差: {np.rad2deg(np.arctan(5/GRAVITY)):.2f}°")
    print(f"   这个偏差会被 EKF 误认为是姿态变化")
    
    # 测试不同 R0 对 accel 场景的影响
    print("\n3. 测试不同 R0:")
    
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    
    r0_values = [1e-6, 3.5e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    for r0 in r0_values:
        adaptive_cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": r0,
            "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": 7.815, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {"lambda_max": 1000.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
            "dual_channel": {"enabled": False},
        }
        
        fixed_cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R_acc": r0,
            "use_direction_meas": True,
            "nis_gating": {"enabled": False},
        }
        
        est_adapt = run_ekf_adaptive(ds, adaptive_cfg)
        est_fixed = run_ekf_fixed(ds, fixed_cfg)
        
        roll_err_adapt = np.rad2deg(est_adapt["roll"] - roll_true)
        pitch_err_adapt = np.rad2deg(est_adapt["pitch"] - pitch_true)
        rmse_adapt = np.sqrt(np.mean(roll_err_adapt**2 + pitch_err_adapt**2))
        
        roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
        pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
        rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
        
        improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
        
        lambda_mean = np.mean(est_adapt["debug"]["lambda_k"])
        nis_mean = np.mean(est_adapt["debug"]["nis_combined"])
        
        print(f"   R0={r0:.0e}: 自适应={rmse_adapt:.3f}°, 固定={rmse_fixed:.3f}°, 改善={improvement:+.2f}%, λ均值={lambda_mean:.2f}, NIS均值={nis_mean:.2f}")
    
    # 关键发现
    print("\n4. 关键发现:")
    print("   - accel 场景的 NIS 不高（约 4-5），因为方向测量对线性加速度不敏感")
    print("   - 自适应 EKF 的 λ 没有显著增加（约 1.2）")
    print("   - 两种 EKF 的表现几乎相同，因为它们使用相同的 R0")
    print("   - 要让自适应 EKF 优于固定 EKF，需要检测线性加速度并增加 λ")
    
    # 测试幅值感知策略
    print("\n5. 测试幅值感知策略:")
    
    # 使用幅值偏差来检测线性加速度
    adaptive_cfg_mag = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 3.5e-6,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 30, "nis_high": 7.815, "nis_low": 3.0, "ewma_alpha": 0.05},
        "adaptation": {
            "lambda_max": 1000.0,
            "lambda_min": 1.0,
            "use_inflate_mapping": False,
            "use_dynamic_aware": True,
            "mag_threshold": 0.1,  # 更敏感的阈值
            "mag_lambda_gain": 20.0,  # 更大的增益
            "gyro_threshold": 0.05,
            "dynamic_alpha": 0.05,
            "inflate_decay_rate": 0.95,
        },
        "dual_channel": {"enabled": False},
    }
    
    est_mag = run_ekf_adaptive(ds, adaptive_cfg_mag)
    
    roll_err_mag = np.rad2deg(est_mag["roll"] - roll_true)
    pitch_err_mag = np.rad2deg(est_mag["pitch"] - pitch_true)
    rmse_mag = np.sqrt(np.mean(roll_err_mag**2 + pitch_err_mag**2))
    
    lambda_mag_mean = np.mean(est_mag["debug"]["lambda_k"])
    lambda_mag_max = np.max(est_mag["debug"]["lambda_k"])
    
    print(f"   幅值感知策略: RMSE={rmse_mag:.3f}°, λ均值={lambda_mag_mean:.2f}, λ最大={lambda_mag_max:.2f}")
    
    # 分析加速段的 λ
    accel_mask = (t >= 5.0) & (t <= 25.0)
    lambda_accel = est_mag["debug"]["lambda_k"][accel_mask]
    print(f"   加速段 λ 均值: {np.mean(lambda_accel):.2f}")

if __name__ == "__main__":
    main()

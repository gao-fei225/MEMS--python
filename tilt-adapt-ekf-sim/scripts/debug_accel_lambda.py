#!/usr/bin/env python3
"""
深入分析 accel 场景的 λ 值
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
    print("深入分析 accel 场景的 λ 值")
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
    
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    t = ds["t"]
    
    adaptive_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 3.5e-6,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 30, "nis_high": 7.815, "nis_low": 3.0, "ewma_alpha": 0.05},
        "adaptation": {"lambda_max": 1000.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
        "dual_channel": {"enabled": False},
    }
    
    fixed_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    
    est_adapt = run_ekf_adaptive(ds, adaptive_cfg)
    est_fixed = run_ekf_fixed(ds, fixed_cfg)
    
    lambda_k = est_adapt["debug"]["lambda_k"]
    nis_combined = est_adapt["debug"]["nis_combined"]
    
    print(f"\nλ 统计:")
    print(f"  均值: {np.mean(lambda_k):.4f}")
    print(f"  最大: {np.max(lambda_k):.4f}")
    print(f"  最小: {np.min(lambda_k):.4f}")
    print(f"  λ > 1.0 的比例: {np.mean(lambda_k > 1.0) * 100:.2f}%")
    print(f"  λ > 1.01 的比例: {np.mean(lambda_k > 1.01) * 100:.2f}%")
    
    print(f"\nNIS 统计:")
    print(f"  均值: {np.mean(nis_combined):.4f}")
    print(f"  最大: {np.max(nis_combined):.4f}")
    print(f"  NIS > 7.815 的比例: {np.mean(nis_combined > 7.815) * 100:.2f}%")
    
    # 分析误差来源
    roll_err_adapt = np.rad2deg(est_adapt["roll"] - roll_true)
    pitch_err_adapt = np.rad2deg(est_adapt["pitch"] - pitch_true)
    
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    
    print(f"\n误差分析:")
    print(f"  自适应 Roll RMSE: {np.sqrt(np.mean(roll_err_adapt**2)):.4f}°")
    print(f"  固定 Roll RMSE: {np.sqrt(np.mean(roll_err_fixed**2)):.4f}°")
    print(f"  自适应 Pitch RMSE: {np.sqrt(np.mean(pitch_err_adapt**2)):.4f}°")
    print(f"  固定 Pitch RMSE: {np.sqrt(np.mean(pitch_err_fixed**2)):.4f}°")
    
    # 分析 λ > 1 时的误差
    lambda_gt_1_mask = lambda_k > 1.001
    if np.any(lambda_gt_1_mask):
        print(f"\n当 λ > 1.001 时:")
        print(f"  样本数: {np.sum(lambda_gt_1_mask)}")
        print(f"  自适应误差均值: {np.mean(np.sqrt(roll_err_adapt[lambda_gt_1_mask]**2 + pitch_err_adapt[lambda_gt_1_mask]**2)):.4f}°")
        print(f"  固定误差均值: {np.mean(np.sqrt(roll_err_fixed[lambda_gt_1_mask]**2 + pitch_err_fixed[lambda_gt_1_mask]**2)):.4f}°")
    
    # 关键发现
    print("\n关键发现:")
    print(f"  accel 场景的 NIS 均值只有 {np.mean(nis_combined):.2f}，远低于阈值 7.815")
    print(f"  因此 λ 几乎一直是 1.0，自适应 EKF 和固定 EKF 行为几乎相同")
    print(f"  微小的差异来自于 λ 偶尔 > 1.0 时的累积效应")

if __name__ == "__main__":
    main()

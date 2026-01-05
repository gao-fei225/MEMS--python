#!/usr/bin/env python3
"""
调试 shock 场景：分析为什么自适应 EKF 略差于固定 EKF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.truth.scenarios import generate_shock
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
        "meta": {"fs": truth["fs"], "seed": seed, "scenario_name": "shock", "sensor_params": sensor_params},
    }
    validate_dataset(ds)
    return ds

def main():
    print("="*70)
    print("调试 shock 场景")
    print("="*70)
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    truth = generate_shock(
        fs=100, duration_s=20,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        shock_peak=50.0, shock_width_s=0.05,
        shock_times=[5.0, 10.0, 15.0],
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    ds = create_dataset(truth, sensor_params)
    
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    t = ds["t"]
    
    # 测试不同的 inflate_decay_rate
    print("\n测试不同的 inflate_decay_rate:")
    
    for decay_rate in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        adaptive_cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 3.5e-6,
            "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": 7.815, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {"lambda_max": 1000.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": decay_rate},
            "dual_channel": {"enabled": False},
        }
        
        est = run_ekf_adaptive(ds, adaptive_cfg)
        
        roll_err = np.rad2deg(est["roll"] - roll_true)
        pitch_err = np.rad2deg(est["pitch"] - pitch_true)
        rmse = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
        
        lambda_mean = np.mean(est["debug"]["lambda_k"])
        lambda_max = np.max(est["debug"]["lambda_k"])
        
        print(f"  decay_rate={decay_rate}: RMSE={rmse:.4f}°, λ均值={lambda_mean:.2f}, λ最大={lambda_max:.2f}")
    
    # 固定 EKF 基准
    fixed_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    
    est_fixed = run_ekf_fixed(ds, fixed_cfg)
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
    
    print(f"\n  固定 EKF: RMSE={rmse_fixed:.4f}°")
    
    # 测试不同的 lambda_min
    print("\n测试不同的 lambda_min:")
    
    for lambda_min in [0.5, 0.8, 1.0, 1.2, 1.5]:
        adaptive_cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 3.5e-6,
            "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": 7.815, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {"lambda_max": 1000.0, "lambda_min": lambda_min, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
            "dual_channel": {"enabled": False},
        }
        
        est = run_ekf_adaptive(ds, adaptive_cfg)
        
        roll_err = np.rad2deg(est["roll"] - roll_true)
        pitch_err = np.rad2deg(est["pitch"] - pitch_true)
        rmse = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
        
        improvement = (rmse_fixed - rmse) / rmse_fixed * 100
        
        print(f"  lambda_min={lambda_min}: RMSE={rmse:.4f}°, 改善={improvement:+.2f}%")
    
    # 分析冲击时刻的行为
    print("\n分析冲击时刻的行为:")
    
    adaptive_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 3.5e-6,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 30, "nis_high": 7.815, "nis_low": 3.0, "ewma_alpha": 0.05},
        "adaptation": {"lambda_max": 1000.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
        "dual_channel": {"enabled": False},
    }
    
    est = run_ekf_adaptive(ds, adaptive_cfg)
    
    nis_combined = est["debug"]["nis_combined"]
    lambda_k = est["debug"]["lambda_k"]
    
    for shock_t in [5.0, 10.0, 15.0]:
        idx = int(shock_t * 100)
        print(f"\n  t={shock_t}s 附近:")
        print(f"    NIS: {nis_combined[idx-5:idx+10]}")
        print(f"    λ: {lambda_k[idx-5:idx+10]}")
        
        # 计算冲击后的误差
        post_shock_mask = (t >= shock_t) & (t <= shock_t + 0.5)
        roll_err = np.rad2deg(est["roll"] - roll_true)
        pitch_err = np.rad2deg(est["pitch"] - pitch_true)
        total_err = np.sqrt(roll_err**2 + pitch_err**2)
        
        roll_err_f = np.rad2deg(est_fixed["roll"] - roll_true)
        pitch_err_f = np.rad2deg(est_fixed["pitch"] - pitch_true)
        total_err_f = np.sqrt(roll_err_f**2 + pitch_err_f**2)
        
        print(f"    冲击后0.5s内误差: 自适应={np.mean(total_err[post_shock_mask]):.4f}°, 固定={np.mean(total_err_f[post_shock_mask]):.4f}°")

if __name__ == "__main__":
    main()

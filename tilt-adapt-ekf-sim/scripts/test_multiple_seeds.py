#!/usr/bin/env python3
"""
测试多个随机种子，验证结果的稳定性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.truth.scenarios import (
    generate_vibration, generate_shock, generate_swing,
    generate_turn, generate_accel
)
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
        "meta": {"fs": truth["fs"], "seed": seed, "scenario_name": "test", "sensor_params": sensor_params},
    }
    validate_dataset(ds)
    return ds

def test_scenario_with_seed(scenario_name, seed):
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    if scenario_name == "accel":
        truth = generate_accel(fs=100, duration_s=30, roll_deg=0, pitch_deg=0, yaw_deg=0,
                              accel_type="ramp", accel_axis="x", accel_peak=5.0,
                              accel_start_s=5.0, accel_duration_s=20.0, temp_C=25, seed=seed)
    elif scenario_name == "shock":
        truth = generate_shock(fs=100, duration_s=20, roll_deg=0, pitch_deg=0, yaw_deg=0,
                              shock_peak=50.0, shock_width_s=0.05, shock_times=[5.0, 10.0, 15.0],
                              temp_C=25, seed=seed)
    elif scenario_name == "swing":
        truth = generate_swing(fs=100, duration_s=30, roll_amp_deg=15.0, pitch_amp_deg=10.0,
                              roll_freq_hz=0.5, pitch_freq_hz=0.3, roll_phase_deg=0, pitch_phase_deg=90,
                              yaw_deg=0, temp_C=25, seed=seed)
    
    truth["fs"] = 100.0
    ds = create_dataset(truth, sensor_params, seed=seed)
    
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    
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
    
    roll_err_adapt = np.rad2deg(est_adapt["roll"] - roll_true)
    pitch_err_adapt = np.rad2deg(est_adapt["pitch"] - pitch_true)
    rmse_adapt = np.sqrt(np.mean(roll_err_adapt**2 + pitch_err_adapt**2))
    
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
    
    improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
    
    return rmse_adapt, rmse_fixed, improvement

def main():
    print("="*70)
    print("测试多个随机种子")
    print("="*70)
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for scenario in ["accel", "shock", "swing"]:
        print(f"\n{scenario} 场景:")
        
        improvements = []
        for seed in seeds:
            rmse_adapt, rmse_fixed, improvement = test_scenario_with_seed(scenario, seed)
            improvements.append(improvement)
            status = "✓" if improvement > 0 else "✗"
            print(f"  seed={seed}: 自适应={rmse_adapt:.4f}°, 固定={rmse_fixed:.4f}°, 改善={improvement:+.3f}% {status}")
        
        print(f"\n  统计: 均值={np.mean(improvements):+.3f}%, 标准差={np.std(improvements):.3f}%")
        print(f"  优于固定 EKF 的比例: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")

if __name__ == "__main__":
    main()

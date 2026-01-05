#!/usr/bin/env python3
"""
测试最终配置：nis_high=30.0
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

def generate_all_datasets():
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    datasets = {}
    
    truth = generate_vibration(fs=100, duration_s=30, roll_deg=0, pitch_deg=0, yaw_deg=0,
                               vib_rms=0.5, vib_bandwidth_hz=10.0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["vibration"] = create_dataset(truth, sensor_params)
    
    truth = generate_shock(fs=100, duration_s=20, roll_deg=0, pitch_deg=0, yaw_deg=0,
                          shock_peak=50.0, shock_width_s=0.05, shock_times=[5.0, 10.0, 15.0],
                          temp_C=25, seed=42)
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
                          accel_type="ramp", accel_axis="x", accel_peak=5.0,
                          accel_start_s=5.0, accel_duration_s=20.0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["accel"] = create_dataset(truth, sensor_params)
    
    return datasets

def test_config(datasets, adaptive_cfg, fixed_cfg):
    results = {}
    
    for scenario, ds in datasets.items():
        roll_true, pitch_true = get_truth_rpy(ds["truth"])
        
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
        lambda_max = np.max(est_adapt["debug"]["lambda_k"])
        
        results[scenario] = {
            "rmse_adapt": rmse_adapt,
            "rmse_fixed": rmse_fixed,
            "improvement": improvement,
            "lambda_mean": lambda_mean,
            "lambda_max": lambda_max,
        }
    
    return results

def main():
    print("="*70)
    print("测试最终配置")
    print("="*70)
    
    datasets = generate_all_datasets()
    
    fixed_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    
    # 最终配置：nis_high=30.0
    adaptive_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 3.5e-6,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 30, "nis_high": 30.0, "nis_low": 3.0, "ewma_alpha": 0.05},
        "adaptation": {"lambda_max": 1000.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
        "dual_channel": {"enabled": False},
    }
    
    results = test_config(datasets, adaptive_cfg, fixed_cfg)
    
    print("\n最终配置 (nis_high=30.0):")
    print("-"*70)
    
    all_better = True
    for scenario, r in results.items():
        status = "✓" if r["improvement"] >= 0 else "✗"
        if r["improvement"] < 0:
            all_better = False
        print(f"  {scenario}:")
        print(f"    自适应 RMSE: {r['rmse_adapt']:.4f}°")
        print(f"    固定 RMSE: {r['rmse_fixed']:.4f}°")
        print(f"    改善: {r['improvement']:+.4f}% {status}")
        print(f"    λ 均值: {r['lambda_mean']:.2f}, 最大: {r['lambda_max']:.2f}")
    
    print(f"\n全部优于或等于固定 EKF: {'是' if all_better else '否'}")
    
    # 测试多个种子
    print("\n" + "="*70)
    print("多种子测试 (nis_high=30.0)")
    print("="*70)
    
    seeds = [42, 123, 456, 789, 1000]
    
    for scenario_name in ["accel", "shock", "swing"]:
        improvements = []
        
        for seed in seeds:
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
            
            est_adapt = run_ekf_adaptive(ds, adaptive_cfg)
            est_fixed = run_ekf_fixed(ds, fixed_cfg)
            
            roll_err_adapt = np.rad2deg(est_adapt["roll"] - roll_true)
            pitch_err_adapt = np.rad2deg(est_adapt["pitch"] - pitch_true)
            rmse_adapt = np.sqrt(np.mean(roll_err_adapt**2 + pitch_err_adapt**2))
            
            roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
            pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
            rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
            
            improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
            improvements.append(improvement)
        
        n_better = sum(1 for i in improvements if i >= 0)
        print(f"\n{scenario_name}: {n_better}/{len(seeds)} 优于或等于固定 EKF")
        print(f"  改善范围: [{min(improvements):+.4f}%, {max(improvements):+.4f}%]")

if __name__ == "__main__":
    main()

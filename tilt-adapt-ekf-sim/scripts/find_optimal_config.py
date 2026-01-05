#!/usr/bin/env python3
"""
寻找最优配置：让自适应 EKF 在所有场景都优于固定 EKF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from itertools import product

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
    min_improvement = float('inf')
    
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
        
        results[scenario] = {
            "rmse_adapt": rmse_adapt,
            "rmse_fixed": rmse_fixed,
            "improvement": improvement,
        }
        
        min_improvement = min(min_improvement, improvement)
    
    all_better = min_improvement > 0
    return results, all_better, min_improvement

def main():
    print("="*70)
    print("寻找最优配置")
    print("="*70)
    
    datasets = generate_all_datasets()
    
    fixed_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    
    # 网格搜索
    r0_values = [1e-6, 2e-6, 3e-6, 3.5e-6, 4e-6, 5e-6]
    lambda_min_values = [0.9, 0.95, 1.0, 1.02, 1.05]
    decay_rate_values = [0.85, 0.9, 0.95]
    nis_high_values = [7.0, 7.815, 8.5]
    
    best_config = None
    best_min_improvement = -float('inf')
    
    print("\n网格搜索中...")
    
    total = len(r0_values) * len(lambda_min_values) * len(decay_rate_values) * len(nis_high_values)
    count = 0
    
    for r0, lambda_min, decay_rate, nis_high in product(r0_values, lambda_min_values, decay_rate_values, nis_high_values):
        count += 1
        
        adaptive_cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": r0,
            "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": nis_high, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {"lambda_max": 1000.0, "lambda_min": lambda_min, "use_inflate_mapping": True, "inflate_decay_rate": decay_rate},
            "dual_channel": {"enabled": False},
        }
        
        results, all_better, min_improvement = test_config(datasets, adaptive_cfg, fixed_cfg)
        
        if min_improvement > best_min_improvement:
            best_min_improvement = min_improvement
            best_config = {
                "r0": r0,
                "lambda_min": lambda_min,
                "decay_rate": decay_rate,
                "nis_high": nis_high,
                "results": results,
                "all_better": all_better,
            }
            
            if all_better:
                print(f"\n  找到全部优于的配置! (进度: {count}/{total})")
                print(f"    R0={r0:.0e}, lambda_min={lambda_min}, decay_rate={decay_rate}, nis_high={nis_high}")
                for scenario, r in results.items():
                    print(f"      {scenario}: 改善={r['improvement']:+.2f}%")
    
    print("\n" + "="*70)
    print("最佳配置:")
    print("="*70)
    
    if best_config:
        print(f"\n  R0={best_config['r0']:.0e}")
        print(f"  lambda_min={best_config['lambda_min']}")
        print(f"  decay_rate={best_config['decay_rate']}")
        print(f"  nis_high={best_config['nis_high']}")
        print(f"  全部优于: {'是' if best_config['all_better'] else '否'}")
        print(f"  最小改善: {best_min_improvement:+.2f}%")
        
        print("\n  各场景结果:")
        for scenario, r in best_config['results'].items():
            status = "✓" if r["improvement"] > 0 else "✗"
            print(f"    {scenario}: 自适应={r['rmse_adapt']:.4f}°, 固定={r['rmse_fixed']:.4f}°, 改善={r['improvement']:+.3f}% {status}")

if __name__ == "__main__":
    main()

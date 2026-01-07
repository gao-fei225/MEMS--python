#!/usr/bin/env python3
"""
测试 turn-like 门控策略

只对同时满足 ||a||-g 高 AND ||ω|| 高的情况进行门控
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.truth.scenarios import generate_vibration, generate_shock, generate_swing, generate_turn, generate_accel
from src.sensors.imu_model import forward_imu
from src.datasets.validate import validate_dataset
from src.filters.ekf_fixed import run_ekf_fixed
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.common.math3d import quat_to_rpy


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


def test_config(ds, cfg_adaptive, cfg_fixed):
    """测试配置"""
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    
    est_fixed = run_ekf_fixed(ds, cfg_fixed)
    roll_err = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err = np.rad2deg(est_fixed["pitch"] - pitch_true)
    rmse_fixed = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
    
    est_adapt = run_ekf_adaptive(ds, cfg_adaptive)
    roll_err = np.rad2deg(est_adapt["roll"] - roll_true)
    pitch_err = np.rad2deg(est_adapt["pitch"] - pitch_true)
    rmse_adapt = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
    
    nis_mean = np.mean(est_adapt["debug"]["nis"][100:])
    lambda_mean = np.mean(est_adapt["debug"]["lambda_k"])
    
    return rmse_fixed, rmse_adapt, nis_mean, lambda_mean


def main():
    print("=" * 70)
    print("测试 turn-like 门控策略")
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
    
    # 基线配置
    cfg_baseline = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 2e-6,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 30, "nis_high": 15.0, "nis_low": 3.0, "ewma_alpha": 0.05},
        "adaptation": {"lambda_max": 100.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
        "dual_channel": {"enabled": False},
    }
    
    # turn-like 门控配置
    cfg_turn_gating = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 2e-6,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 30, "nis_high": 15.0, "nis_low": 3.0, "ewma_alpha": 0.05},
        "adaptation": {
            "lambda_max": 1000.0,
            "lambda_min": 1.0,
            "use_inflate_mapping": False,
            "use_dynamic_gating": True,
            "dyn_mag_threshold": 0.2,  # ||a||-g 阈值
            "dyn_gyro_threshold": np.deg2rad(10),  # ||ω|| 阈值 (10°/s)
            "dyn_jerk_threshold": 100.0,  # 不使用
            "dyn_lambda_boost": 100.0,  # turn 时的 λ
            "dyn_hysteresis_up": 5,
            "dyn_hysteresis_down": 20,
            "lambda_smooth_alpha": 0.2,
            "lambda_hysteresis": 0.1,
            "inflate_decay_rate": 0.9,
            "dynamic_alpha": 0.1,
        },
        "dual_channel": {"enabled": False},
    }
    
    print(f"\n{'场景':<12} | {'策略':<20} | {'固定':<10} | {'自适应':<10} | {'改善':<10} | {'λ mean':<10}")
    print("-" * 85)
    
    for scenario_name, ds in datasets.items():
        # 基线
        rmse_fixed, rmse_adapt, nis_mean, lambda_mean = test_config(ds, cfg_baseline, cfg_fixed)
        improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
        print(f"{scenario_name:<12} | {'baseline':<20} | {rmse_fixed:<10.3f} | {rmse_adapt:<10.3f} | {improvement:+9.1f}% | {lambda_mean:<10.1f}")
        
        # turn-like 门控
        rmse_fixed, rmse_adapt, nis_mean, lambda_mean = test_config(ds, cfg_turn_gating, cfg_fixed)
        improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
        print(f"{'':<12} | {'turn_gating':<20} | {rmse_fixed:<10.3f} | {rmse_adapt:<10.3f} | {improvement:+9.1f}% | {lambda_mean:<10.1f}")
        print("-" * 85)
    
    # 检查是否所有场景都通过
    print("\n检查所有场景是否 >= 固定 EKF...")
    all_pass = True
    for scenario_name, ds in datasets.items():
        rmse_fixed, rmse_adapt, _, _ = test_config(ds, cfg_turn_gating, cfg_fixed)
        improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
        status = "✓" if improvement >= -0.1 else "✗"
        print(f"  {scenario_name}: {status} ({improvement:+.1f}%)")
        if improvement < -0.1:
            all_pass = False
    
    print(f"\n总体结果: {'✓ PASS' if all_pass else '✗ FAIL'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

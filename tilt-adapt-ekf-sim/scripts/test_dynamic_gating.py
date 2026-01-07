#!/usr/bin/env python3
"""
测试增强动态门控策略 (Plan H)

目标：改善 turn/accel 场景的性能
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


def test_scenario(ds, name, cfg_adaptive, cfg_fixed):
    """测试单个场景"""
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    
    # 固定 EKF
    est_fixed = run_ekf_fixed(ds, cfg_fixed)
    roll_err = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err = np.rad2deg(est_fixed["pitch"] - pitch_true)
    rmse_fixed = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
    
    # 自适应 EKF
    est_adapt = run_ekf_adaptive(ds, cfg_adaptive)
    roll_err = np.rad2deg(est_adapt["roll"] - roll_true)
    pitch_err = np.rad2deg(est_adapt["pitch"] - pitch_true)
    rmse_adapt = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
    
    nis_mean = np.mean(est_adapt["debug"]["nis"][100:])
    lambda_mean = np.mean(est_adapt["debug"]["lambda_k"])
    lambda_max = np.max(est_adapt["debug"]["lambda_k"])
    
    improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
    
    return {
        "rmse_fixed": rmse_fixed,
        "rmse_adapt": rmse_adapt,
        "improvement": improvement,
        "nis_mean": nis_mean,
        "lambda_mean": lambda_mean,
        "lambda_max": lambda_max,
    }


def main():
    print("=" * 70)
    print("测试增强动态门控策略 (Plan H)")
    print("=" * 70)
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    # 生成数据集
    print("\n生成数据集...")
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
    
    # 测试不同配置
    configs = {
        "baseline (inflate)": {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 2e-6,
            "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": 15.0, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {"lambda_max": 100.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
            "dual_channel": {"enabled": False},
        },
        "dynamic_gating (conservative)": {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 2e-6,
            "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": 15.0, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {
                "lambda_max": 10000.0,
                "lambda_min": 1.0,
                "use_inflate_mapping": False,
                "use_dynamic_gating": True,
                "dyn_mag_threshold": 0.5,  # ||a||-g 阈值
                "dyn_gyro_threshold": 0.1,  # ||ω|| 阈值 (~6°/s)
                "dyn_jerk_threshold": 10.0,  # jerk 阈值
                "dyn_lambda_boost": 100.0,  # 动态时的 λ
                "dyn_hysteresis_up": 5,
                "dyn_hysteresis_down": 20,
                "lambda_smooth_alpha": 0.2,
                "lambda_hysteresis": 0.1,
                "inflate_decay_rate": 0.95,
                "dynamic_alpha": 0.1,
            },
            "dual_channel": {"enabled": False},
        },
        "dynamic_gating (aggressive)": {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 2e-6,
            "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": 15.0, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {
                "lambda_max": 100000.0,
                "lambda_min": 1.0,
                "use_inflate_mapping": False,
                "use_dynamic_gating": True,
                "dyn_mag_threshold": 0.3,  # 更敏感
                "dyn_gyro_threshold": 0.05,  # 更敏感 (~3°/s)
                "dyn_jerk_threshold": 5.0,  # 更敏感
                "dyn_lambda_boost": 1000.0,  # 更激进
                "dyn_hysteresis_up": 3,
                "dyn_hysteresis_down": 30,
                "lambda_smooth_alpha": 0.3,
                "lambda_hysteresis": 0.05,
                "inflate_decay_rate": 0.98,
                "dynamic_alpha": 0.05,
            },
            "dual_channel": {"enabled": False},
        },
    }
    
    # 运行测试
    print("\n" + "=" * 100)
    print(f"{'配置':<30} | {'场景':<12} | {'固定 RMSE':<12} | {'自适应 RMSE':<14} | {'改善':<10} | {'λ mean':<10}")
    print("=" * 100)
    
    for cfg_name, cfg in configs.items():
        for scenario_name, ds in datasets.items():
            result = test_scenario(ds, scenario_name, cfg, cfg_fixed)
            print(f"{cfg_name:<30} | {scenario_name:<12} | {result['rmse_fixed']:<12.3f} | {result['rmse_adapt']:<14.3f} | {result['improvement']:+9.1f}% | {result['lambda_mean']:<10.1f}")
        print("-" * 100)
    
    # 详细分析 turn 和 accel 场景
    print("\n" + "=" * 70)
    print("详细分析 turn/accel 场景")
    print("=" * 70)
    
    for scenario_name in ["turn", "accel"]:
        ds = datasets[scenario_name]
        roll_true, pitch_true = get_truth_rpy(ds["truth"])
        
        print(f"\n场景: {scenario_name}")
        
        # 分析加速度幅值偏差
        acc = ds["meas"]["acc"]
        acc_norm = np.linalg.norm(acc, axis=1)
        mag_error = np.abs(acc_norm - 9.80665)
        
        print(f"  加速度幅值偏差: mean={np.mean(mag_error):.3f}, max={np.max(mag_error):.3f} m/s²")
        
        # 分析角速度
        gyro = ds["meas"]["gyro"]
        gyro_norm = np.linalg.norm(gyro, axis=1)
        
        print(f"  角速度幅值: mean={np.rad2deg(np.mean(gyro_norm)):.2f}, max={np.rad2deg(np.max(gyro_norm)):.2f} °/s")
        
        # 使用激进配置测试
        cfg = configs["dynamic_gating (aggressive)"]
        est = run_ekf_adaptive(ds, cfg)
        
        roll_err = np.rad2deg(est["roll"] - roll_true)
        pitch_err = np.rad2deg(est["pitch"] - pitch_true)
        
        print(f"  Roll 误差: mean={np.mean(np.abs(roll_err)):.3f}, max={np.max(np.abs(roll_err)):.3f}°")
        print(f"  Pitch 误差: mean={np.mean(np.abs(pitch_err)):.3f}, max={np.max(np.abs(pitch_err)):.3f}°")
        print(f"  λ: mean={np.mean(est['debug']['lambda_k']):.1f}, max={np.max(est['debug']['lambda_k']):.1f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

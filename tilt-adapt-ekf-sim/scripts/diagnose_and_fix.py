#!/usr/bin/env python3
"""
诊断并修复 EKF 问题：
1. 检查 NIS 计算是否使用膨胀后的 S
2. 标定 R0 使静态 NIS 均值 ≈ 观测维数 m
3. 测试不同 nis_high 阈值
4. 确保公平对比
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

from src.truth.scenarios import generate_vibration, generate_quasi_static
from src.sensors.imu_model import forward_imu
from src.datasets.validate import validate_dataset
from src.filters.ekf_fixed import run_ekf_fixed
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.common.math3d import quat_to_rpy


def create_dataset(truth, sensor_params, seed=42):
    """创建数据集"""
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


def get_truth_rpy(truth):
    n = len(truth["q_nb"])
    roll_true = np.zeros(n)
    pitch_true = np.zeros(n)
    for i in range(n):
        r, p, y = quat_to_rpy(truth["q_nb"][i])
        roll_true[i] = r
        pitch_true[i] = p
    return roll_true, pitch_true


def calibrate_R0_from_static():
    """
    步骤 1: 用静态数据标定 R0
    目标：使静态场景的 NIS 均值 ≈ 观测维数 m (m=3 for 3D direction)
    """
    print("=" * 70)
    print("步骤 1: 用静态数据标定 R0")
    print("=" * 70)
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    # 生成静态数据
    truth = generate_quasi_static(
        fs=100, duration_s=30,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    ds = create_dataset(truth, sensor_params)
    
    # 测试不同 R0 值
    R0_values = [1e-7, 5e-7, 1e-6, 2e-6, 3.5e-6, 5e-6, 1e-5, 2e-5, 5e-5]
    
    print("\n  R0 标定结果:")
    print(f"  {'R0':<12} | {'NIS_raw mean':<12} | {'NIS_adapt mean':<14} | {'RMSE (°)':<10}")
    print(f"  {'-'*12} | {'-'*12} | {'-'*14} | {'-'*10}")
    
    best_R0 = None
    best_diff = float('inf')
    target_nis = 3.0  # 观测维数
    
    for R0 in R0_values:
        cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": R0,
            "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": 7.815, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {"lambda_max": 100.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
            "dual_channel": {"enabled": False},
        }
        
        est = run_ekf_adaptive(ds, cfg)
        roll_true, pitch_true = get_truth_rpy(ds["truth"])
        
        nis_raw = est["debug"]["nis_raw"][100:]  # 跳过 burn-in
        nis_adapt = est["debug"]["nis"][100:]
        
        roll_err = np.rad2deg(est["roll"] - roll_true)
        pitch_err = np.rad2deg(est["pitch"] - pitch_true)
        rmse = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
        
        nis_raw_mean = np.mean(nis_raw)
        nis_adapt_mean = np.mean(nis_adapt)
        
        print(f"  {R0:<12.2e} | {nis_raw_mean:<12.2f} | {nis_adapt_mean:<14.2f} | {rmse:<10.4f}")
        
        # 找最接近目标的 R0
        diff = abs(nis_raw_mean - target_nis)
        if diff < best_diff:
            best_diff = diff
            best_R0 = R0
    
    print(f"\n  推荐 R0 = {best_R0:.2e} (使静态 NIS 均值最接近 {target_nis})")
    return best_R0


def test_nis_high_sweep(R0_calibrated):
    """
    步骤 2: 测试不同 nis_high 阈值
    """
    print("\n" + "=" * 70)
    print("步骤 2: 测试不同 nis_high 阈值 (振动场景)")
    print("=" * 70)
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    truth = generate_vibration(
        fs=100, duration_s=30,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        vib_rms=0.5, vib_bandwidth_hz=10.0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    ds = create_dataset(truth, sensor_params)
    
    # 先跑固定 EKF 作为基线
    fixed_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": R0_calibrated,
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    est_fixed = run_ekf_fixed(ds, fixed_cfg)
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
    
    print(f"\n  固定 EKF 基线 (R={R0_calibrated:.2e}): RMSE = {rmse_fixed:.3f}°")
    
    # 测试不同 nis_high
    nis_high_values = [3.0, 5.0, 7.815, 10.0, 15.0, 20.0, 35.0]
    
    print(f"\n  {'nis_high':<10} | {'RMSE (°)':<10} | {'NIS mean':<10} | {'λ mean':<10} | {'改善':<10}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    
    best_nis_high = None
    best_rmse = float('inf')
    
    for nis_high in nis_high_values:
        cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": R0_calibrated,
            "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": nis_high, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {"lambda_max": 100.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
            "dual_channel": {"enabled": False},
        }
        
        est = run_ekf_adaptive(ds, cfg)
        
        roll_err = np.rad2deg(est["roll"] - roll_true)
        pitch_err = np.rad2deg(est["pitch"] - pitch_true)
        rmse = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
        
        nis_mean = np.mean(est["debug"]["nis"][100:])
        lambda_mean = np.mean(est["debug"]["lambda_k"])
        improvement = (rmse_fixed - rmse) / rmse_fixed * 100
        
        print(f"  {nis_high:<10.1f} | {rmse:<10.3f} | {nis_mean:<10.2f} | {lambda_mean:<10.2f} | {improvement:+.1f}%")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_nis_high = nis_high
    
    print(f"\n  最佳 nis_high = {best_nis_high} (RMSE = {best_rmse:.3f}°)")
    return best_nis_high


def test_pure_inflate_strategy(R0_calibrated, nis_high_best):
    """
    步骤 3: 测试纯膨胀策略（去掉门限硬切）
    """
    print("\n" + "=" * 70)
    print("步骤 3: 测试纯膨胀策略 vs 门限策略")
    print("=" * 70)
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    truth = generate_vibration(
        fs=100, duration_s=30,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        vib_rms=0.5, vib_bandwidth_hz=10.0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    ds = create_dataset(truth, sensor_params)
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    
    # 策略 A: 纯膨胀（连续映射，无门限）
    # λ = max(1, NIS / target_nis)，其中 target_nis = 3
    cfg_pure_inflate = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": R0_calibrated,
        "use_direction_meas": True,
        "innovation_stat": {
            "window_W": 30,
            "nis_high": 3.0,  # 使用观测维数作为阈值
            "nis_low": 1.0,
            "ewma_alpha": 0.05,
        },
        "adaptation": {
            "lambda_max": 100.0,
            "lambda_min": 1.0,
            "use_inflate_mapping": True,
            "inflate_decay_rate": 0.9,
        },
        "dual_channel": {"enabled": False},
    }
    
    est_pure = run_ekf_adaptive(ds, cfg_pure_inflate)
    roll_err = np.rad2deg(est_pure["roll"] - roll_true)
    pitch_err = np.rad2deg(est_pure["pitch"] - pitch_true)
    rmse_pure = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
    nis_mean_pure = np.mean(est_pure["debug"]["nis"][100:])
    lambda_mean_pure = np.mean(est_pure["debug"]["lambda_k"])
    
    # 策略 B: 使用最佳 nis_high
    cfg_best = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": R0_calibrated,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 30, "nis_high": nis_high_best, "nis_low": 3.0, "ewma_alpha": 0.05},
        "adaptation": {"lambda_max": 100.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
        "dual_channel": {"enabled": False},
    }
    
    est_best = run_ekf_adaptive(ds, cfg_best)
    roll_err = np.rad2deg(est_best["roll"] - roll_true)
    pitch_err = np.rad2deg(est_best["pitch"] - pitch_true)
    rmse_best = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
    nis_mean_best = np.mean(est_best["debug"]["nis"][100:])
    lambda_mean_best = np.mean(est_best["debug"]["lambda_k"])
    
    # 固定 EKF
    fixed_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": R0_calibrated,
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    est_fixed = run_ekf_fixed(ds, fixed_cfg)
    roll_err = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err = np.rad2deg(est_fixed["pitch"] - pitch_true)
    rmse_fixed = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
    
    print(f"\n  {'策略':<25} | {'RMSE (°)':<10} | {'NIS mean':<10} | {'λ mean':<10}")
    print(f"  {'-'*25} | {'-'*10} | {'-'*10} | {'-'*10}")
    print(f"  {'固定 EKF':<25} | {rmse_fixed:<10.3f} | {'N/A':<10} | {'1.0':<10}")
    print(f"  {'纯膨胀 (nis_high=3)':<25} | {rmse_pure:<10.3f} | {nis_mean_pure:<10.2f} | {lambda_mean_pure:<10.2f}")
    print(f"  {f'最佳阈值 (nis_high={nis_high_best})':<25} | {rmse_best:<10.3f} | {nis_mean_best:<10.2f} | {lambda_mean_best:<10.2f}")
    
    return rmse_pure, rmse_best, rmse_fixed


def run_all_scenarios_comparison(R0_calibrated, nis_high_best):
    """
    步骤 4: 在所有场景上对比
    """
    print("\n" + "=" * 70)
    print("步骤 4: 所有场景对比 (公平基线)")
    print("=" * 70)
    
    from src.truth.scenarios import generate_shock, generate_swing, generate_turn, generate_accel
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    scenarios = {
        "vibration": generate_vibration(fs=100, duration_s=30, roll_deg=0, pitch_deg=0, yaw_deg=0,
                                        vib_rms=0.5, vib_bandwidth_hz=10.0, temp_C=25, seed=42),
        "shock": generate_shock(fs=100, duration_s=20, roll_deg=0, pitch_deg=0, yaw_deg=0,
                               shock_peak=50.0, shock_width_s=0.05, shock_times=[5.0, 10.0, 15.0], temp_C=25, seed=42),
        "swing": generate_swing(fs=100, duration_s=30, roll_amp_deg=15.0, pitch_amp_deg=10.0,
                               roll_freq_hz=0.5, pitch_freq_hz=0.3, roll_phase_deg=0, pitch_phase_deg=90,
                               yaw_deg=0, temp_C=25, seed=42),
        "turn": generate_turn(fs=100, duration_s=40, roll_deg=0, pitch_deg=0,
                             yaw_rate_dps=30.0, turn_radius_m=10.0, turn_start_s=5.0, turn_duration_s=30.0,
                             temp_C=25, seed=42),
        "accel": generate_accel(fs=100, duration_s=30, roll_deg=0, pitch_deg=0, yaw_deg=0,
                               accel_type="ramp", accel_axis="x", accel_peak=5.0, accel_start_s=5.0,
                               accel_duration_s=20.0, temp_C=25, seed=42),
    }
    
    print(f"\n  使用标定后的 R0 = {R0_calibrated:.2e}, nis_high = {nis_high_best}")
    print(f"\n  {'场景':<12} | {'固定 RMSE':<12} | {'自适应 RMSE':<14} | {'NIS mean':<10} | {'改善':<10}")
    print(f"  {'-'*12} | {'-'*12} | {'-'*14} | {'-'*10} | {'-'*10}")
    
    results = {}
    
    for name, truth in scenarios.items():
        truth["fs"] = 100.0
        ds = create_dataset(truth, sensor_params)
        roll_true, pitch_true = get_truth_rpy(ds["truth"])
        
        # 固定 EKF
        fixed_cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R_acc": R0_calibrated,
            "use_direction_meas": True,
            "nis_gating": {"enabled": False},
        }
        est_fixed = run_ekf_fixed(ds, fixed_cfg)
        roll_err = np.rad2deg(est_fixed["roll"] - roll_true)
        pitch_err = np.rad2deg(est_fixed["pitch"] - pitch_true)
        rmse_fixed = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
        
        # 自适应 EKF
        adapt_cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": R0_calibrated,
            "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": nis_high_best, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {"lambda_max": 100.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
            "dual_channel": {"enabled": False},
        }
        est_adapt = run_ekf_adaptive(ds, adapt_cfg)
        roll_err = np.rad2deg(est_adapt["roll"] - roll_true)
        pitch_err = np.rad2deg(est_adapt["pitch"] - pitch_true)
        rmse_adapt = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
        
        nis_mean = np.mean(est_adapt["debug"]["nis"][100:])
        improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
        
        results[name] = {
            "rmse_fixed": rmse_fixed,
            "rmse_adapt": rmse_adapt,
            "nis_mean": nis_mean,
            "improvement": improvement,
        }
        
        print(f"  {name:<12} | {rmse_fixed:<12.3f} | {rmse_adapt:<14.3f} | {nis_mean:<10.2f} | {improvement:+.1f}%")
    
    # 检查是否所有场景都 >= 固定 EKF
    all_pass = all(r["improvement"] >= -0.1 for r in results.values())
    print(f"\n  所有场景自适应 >= 固定: {'✓ PASS' if all_pass else '✗ FAIL'}")
    
    return results


def main():
    print("=" * 70)
    print("EKF 诊断与修复")
    print("=" * 70)
    
    # 步骤 1: 标定 R0
    R0_calibrated = calibrate_R0_from_static()
    
    # 步骤 2: 测试 nis_high
    nis_high_best = test_nis_high_sweep(R0_calibrated)
    
    # 步骤 3: 测试纯膨胀策略
    test_pure_inflate_strategy(R0_calibrated, nis_high_best)
    
    # 步骤 4: 所有场景对比
    results = run_all_scenarios_comparison(R0_calibrated, nis_high_best)
    
    print("\n" + "=" * 70)
    print("推荐配置")
    print("=" * 70)
    print(f"""
DEFAULT_ADAPTIVE_CFG = {{
    "Q_gyro": 1e-5,
    "Q_bias": 1e-8,
    "R0": {R0_calibrated:.2e},
    "use_direction_meas": True,
    "innovation_stat": {{
        "window_W": 30,
        "nis_high": {nis_high_best},
        "nis_low": 3.0,
        "ewma_alpha": 0.05,
    }},
    "adaptation": {{
        "lambda_max": 100.0,
        "lambda_min": 1.0,
        "use_inflate_mapping": True,
        "inflate_decay_rate": 0.9,
    }},
    "dual_channel": {{"enabled": False}},
}}
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

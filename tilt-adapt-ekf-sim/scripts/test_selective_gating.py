#!/usr/bin/env python3
"""
测试选择性门控策略

核心思想：
1. 对于 turn 场景（高角速度）：降低加速度计权重
2. 对于 accel 场景（高线性加速度，低角速度）：这是 IMU 的固有限制，无法通过自适应解决
3. 对于 swing 场景（周期性运动）：保持正常更新

关键洞察：
- turn/accel 的 13° 误差是 IMU 的物理限制（无法区分重力和线性加速度）
- 自适应 EKF 能做的是：在动态时减少错误更新，但无法恢复真实姿态
- 最好的策略是：在动态时"冻结"姿态估计，等待静态时再校正
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


def analyze_scenario(ds, name):
    """分析场景特征"""
    acc = ds["meas"]["acc"]
    gyro = ds["meas"]["gyro"]
    
    acc_norm = np.linalg.norm(acc, axis=1)
    mag_error = np.abs(acc_norm - 9.80665)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    
    # 计算 jerk
    dt = 1.0 / ds["meta"]["fs"]
    jerk = np.zeros(len(acc))
    jerk[1:] = np.linalg.norm(np.diff(acc, axis=0), axis=1) / dt
    
    print(f"\n场景: {name}")
    print(f"  ||a||-g: mean={np.mean(mag_error):.3f}, max={np.max(mag_error):.3f}, std={np.std(mag_error):.3f} m/s²")
    print(f"  ||ω||: mean={np.rad2deg(np.mean(gyro_norm)):.2f}, max={np.rad2deg(np.max(gyro_norm)):.2f} °/s")
    print(f"  jerk: mean={np.mean(jerk):.2f}, max={np.max(jerk):.2f} m/s³")
    
    # 动态检测统计
    mag_dynamic = np.mean(mag_error > 0.3) * 100
    gyro_dynamic = np.mean(gyro_norm > np.deg2rad(5)) * 100
    
    print(f"  动态比例: ||a||-g>0.3={mag_dynamic:.1f}%, ||ω||>5°/s={gyro_dynamic:.1f}%")
    
    return {
        "mag_error_mean": np.mean(mag_error),
        "mag_error_max": np.max(mag_error),
        "gyro_norm_mean": np.rad2deg(np.mean(gyro_norm)),
        "gyro_norm_max": np.rad2deg(np.max(gyro_norm)),
    }


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
    
    return rmse_fixed, rmse_adapt, est_adapt


def main():
    print("=" * 70)
    print("选择性门控策略分析")
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
    
    # 分析各场景特征
    print("\n" + "=" * 70)
    print("场景特征分析")
    print("=" * 70)
    
    for name, ds in datasets.items():
        analyze_scenario(ds, name)
    
    # 固定 EKF 配置
    cfg_fixed = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 2e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    
    # 测试不同策略
    print("\n" + "=" * 70)
    print("策略对比")
    print("=" * 70)
    
    # 策略 1: 基线 (inflate)
    cfg_baseline = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 2e-6,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 30, "nis_high": 15.0, "nis_low": 3.0, "ewma_alpha": 0.05},
        "adaptation": {"lambda_max": 100.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
        "dual_channel": {"enabled": False},
    }
    
    # 策略 2: 只对高角速度场景门控（针对 turn）
    cfg_gyro_gating = {
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
            "dyn_mag_threshold": 10.0,  # 很高，基本不触发
            "dyn_gyro_threshold": 0.1,  # ~6°/s，只对 turn 触发
            "dyn_jerk_threshold": 100.0,  # 很高，基本不触发
            "dyn_lambda_boost": 100.0,
            "dyn_hysteresis_up": 5,
            "dyn_hysteresis_down": 20,
            "lambda_smooth_alpha": 0.2,
            "lambda_hysteresis": 0.1,
            "inflate_decay_rate": 0.95,
            "dynamic_alpha": 0.1,
        },
        "dual_channel": {"enabled": False},
    }
    
    # 策略 3: 结合 inflate + 高角速度门控
    cfg_combined = {
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
            "dyn_mag_threshold": 1.0,  # 较高，只对明显线性加速度触发
            "dyn_gyro_threshold": 0.15,  # ~9°/s，只对明显转弯触发
            "dyn_jerk_threshold": 50.0,  # 较高
            "dyn_lambda_boost": 50.0,  # 适度提升
            "dyn_hysteresis_up": 10,  # 更长的延迟
            "dyn_hysteresis_down": 30,
            "lambda_smooth_alpha": 0.1,  # 更平滑
            "lambda_hysteresis": 0.2,
            "inflate_decay_rate": 0.9,
            "dynamic_alpha": 0.05,  # 更平滑
        },
        "dual_channel": {"enabled": False},
    }
    
    configs = {
        "baseline (inflate)": cfg_baseline,
        "gyro_gating (turn优化)": cfg_gyro_gating,
        "combined (保守)": cfg_combined,
    }
    
    print(f"\n{'配置':<25} | {'场景':<12} | {'固定':<10} | {'自适应':<10} | {'改善':<10}")
    print("-" * 80)
    
    for cfg_name, cfg in configs.items():
        for scenario_name, ds in datasets.items():
            rmse_fixed, rmse_adapt, _ = test_config(ds, cfg, cfg_fixed)
            improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
            print(f"{cfg_name:<25} | {scenario_name:<12} | {rmse_fixed:<10.3f} | {rmse_adapt:<10.3f} | {improvement:+9.1f}%")
        print("-" * 80)
    
    # 结论
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
1. turn/accel 场景的 ~13° 误差是 IMU 的物理限制：
   - 加速度计无法区分重力和线性加速度
   - 这不是滤波器的问题，而是传感器的固有限制

2. 自适应 EKF 能做的改进有限：
   - 在动态时降低加速度计权重，减少错误更新
   - 但无法恢复真实姿态（因为没有额外信息）

3. 要真正改善 turn/accel 性能，需要：
   - 额外传感器（GPS、里程计、视觉）
   - 运动学约束（车辆模型）
   - 或者接受这是 IMU 的局限性

4. 对于论文/答辩：
   - 明确说明这是 IMU 的物理限制
   - 展示自适应 EKF 在 vibration/shock 场景的优势
   - turn/accel 作为局限性讨论
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

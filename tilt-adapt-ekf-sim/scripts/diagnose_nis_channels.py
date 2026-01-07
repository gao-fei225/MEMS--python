#!/usr/bin/env python3
"""
诊断 NIS 双通道是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.truth.scenarios import generate_vibration, generate_turn, generate_accel
from src.sensors.imu_model import forward_imu
from src.datasets.validate import validate_dataset
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


def diagnose_scenario(ds, name, cfg):
    """诊断单个场景的 NIS 通道"""
    print(f"\n{'='*60}")
    print(f"场景: {name}")
    print(f"{'='*60}")
    
    est = run_ekf_adaptive(ds, cfg)
    
    # 提取调试信息
    nis_raw = est["debug"]["nis_raw"]  # 方向 NIS (用 R0 计算)
    nis_mag = est["debug"]["nis_mag"]  # 幅值 NIS
    nis_combined = est["debug"]["nis_combined"]  # 组合 NIS
    nis_adaptive = est["debug"]["nis"]  # 自适应后的 NIS (用 λ*R0 计算)
    lambda_k = est["debug"]["lambda_k"]
    
    # 跳过 burn-in
    start = 100
    
    print(f"\n1. NIS 通道统计 (跳过前 {start} 样本):")
    print(f"   NIS_dir (方向):  mean={np.mean(nis_raw[start:]):.2f}, max={np.max(nis_raw[start:]):.2f}, std={np.std(nis_raw[start:]):.2f}")
    print(f"   NIS_mag (幅值):  mean={np.mean(nis_mag[start:]):.2f}, max={np.max(nis_mag[start:]):.2f}, std={np.std(nis_mag[start:]):.2f}")
    print(f"   NIS_combined:    mean={np.mean(nis_combined[start:]):.2f}, max={np.max(nis_combined[start:]):.2f}")
    print(f"   NIS_adaptive:    mean={np.mean(nis_adaptive[start:]):.2f}, max={np.max(nis_adaptive[start:]):.2f}")
    
    print(f"\n2. λ 统计:")
    print(f"   mean={np.mean(lambda_k):.2f}, max={np.max(lambda_k):.2f}, min={np.min(lambda_k):.2f}")
    print(f"   λ > 1 的比例: {np.mean(lambda_k > 1.1) * 100:.1f}%")
    print(f"   λ > 10 的比例: {np.mean(lambda_k > 10) * 100:.1f}%")
    print(f"   λ > 100 的比例: {np.mean(lambda_k > 100) * 100:.1f}%")
    
    # 检查 NIS_mag 是否在工作
    if np.max(nis_mag) < 0.01:
        print(f"\n   ⚠️ 警告: NIS_mag 几乎为 0，幅值通道可能没有工作！")
    
    # 检查 NIS_combined 是否等于 NIS_dir
    if np.allclose(nis_combined, nis_raw):
        print(f"\n   ⚠️ 警告: NIS_combined ≈ NIS_dir，幅值通道没有贡献！")
    
    # 分析加速度幅值偏差
    acc = ds["meas"]["acc"]
    acc_norm = np.linalg.norm(acc, axis=1)
    mag_error = np.abs(acc_norm - 9.80665)
    
    print(f"\n3. 加速度幅值偏差 ||a||-g:")
    print(f"   mean={np.mean(mag_error):.3f}, max={np.max(mag_error):.3f} m/s²")
    print(f"   > 0.15g (1.47 m/s²): {np.mean(mag_error > 1.47) * 100:.1f}%")
    print(f"   > 0.3g (2.94 m/s²): {np.mean(mag_error > 2.94) * 100:.1f}%")
    
    # 分析角速度
    gyro = ds["meas"]["gyro"]
    gyro_norm = np.linalg.norm(gyro, axis=1)
    
    print(f"\n4. 角速度 ||ω||:")
    print(f"   mean={np.rad2deg(np.mean(gyro_norm)):.2f}, max={np.rad2deg(np.max(gyro_norm)):.2f} °/s")
    
    # 分析 gyro bias 估计
    bias_gyro = est["bias_gyro"]
    
    print(f"\n5. Gyro bias 估计:")
    print(f"   最终值: [{bias_gyro[-1, 0]:.6f}, {bias_gyro[-1, 1]:.6f}, {bias_gyro[-1, 2]:.6f}] rad/s")
    print(f"   最终值: [{np.rad2deg(bias_gyro[-1, 0]):.4f}, {np.rad2deg(bias_gyro[-1, 1]):.4f}, {np.rad2deg(bias_gyro[-1, 2]):.4f}] °/s")
    print(f"   变化范围: [{np.min(bias_gyro):.6f}, {np.max(bias_gyro):.6f}] rad/s")
    
    # 打印 1 秒的详细数据
    print(f"\n6. 详细数据 (t=5.0~5.1s, 10 samples):")
    fs = ds["meta"]["fs"]
    t_start = int(5.0 * fs)
    t_end = t_start + 10
    
    print(f"   {'t':<6} | {'NIS_dir':<10} | {'NIS_mag':<10} | {'NIS_comb':<10} | {'λ':<10} | {'||a||-g':<10}")
    print(f"   {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    for i in range(t_start, min(t_end, len(nis_raw))):
        t = i / fs
        print(f"   {t:<6.2f} | {nis_raw[i]:<10.2f} | {nis_mag[i]:<10.2f} | {nis_combined[i]:<10.2f} | {lambda_k[i]:<10.2f} | {mag_error[i]:<10.3f}")
    
    return est


def main():
    print("=" * 60)
    print("NIS 双通道诊断")
    print("=" * 60)
    
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
    
    # 测试配置 - 启用双通道
    cfg_dual = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 2e-6,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 30, "nis_high": 15.0, "nis_low": 3.0, "ewma_alpha": 0.05},
        "adaptation": {"lambda_max": 100.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
        "dual_channel": {
            "enabled": True,  # 启用双通道
            "mag_weight": 1.0,
            "mag_sigma": 0.5,  # 幅值噪声标准差
            "combine_mode": "max",
        },
    }
    
    # 测试配置 - 单通道
    cfg_single = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 2e-6,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 30, "nis_high": 15.0, "nis_low": 3.0, "ewma_alpha": 0.05},
        "adaptation": {"lambda_max": 100.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
        "dual_channel": {"enabled": False},
    }
    
    print("\n" + "=" * 60)
    print("测试双通道配置")
    print("=" * 60)
    
    for name, ds in datasets.items():
        diagnose_scenario(ds, name, cfg_dual)
    
    print("\n" + "=" * 60)
    print("测试单通道配置")
    print("=" * 60)
    
    for name, ds in datasets.items():
        diagnose_scenario(ds, name, cfg_single)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

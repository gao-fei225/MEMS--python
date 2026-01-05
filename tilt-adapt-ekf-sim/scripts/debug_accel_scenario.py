#!/usr/bin/env python3
"""
调试 accel 场景：分析为什么自适应 EKF 不如固定 EKF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

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
    print("调试 accel 场景")
    print("="*70)
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    # 生成 accel 场景
    truth = generate_accel(
        fs=100, duration_s=30,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        accel_type="ramp", accel_axis="x",
        accel_peak=5.0, accel_start_s=5.0, accel_duration_s=20.0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    ds = create_dataset(truth, sensor_params)
    
    # 当前自适应配置
    adaptive_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 3.5e-6,
        "use_direction_meas": True,
        "innovation_stat": {
            "window_W": 30,
            "nis_high": 7.815,
            "nis_low": 3.0,
            "ewma_alpha": 0.05,
        },
        "adaptation": {
            "lambda_max": 1000.0,
            "lambda_min": 1.0,
            "use_inflate_mapping": True,
            "inflate_decay_rate": 0.9,
        },
        "dual_channel": {"enabled": False},
    }
    
    fixed_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    
    # 运行滤波器
    est_adapt = run_ekf_adaptive(ds, adaptive_cfg)
    est_fixed = run_ekf_fixed(ds, fixed_cfg)
    
    # 获取真值
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    
    # 计算误差
    roll_err_adapt = np.rad2deg(est_adapt["roll"] - roll_true)
    pitch_err_adapt = np.rad2deg(est_adapt["pitch"] - pitch_true)
    total_err_adapt = np.sqrt(roll_err_adapt**2 + pitch_err_adapt**2)
    
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    total_err_fixed = np.sqrt(roll_err_fixed**2 + pitch_err_fixed**2)
    
    rmse_adapt = np.sqrt(np.mean(total_err_adapt**2))
    rmse_fixed = np.sqrt(np.mean(total_err_fixed**2))
    
    print(f"\n当前结果:")
    print(f"  自适应 RMSE: {rmse_adapt:.4f}°")
    print(f"  固定 RMSE: {rmse_fixed:.4f}°")
    print(f"  改善: {(rmse_fixed - rmse_adapt) / rmse_fixed * 100:+.2f}%")
    
    # 分析 NIS 和 λ
    t = ds["t"]
    nis_combined = est_adapt["debug"]["nis_combined"]
    nis_adaptive = est_adapt["debug"]["nis"]
    lambda_k = est_adapt["debug"]["lambda_k"]
    a_lin_n = truth["a_lin_n"]
    
    print(f"\n  NIS 统计:")
    print(f"    原始 NIS 均值: {np.mean(nis_combined):.2f}")
    print(f"    原始 NIS 最大: {np.max(nis_combined):.2f}")
    print(f"    自适应 NIS 均值: {np.mean(nis_adaptive):.2f}")
    print(f"    λ 均值: {np.mean(lambda_k):.2f}")
    print(f"    λ 最大: {np.max(lambda_k):.2f}")
    
    # 分析加速段
    accel_mask = (t >= 5.0) & (t <= 25.0)
    print(f"\n  加速段 (5-25s) 分析:")
    print(f"    加速度幅值: {np.max(np.abs(a_lin_n[accel_mask, 0])):.2f} m/s²")
    print(f"    原始 NIS 均值: {np.mean(nis_combined[accel_mask]):.2f}")
    print(f"    λ 均值: {np.mean(lambda_k[accel_mask]):.2f}")
    print(f"    自适应误差均值: {np.mean(total_err_adapt[accel_mask]):.4f}°")
    print(f"    固定误差均值: {np.mean(total_err_fixed[accel_mask]):.4f}°")
    
    # 绘图
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    
    ax = axes[0]
    ax.plot(t, a_lin_n[:, 0], 'b-', label='a_lin_x')
    ax.set_ylabel('加速度 (m/s²)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Accel 场景分析')
    
    ax = axes[1]
    ax.plot(t, total_err_adapt, 'b-', label=f'自适应 (RMSE={rmse_adapt:.3f}°)')
    ax.plot(t, total_err_fixed, 'r--', label=f'固定 (RMSE={rmse_fixed:.3f}°)')
    ax.set_ylabel('姿态误差 (°)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(t, nis_combined, 'gray', label='原始 NIS', alpha=0.5)
    ax.plot(t, nis_adaptive, 'b-', label='自适应 NIS')
    ax.axhline(y=7.815, color='r', linestyle='--', label='阈值')
    ax.set_ylabel('NIS')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[3]
    ax.plot(t, lambda_k, 'g-')
    ax.set_ylabel('λ')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax = axes[4]
    ax.plot(t, roll_err_adapt, 'b-', label='Roll 误差')
    ax.plot(t, pitch_err_adapt, 'r-', label='Pitch 误差')
    ax.set_ylabel('分量误差 (°)')
    ax.set_xlabel('时间 (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tilt-adapt-ekf-sim/outputs/debug_accel.png', dpi=150)
    print(f"\n图表已保存到: tilt-adapt-ekf-sim/outputs/debug_accel.png")
    
    # 测试不同的 λ 策略
    print("\n" + "="*70)
    print("测试不同策略")
    print("="*70)
    
    strategies = [
        ("当前 inflate", {"use_inflate_mapping": True, "inflate_decay_rate": 0.9}),
        ("inflate + 更快衰减", {"use_inflate_mapping": True, "inflate_decay_rate": 0.95}),
        ("inflate + 更慢衰减", {"use_inflate_mapping": True, "inflate_decay_rate": 0.7}),
        ("动态感知", {"use_inflate_mapping": False, "use_dynamic_aware": True, "mag_threshold": 0.2, "mag_lambda_gain": 10.0}),
    ]
    
    for name, adapt_params in strategies:
        cfg = adaptive_cfg.copy()
        cfg["adaptation"] = {**adaptive_cfg["adaptation"], **adapt_params}
        
        est = run_ekf_adaptive(ds, cfg)
        roll_err = np.rad2deg(est["roll"] - roll_true)
        pitch_err = np.rad2deg(est["pitch"] - pitch_true)
        rmse = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
        improvement = (rmse_fixed - rmse) / rmse_fixed * 100
        
        print(f"  {name}: RMSE={rmse:.4f}°, 改善={improvement:+.2f}%")

if __name__ == "__main__":
    main()

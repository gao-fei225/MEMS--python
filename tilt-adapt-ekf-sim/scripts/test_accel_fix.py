#!/usr/bin/env python
"""
测试 Accel 场景修复效果

验证双通道检测 + 动态感知策略是否能解决加减速场景的"动态失灵"问题。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from src.truth.scenarios import generate_accel
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.common.math3d import rad2deg


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_test():
    # 加载配置
    filter_cfg = load_yaml("configs/filters/ekf_adaptive_innovation.yaml")
    
    # 场景配置
    scenario_params = {
        "fs": 100.0,
        "duration_s": 30.0,
        "roll_deg": 2.0,
        "pitch_deg": -1.0,
        "yaw_deg": 0.0,
        "accel_type": "step",
        "accel_axis": "x",
        "accel_peak": 2.0,  # 约 0.2g
        "accel_start_s": 5.0,
        "accel_duration_s": 10.0,
        "temp_C": 25.0,
        "seed": 1,
    }
    
    # 传感器配置
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    print("=" * 60)
    print("测试配置:")
    print(f"  场景: accel (step)")
    print(f"  加速度峰值: {scenario_params['accel_peak']} m/s²")
    print(f"  双通道: {filter_cfg.get('dual_channel', {}).get('enabled', False)}")
    print(f"  mag_sigma: {filter_cfg.get('dual_channel', {}).get('mag_sigma', 0.5)}")
    print(f"  mag_weight: {filter_cfg.get('dual_channel', {}).get('mag_weight', 1.0)}")
    print(f"  动态感知: {filter_cfg.get('adaptation', {}).get('use_dynamic_aware', False)}")
    print(f"  mag_threshold: {filter_cfg.get('adaptation', {}).get('mag_threshold', 0.3)}")
    print(f"  mag_lambda_gain: {filter_cfg.get('adaptation', {}).get('mag_lambda_gain', 5.0)}")
    print(f"  lambda_max: {filter_cfg.get('adaptation', {}).get('lambda_max', 100.0)}")
    print("=" * 60)
    
    # 生成真值轨迹
    print("\n生成数据...")
    truth = generate_accel(**scenario_params)
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params["fs"]},
    }
    
    # 运行自适应 EKF
    print("运行自适应 EKF...")
    result = run_ekf_adaptive(ds, filter_cfg)
    
    # 计算误差
    roll_true = truth["rpy_deg"][:, 0]
    pitch_true = truth["rpy_deg"][:, 1]
    roll_est = rad2deg(result["roll"])
    pitch_est = rad2deg(result["pitch"])
    
    roll_err = roll_est - roll_true
    pitch_err = pitch_est - pitch_true
    total_err = np.sqrt(roll_err**2 + pitch_err**2)
    
    # 统计（跳过前1秒）
    burn_in = int(scenario_params["fs"])
    rmse = np.sqrt(np.mean(total_err[burn_in:]**2))
    max_err = np.max(np.abs(total_err[burn_in:]))
    
    print(f"\n结果:")
    print(f"  RMSE: {rmse:.3f}°")
    print(f"  最大误差: {max_err:.3f}°")
    print(f"  λ 最大值: {np.max(result['debug']['lambda_k']):.1f}")
    print(f"  λ 平均值: {np.mean(result['debug']['lambda_k'][burn_in:]):.1f}")
    
    # 分段统计
    t = truth["t"]
    static_mask = (t < 5.0) | (t > 15.0)
    accel_mask = (t >= 5.0) & (t <= 15.0)
    
    rmse_static = np.sqrt(np.mean(total_err[static_mask]**2))
    rmse_accel = np.sqrt(np.mean(total_err[accel_mask]**2))
    lambda_accel_mean = np.mean(result['debug']['lambda_k'][accel_mask])
    
    print(f"\n分段统计:")
    print(f"  静止段 RMSE: {rmse_static:.3f}°")
    print(f"  加速段 RMSE: {rmse_accel:.3f}°")
    print(f"  加速段 λ 均值: {lambda_accel_mean:.1f}")
    
    # 判断是否通过
    if rmse < 5.0:
        print(f"\n✓ 测试通过! RMSE < 5°")
    else:
        print(f"\n✗ 测试失败! RMSE >= 5°")
    
    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # 1. 姿态误差
    ax = axes[0]
    ax.plot(t, roll_err, label="Roll Error", alpha=0.8)
    ax.plot(t, pitch_err, label="Pitch Error", alpha=0.8)
    ax.plot(t, total_err, "k--", label="Total Error", alpha=0.6)
    ax.axhline(5, color="r", linestyle=":", label="5° threshold")
    ax.axhline(-5, color="r", linestyle=":")
    ax.axvspan(5, 15, alpha=0.2, color='yellow', label='Accel period')
    ax.set_ylabel("Error (°)")
    ax.set_title(f"Accel Scenario Fix Test - RMSE: {rmse:.3f}°, Max: {max_err:.3f}°")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # 2. 加速度幅值
    ax = axes[1]
    acc_norm = np.linalg.norm(ds["meas"]["acc"], axis=1)
    ax.plot(t, acc_norm, label="||acc||")
    ax.axhline(GRAVITY_STANDARD, color="r", linestyle="--", label="g")
    ax.axvspan(5, 15, alpha=0.2, color='yellow')
    ax.set_ylabel("Acc (m/s²)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # 3. NIS
    ax = axes[2]
    ax.semilogy(t, result["debug"]["nis_raw"] + 0.1, label="NIS_dir", alpha=0.7)
    ax.semilogy(t, result["debug"]["nis_mag"] + 0.1, label="NIS_mag", alpha=0.7)
    ax.semilogy(t, result["debug"]["nis_combined"] + 0.1, label="NIS_combined", alpha=0.7)
    ax.axhline(7.815, color="r", linestyle="--", label="χ²(3,0.95)")
    ax.axvspan(5, 15, alpha=0.2, color='yellow')
    ax.set_ylabel("NIS")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # 4. Lambda
    ax = axes[3]
    ax.semilogy(t, result["debug"]["lambda_k"], label="λ")
    ax.axhline(1, color="gray", linestyle="--")
    ax.axvspan(5, 15, alpha=0.2, color='yellow')
    ax.set_ylabel("λ")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path("outputs/accel_fix_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "accel_fix_result.png", dpi=150)
    print(f"\n图片已保存到: {output_dir / 'accel_fix_result.png'}")
    
    plt.close()
    
    return rmse < 5.0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

#!/usr/bin/env python
"""
测试静止段在线零偏估计功能

对比：
1. 标准互补滤波
2. 带静止校准的互补滤波

预期：静止校准后，准静态工况的稳态偏置应趋近 0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.truth.scenarios import generate_quasi_static
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.complementary import (
    run_complementary,
    run_complementary_with_static_calibration,
)
from src.metrics.tilt_error import compute_tilt_metrics, print_tilt_metrics
from src.common.math3d import quat_to_rpy, rad2deg


def main():
    print("=" * 60)
    print("静止段在线零偏估计测试")
    print("=" * 60)
    
    # 工况配置
    scenario_params = {
        "fs": 100.0,
        "duration_s": 30.0,
        "roll_deg": 5.0,
        "pitch_deg": -3.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 1,
    }
    
    # 传感器配置（有偏置）
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    # 滤波器配置
    filter_cfg = {"alpha": 0.98}
    
    # 生成数据
    print("\n生成数据...")
    truth = generate_quasi_static(**scenario_params)
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params["fs"]},
    }
    
    # 准备真值格式
    n_samples = len(truth["t"])
    rpy_deg = np.zeros((n_samples, 3), dtype=np.float64)
    for i in range(n_samples):
        roll, pitch, yaw = quat_to_rpy(truth["q_nb"][i])
        rpy_deg[i] = [rad2deg(roll), rad2deg(pitch), rad2deg(yaw)]
    truth_for_metrics = {"rpy_deg": rpy_deg}
    
    # ========== 测试 1: 标准互补滤波 ==========
    print("\n" + "=" * 60)
    print("测试 1: 标准互补滤波")
    print("=" * 60)
    
    est1 = run_complementary(ds, filter_cfg)
    metrics1 = compute_tilt_metrics(
        truth=truth_for_metrics,
        est=est1,
        burn_in_s=1.0,
        fs=scenario_params["fs"],
    )
    print_tilt_metrics(metrics1, name="标准互补滤波", fs=scenario_params["fs"])
    
    # ========== 测试 2: 带静止校准的互补滤波（使用真值） ==========
    print("\n" + "=" * 60)
    print("测试 2: 带静止校准的互补滤波（使用真值校准）")
    print("=" * 60)
    
    # 传入真值用于校准
    est2 = run_complementary_with_static_calibration(ds, filter_cfg, truth=truth_for_metrics)
    metrics2 = compute_tilt_metrics(
        truth=truth_for_metrics,
        est=est2,
        burn_in_s=1.0,
        fs=scenario_params["fs"],
    )
    print_tilt_metrics(metrics2, name="静止校准互补滤波", fs=scenario_params["fs"])
    
    # 打印校准信息
    if "static_calibration" in est2:
        cal = est2["static_calibration"]
        print("\n校准信息:")
        print(f"  静止段比例: {cal['static_ratio']*100:.1f}%")
        if cal.get("calibrated", False):
            print(f"  Roll 偏置补偿: {cal['roll_bias_deg']:.4f}°")
            print(f"  Pitch 偏置补偿: {cal['pitch_bias_deg']:.4f}°")
    
    # ========== 对比 ==========
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    
    print(f"\n稳态偏置改善:")
    print(f"  Roll:  {abs(metrics1['bias_roll']):.4f}° → {abs(metrics2['bias_roll']):.4f}° "
          f"(改善 {(1 - abs(metrics2['bias_roll'])/abs(metrics1['bias_roll']))*100:.1f}%)")
    print(f"  Pitch: {abs(metrics1['bias_pitch']):.4f}° → {abs(metrics2['bias_pitch']):.4f}° "
          f"(改善 {(1 - abs(metrics2['bias_pitch'])/abs(metrics1['bias_pitch']))*100:.1f}%)")
    
    print(f"\nRMSE 改善:")
    print(f"  Roll:  {metrics1['rmse_roll']:.4f}° → {metrics2['rmse_roll']:.4f}° "
          f"(改善 {(1 - metrics2['rmse_roll']/metrics1['rmse_roll'])*100:.1f}%)")
    print(f"  Pitch: {metrics1['rmse_pitch']:.4f}° → {metrics2['rmse_pitch']:.4f}° "
          f"(改善 {(1 - metrics2['rmse_pitch']/metrics1['rmse_pitch'])*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

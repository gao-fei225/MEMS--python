#!/usr/bin/env python
"""
指标与可视化自检脚本

验证内容：
1. compute_tilt_metrics 函数
2. 可视化函数（不显示，只保存）

运行方式：
    python scripts/smoke_metrics_viz.py
"""

import sys
from pathlib import Path
import numpy as np

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置 matplotlib 后端为非交互式
import matplotlib
matplotlib.use('Agg')

from src.common.math3d import deg2rad, rad2deg
from src.truth.scenarios import generate_swing
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.complementary import run_complementary
from src.metrics.tilt_error import compute_tilt_metrics, print_tilt_metrics
from src.viz.plot_timeseries import plot_attitude_comparison, plot_attitude_error


def test_compute_tilt_metrics():
    """测试 1: compute_tilt_metrics 函数"""
    print("=" * 60)
    print("测试 1: compute_tilt_metrics 函数")
    print("=" * 60)
    
    g = GRAVITY_STANDARD
    
    # 生成摆动真值
    truth = generate_swing(
        fs=100.0, duration_s=10.0,
        roll_amp_deg=10.0, pitch_amp_deg=5.0,
        roll_freq_hz=0.2, pitch_freq_hz=0.15,
        roll_phase_deg=0.0, pitch_phase_deg=90.0,
        yaw_deg=0.0, temp_C=25.0, seed=42
    )
    
    # 生成测量
    sensor_params = {
        "acc": {"bias0": [0, 0, 0], "sigma_white": 0.02},
        "gyro": {"bias0": [0, 0, 0], "sigma_white": 0.001},
    }
    meas = forward_imu(truth, sensor_params, seed=42, g=g)
    
    # 运行滤波
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": 100.0},
    }
    est = run_complementary(ds, {"alpha": 0.98})
    
    # 计算指标
    metrics = compute_tilt_metrics(truth, est, skip_samples=100)
    
    all_passed = True
    
    # 检查指标存在
    required_keys = ["rmse_roll", "rmse_pitch", "peak_roll", "peak_pitch",
                     "mean_roll", "mean_pitch", "std_roll", "std_pitch"]
    for key in required_keys:
        if key not in metrics:
            print(f"  ✗ 缺少指标: {key}")
            all_passed = False
        else:
            print(f"  ✓ {key}: {metrics[key]:.4f}°")
    
    # 检查指标合理性
    if metrics["rmse_roll"] < 0:
        print("  ✗ rmse_roll 应该非负")
        all_passed = False
    
    if metrics["peak_roll"] < metrics["rmse_roll"]:
        print("  ✗ peak_roll 应该 >= rmse_roll")
        all_passed = False
    
    return all_passed


def test_visualization():
    """测试 2: 可视化函数"""
    print("\n" + "=" * 60)
    print("测试 2: 可视化函数")
    print("=" * 60)
    
    g = GRAVITY_STANDARD
    
    # 生成摆动真值
    truth = generate_swing(
        fs=100.0, duration_s=5.0,
        roll_amp_deg=10.0, pitch_amp_deg=5.0,
        roll_freq_hz=0.3, pitch_freq_hz=0.2,
        roll_phase_deg=0.0, pitch_phase_deg=90.0,
        yaw_deg=0.0, temp_C=25.0, seed=42
    )
    
    # 生成测量
    sensor_params = {
        "acc": {"bias0": [0, 0, 0], "sigma_white": 0.02},
        "gyro": {"bias0": [0, 0, 0], "sigma_white": 0.001},
    }
    meas = forward_imu(truth, sensor_params, seed=42, g=g)
    
    # 运行滤波
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": 100.0},
    }
    est = run_complementary(ds, {"alpha": 0.98})
    
    all_passed = True
    
    # 测试对比图
    try:
        fig1 = plot_attitude_comparison(
            truth["t"], truth, est,
            save_path="outputs/figures/test_comparison.png",
            title="Test Comparison",
            show=False
        )
        print("  ✓ plot_attitude_comparison 成功")
        print("    保存到: outputs/figures/test_comparison.png")
    except Exception as e:
        print(f"  ✗ plot_attitude_comparison 失败: {e}")
        all_passed = False
    
    # 测试误差图
    try:
        fig2 = plot_attitude_error(
            truth["t"], truth, est,
            save_path="outputs/figures/test_error.png",
            title="Test Error",
            show=False
        )
        print("  ✓ plot_attitude_error 成功")
        print("    保存到: outputs/figures/test_error.png")
    except Exception as e:
        print(f"  ✗ plot_attitude_error 失败: {e}")
        all_passed = False
    
    # 检查文件是否存在
    if Path("outputs/figures/test_comparison.png").exists():
        print("  ✓ test_comparison.png 文件存在")
    else:
        print("  ✗ test_comparison.png 文件不存在")
        all_passed = False
    
    if Path("outputs/figures/test_error.png").exists():
        print("  ✓ test_error.png 文件存在")
    else:
        print("  ✗ test_error.png 文件不存在")
        all_passed = False
    
    return all_passed


def main():
    print("\n" + "=" * 60)
    print("指标与可视化自检脚本")
    print("=" * 60)
    
    results = []
    
    results.append(("compute_tilt_metrics", test_compute_tilt_metrics()))
    results.append(("可视化函数", test_visualization()))
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！指标与可视化模块正常。")
        return 0
    else:
        print("存在测试失败！请检查指标与可视化模块。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

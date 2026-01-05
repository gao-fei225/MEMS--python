#!/usr/bin/env python
"""
互补滤波器自检脚本

验证内容：
1. acc_to_roll_pitch 函数正确性
2. 准静态工况下的姿态估计
3. 摆动工况下的跟踪性能

运行方式：
    python scripts/smoke_complementary.py
"""

import sys
from pathlib import Path
import numpy as np

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.math3d import deg2rad, rad2deg, rpy_to_quat, quat_to_rpy
from src.truth.scenarios import generate_quasi_static, generate_swing
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.complementary import acc_to_roll_pitch, run_complementary


def test_acc_to_roll_pitch():
    """测试 1: acc_to_roll_pitch 函数"""
    print("=" * 60)
    print("测试 1: acc_to_roll_pitch 函数")
    print("=" * 60)
    
    g = GRAVITY_STANDARD
    
    # 测试用例：(roll_deg, pitch_deg, expected_roll, expected_pitch)
    test_cases = [
        (0.0, 0.0),
        (10.0, 0.0),
        (0.0, 10.0),
        (10.0, -5.0),
        (-15.0, 20.0),
    ]
    
    all_passed = True
    
    for roll_deg, pitch_deg in test_cases:
        # 生成准静态真值
        truth = generate_quasi_static(
            fs=100.0, duration_s=0.1,
            roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=0.0,
            temp_C=25.0, seed=42
        )
        
        # 生成无噪声测量
        sensor_params = {
            "acc": {"bias0": [0, 0, 0], "sigma_white": 0.0},
            "gyro": {"bias0": [0, 0, 0], "sigma_white": 0.0},
        }
        meas = forward_imu(truth, sensor_params, seed=42, g=g)
        
        # 计算 roll/pitch
        roll_acc, pitch_acc = acc_to_roll_pitch(meas["acc"], g)
        
        roll_err = abs(rad2deg(roll_acc[0]) - roll_deg)
        pitch_err = abs(rad2deg(pitch_acc[0]) - pitch_deg)
        
        passed = roll_err < 0.1 and pitch_err < 0.1
        status = "✓" if passed else "✗"
        all_passed = all_passed and passed
        
        print(f"  输入: roll={roll_deg:+.1f}°, pitch={pitch_deg:+.1f}°")
        print(f"  输出: roll={rad2deg(roll_acc[0]):+.1f}°, pitch={rad2deg(pitch_acc[0]):+.1f}°")
        print(f"  误差: roll={roll_err:.4f}°, pitch={pitch_err:.4f}° {status}")
    
    return all_passed


def test_quasi_static_estimation():
    """测试 2: 准静态工况姿态估计"""
    print("\n" + "=" * 60)
    print("测试 2: 准静态工况姿态估计")
    print("=" * 60)
    
    g = GRAVITY_STANDARD
    
    # 生成准静态真值
    roll_true = 10.0
    pitch_true = -5.0
    
    truth = generate_quasi_static(
        fs=100.0, duration_s=5.0,
        roll_deg=roll_true, pitch_deg=pitch_true, yaw_deg=0.0,
        temp_C=25.0, seed=42
    )
    
    # 生成带噪声测量
    sensor_params = {
        "acc": {"bias0": [0, 0, 0], "sigma_white": 0.02},
        "gyro": {"bias0": [0, 0, 0], "sigma_white": 0.001},
    }
    meas = forward_imu(truth, sensor_params, seed=42, g=g)
    
    # 构建数据集
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": 100.0},
    }
    
    # 运行互补滤波
    cfg = {"alpha": 0.98}
    est = run_complementary(ds, cfg)
    
    # 计算稳态误差（取后半段）
    n = len(est["roll"])
    roll_mean = np.mean(est["roll"][n//2:])
    pitch_mean = np.mean(est["pitch"][n//2:])
    
    roll_err = abs(rad2deg(roll_mean) - roll_true)
    pitch_err = abs(rad2deg(pitch_mean) - pitch_true)
    
    all_passed = True
    
    passed = roll_err < 1.0
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  Roll: 真值={roll_true:.1f}°, 估计={rad2deg(roll_mean):.2f}°, 误差={roll_err:.4f}° {status}")
    
    passed = pitch_err < 1.0
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  Pitch: 真值={pitch_true:.1f}°, 估计={rad2deg(pitch_mean):.2f}°, 误差={pitch_err:.4f}° {status}")
    
    return all_passed


def test_swing_tracking():
    """测试 3: 摆动工况跟踪"""
    print("\n" + "=" * 60)
    print("测试 3: 摆动工况跟踪")
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
    
    # 生成带噪声测量
    sensor_params = {
        "acc": {"bias0": [0, 0, 0], "sigma_white": 0.02},
        "gyro": {"bias0": [0, 0, 0], "sigma_white": 0.001},
    }
    meas = forward_imu(truth, sensor_params, seed=42, g=g)
    
    # 构建数据集
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": 100.0},
    }
    
    # 运行互补滤波
    cfg = {"alpha": 0.98}
    est = run_complementary(ds, cfg)
    
    # 计算真值 roll/pitch
    roll_true = truth["rpy_deg"][:, 0]
    pitch_true = truth["rpy_deg"][:, 1]
    
    # 计算 RMSE（跳过初始收敛段）
    n = len(est["roll"])
    start = n // 5  # 跳过前 20%
    
    roll_rmse = np.sqrt(np.mean((rad2deg(est["roll"][start:]) - roll_true[start:])**2))
    pitch_rmse = np.sqrt(np.mean((rad2deg(est["pitch"][start:]) - pitch_true[start:])**2))
    
    all_passed = True
    
    passed = roll_rmse < 2.0
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  Roll RMSE: {roll_rmse:.4f}° {status}")
    
    passed = pitch_rmse < 2.0
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  Pitch RMSE: {pitch_rmse:.4f}° {status}")
    
    # 检查跟踪范围
    roll_range = np.max(rad2deg(est["roll"])) - np.min(rad2deg(est["roll"]))
    pitch_range = np.max(rad2deg(est["pitch"])) - np.min(rad2deg(est["pitch"]))
    
    print(f"  Roll 范围: {roll_range:.2f}° (真值范围 ~20°)")
    print(f"  Pitch 范围: {pitch_range:.2f}° (真值范围 ~10°)")
    
    return all_passed


def main():
    print("\n" + "=" * 60)
    print("互补滤波器自检脚本")
    print("=" * 60)
    
    results = []
    
    results.append(("acc_to_roll_pitch", test_acc_to_roll_pitch()))
    results.append(("准静态估计", test_quasi_static_estimation()))
    results.append(("摆动跟踪", test_swing_tracking()))
    
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
        print("所有测试通过！互补滤波器正常。")
        return 0
    else:
        print("存在测试失败！请检查互补滤波器。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

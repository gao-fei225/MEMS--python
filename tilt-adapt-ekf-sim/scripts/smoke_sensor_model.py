#!/usr/bin/env python
"""
IMU 传感器模型自检脚本

验证内容：
1. 准静态工况下的加速度计测量
2. 白噪声统计特性
3. 偏置注入

运行方式：
    python scripts/smoke_sensor_model.py
"""

import sys
from pathlib import Path
import numpy as np

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.math3d import deg2rad, rad2deg
from src.truth.scenarios import generate_quasi_static
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu


def test_quasi_static_acc():
    """测试 1: 准静态工况加速度计测量"""
    print("=" * 60)
    print("测试 1: 准静态工况加速度计测量")
    print("=" * 60)
    
    g = GRAVITY_STANDARD
    
    # 无偏置、无噪声
    sensor_params = {
        "acc": {"bias0": [0, 0, 0], "sigma_white": 0.0},
        "gyro": {"bias0": [0, 0, 0], "sigma_white": 0.0},
    }
    
    all_passed = True
    
    # 测试用例：(roll, pitch, yaw, expected_acc_z)
    # 注意：加速度计测量的是比力 (specific force)
    # 静止时 a_meas = -R_bn @ g_n
    # 水平静止时 a_meas = [0, 0, +g]（指向天空）
    test_cases = [
        (0.0, 0.0, 0.0, g),      # 水平静止，acc_z = +g
        (0.0, 90.0, 0.0, 0.0),   # 抬头 90°，acc_z ≈ 0
        (90.0, 0.0, 0.0, 0.0),   # 左翼下沉 90°，acc_z ≈ 0
    ]
    
    for roll_deg, pitch_deg, yaw_deg, expected_z in test_cases:
        truth = generate_quasi_static(
            fs=100.0, duration_s=0.1,
            roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
            temp_C=25.0, seed=42
        )
        
        meas = forward_imu(truth, sensor_params, seed=42, g=g)
        
        acc_mean = np.mean(meas["acc"], axis=0)
        acc_z = acc_mean[2]
        
        err = abs(acc_z - expected_z)
        passed = err < 0.01
        status = "✓" if passed else "✗"
        all_passed = all_passed and passed
        
        print(f"  姿态: roll={roll_deg:+.0f}°, pitch={pitch_deg:+.0f}°, yaw={yaw_deg:+.0f}°")
        print(f"    acc_mean = [{acc_mean[0]:+.4f}, {acc_mean[1]:+.4f}, {acc_mean[2]:+.4f}]")
        print(f"    acc_z 期望: {expected_z:.4f}, 实际: {acc_z:.4f}, 误差: {err:.4f} {status}")
    
    return all_passed


def test_specific_force_sign_convention():
    """测试 1b: 比力符号约定一致性"""
    print("\n" + "=" * 60)
    print("测试 1b: 比力符号约定一致性")
    print("=" * 60)
    
    from src.common.math3d import quat_to_R_bn
    from src.truth.frames import gravity_n
    
    g = GRAVITY_STANDARD
    g_n = gravity_n(g)  # [0, 0, +g]
    
    sensor_params = {
        "acc": {"bias0": [0, 0, 0], "sigma_white": 0.0},
        "gyro": {"bias0": [0, 0, 0], "sigma_white": 0.0},
    }
    
    all_passed = True
    
    # 测试多个姿态
    test_cases = [
        (0.0, 0.0, 0.0),
        (10.0, -5.0, 0.0),
        (30.0, 20.0, 45.0),
    ]
    
    print("  符号约定：")
    print("    g_n = [0, 0, +g] (NED, 重力指向地心)")
    print("    acc_meas = R_bn @ (a_lin_n + g_n)")
    print("    静止水平时 acc_meas = [0, 0, +g] (比力指向天空)")
    print()
    
    for roll_deg, pitch_deg, yaw_deg in test_cases:
        truth = generate_quasi_static(
            fs=100.0, duration_s=0.1,
            roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
            temp_C=25.0, seed=42
        )
        
        meas = forward_imu(truth, sensor_params, seed=42, g=g)
        
        # 计算期望值：acc_meas = R_bn @ g_n（当前实现）
        # 这表示"支撑力"或"比力"，静止水平时指向天空
        q_nb = truth["q_nb"][0]
        R_bn = quat_to_R_bn(q_nb)
        expected_acc = R_bn @ g_n  # 当前实现的约定
        
        acc_mean = np.mean(meas["acc"], axis=0)
        
        err = np.linalg.norm(acc_mean - expected_acc)
        passed = err < 1e-10
        status = "✓" if passed else "✗"
        all_passed = all_passed and passed
        
        print(f"  姿态: roll={roll_deg:+.1f}°, pitch={pitch_deg:+.1f}°, yaw={yaw_deg:+.1f}°")
        print(f"    期望 (R_bn @ g_n): [{expected_acc[0]:+.4f}, {expected_acc[1]:+.4f}, {expected_acc[2]:+.4f}]")
        print(f"    实际:              [{acc_mean[0]:+.4f}, {acc_mean[1]:+.4f}, {acc_mean[2]:+.4f}]")
        print(f"    误差: {err:.2e} {status}")
    
    return all_passed


def test_white_noise_statistics():
    """测试 2: 白噪声统计特性"""
    print("\n" + "=" * 60)
    print("测试 2: 白噪声统计特性")
    print("=" * 60)
    
    g = GRAVITY_STANDARD
    sigma_acc = 0.1
    sigma_gyro = 0.01
    
    sensor_params = {
        "acc": {"bias0": [0, 0, 0], "sigma_white": sigma_acc},
        "gyro": {"bias0": [0, 0, 0], "sigma_white": sigma_gyro},
    }
    
    # 生成较长序列以获得稳定统计
    truth = generate_quasi_static(
        fs=100.0, duration_s=10.0,
        roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
        temp_C=25.0, seed=42
    )
    
    meas = forward_imu(truth, sensor_params, seed=42, g=g)
    
    all_passed = True
    
    # 计算加速度计噪声（去除均值后）
    acc_noise = meas["acc"] - np.mean(meas["acc"], axis=0)
    acc_std = np.std(acc_noise)
    
    # 检查标准差是否接近配置值
    acc_std_err = abs(acc_std - sigma_acc) / sigma_acc
    passed = acc_std_err < 0.1  # 允许 10% 误差
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  加速度计噪声 std: 期望 {sigma_acc:.4f}, 实际 {acc_std:.4f}, 相对误差 {acc_std_err*100:.1f}% {status}")
    
    # 计算陀螺仪噪声
    gyro_noise = meas["gyro"] - np.mean(meas["gyro"], axis=0)
    gyro_std = np.std(gyro_noise)
    
    gyro_std_err = abs(gyro_std - sigma_gyro) / sigma_gyro
    passed = gyro_std_err < 0.1
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  陀螺仪噪声 std: 期望 {sigma_gyro:.4f}, 实际 {gyro_std:.4f}, 相对误差 {gyro_std_err*100:.1f}% {status}")
    
    # 检查均值是否接近零（噪声应该是零均值）
    acc_noise_mean = np.mean(acc_noise)
    gyro_noise_mean = np.mean(gyro_noise)
    
    passed = abs(acc_noise_mean) < 0.01
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  加速度计噪声均值: {acc_noise_mean:.6f} {status}")
    
    passed = abs(gyro_noise_mean) < 0.001
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  陀螺仪噪声均值: {gyro_noise_mean:.6f} {status}")
    
    return all_passed


def test_bias_injection():
    """测试 3: 偏置注入"""
    print("\n" + "=" * 60)
    print("测试 3: 偏置注入")
    print("=" * 60)
    
    g = GRAVITY_STANDARD
    
    acc_bias = [0.1, -0.05, 0.02]
    gyro_bias = [0.001, -0.002, 0.0005]
    
    sensor_params = {
        "acc": {"bias0": acc_bias, "sigma_white": 0.0},
        "gyro": {"bias0": gyro_bias, "sigma_white": 0.0},
    }
    
    truth = generate_quasi_static(
        fs=100.0, duration_s=1.0,
        roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
        temp_C=25.0, seed=42
    )
    
    meas = forward_imu(truth, sensor_params, seed=42, g=g)
    
    all_passed = True
    
    # 检查加速度计偏置
    # 水平静止时，理想测量为 [0, 0, g]
    # 加偏置后应为 [0+0.1, 0-0.05, g+0.02]
    acc_mean = np.mean(meas["acc"], axis=0)
    expected_acc = np.array([0.0 + acc_bias[0], 0.0 + acc_bias[1], g + acc_bias[2]])
    
    acc_err = np.linalg.norm(acc_mean - expected_acc)
    passed = acc_err < 1e-10
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  加速度计测量: 期望 [{expected_acc[0]:.4f}, {expected_acc[1]:.4f}, {expected_acc[2]:.4f}]")
    print(f"                实际 [{acc_mean[0]:.4f}, {acc_mean[1]:.4f}, {acc_mean[2]:.4f}]")
    print(f"                误差: {acc_err:.2e} {status}")
    
    # 检查陀螺仪偏置
    # 静止时，理想测量为 [0, 0, 0]
    # 加偏置后应为 gyro_bias
    gyro_mean = np.mean(meas["gyro"], axis=0)
    expected_gyro = np.array(gyro_bias)
    
    gyro_err = np.linalg.norm(gyro_mean - expected_gyro)
    passed = gyro_err < 1e-10
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  陀螺仪测量: 期望 [{expected_gyro[0]:.6f}, {expected_gyro[1]:.6f}, {expected_gyro[2]:.6f}]")
    print(f"              实际 [{gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}]")
    print(f"              误差: {gyro_err:.2e} {status}")
    
    # 检查真实偏置记录
    acc_bias_true = meas["acc_bias_true"]
    gyro_bias_true = meas["gyro_bias_true"]
    
    passed = np.allclose(acc_bias_true[0], acc_bias)
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  acc_bias_true 记录正确: {status}")
    
    passed = np.allclose(gyro_bias_true[0], gyro_bias)
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  gyro_bias_true 记录正确: {status}")
    
    return all_passed


def test_swing_scenario():
    """测试 4: 摆动工况"""
    print("\n" + "=" * 60)
    print("测试 4: 摆动工况")
    print("=" * 60)
    
    from src.truth.scenarios import generate_swing
    
    g = GRAVITY_STANDARD
    
    sensor_params = {
        "acc": {"bias0": [0, 0, 0], "sigma_white": 0.01},
        "gyro": {"bias0": [0, 0, 0], "sigma_white": 0.001},
    }
    
    truth = generate_swing(
        fs=100.0, duration_s=2.0,
        roll_amp_deg=10.0, pitch_amp_deg=5.0,
        roll_freq_hz=0.5, pitch_freq_hz=0.3,
        roll_phase_deg=0.0, pitch_phase_deg=90.0,
        yaw_deg=0.0, temp_C=25.0, seed=42
    )
    
    meas = forward_imu(truth, sensor_params, seed=42, g=g)
    
    all_passed = True
    
    # 检查输出形状
    n_samples = len(truth["t"])
    
    passed = meas["acc"].shape == (n_samples, 3)
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  acc shape: {meas['acc'].shape} {status}")
    
    passed = meas["gyro"].shape == (n_samples, 3)
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  gyro shape: {meas['gyro'].shape} {status}")
    
    # 检查加速度计测量范围（应该在 g 附近）
    acc_norm = np.linalg.norm(meas["acc"], axis=1)
    acc_norm_mean = np.mean(acc_norm)
    
    passed = abs(acc_norm_mean - g) < 0.5
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  acc 模长均值: {acc_norm_mean:.4f} (期望 ~{g:.4f}) {status}")
    
    # 检查陀螺仪测量范围
    gyro_max = np.max(np.abs(meas["gyro"]))
    expected_gyro_max = deg2rad(10.0) * 2 * np.pi * 0.5  # roll_amp * 2*pi*freq
    
    passed = abs(gyro_max - expected_gyro_max) < 0.1
    status = "✓" if passed else "✗"
    all_passed = all_passed and passed
    print(f"  gyro 最大值: {rad2deg(gyro_max):.2f} deg/s (期望 ~{rad2deg(expected_gyro_max):.2f} deg/s) {status}")
    
    return all_passed


def main():
    print("\n" + "=" * 60)
    print("IMU 传感器模型自检脚本")
    print("=" * 60)
    
    results = []
    
    results.append(("准静态加速度计", test_quasi_static_acc()))
    results.append(("比力符号约定", test_specific_force_sign_convention()))
    results.append(("白噪声统计", test_white_noise_statistics()))
    results.append(("偏置注入", test_bias_injection()))
    results.append(("摆动工况", test_swing_scenario()))
    
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
        print("所有测试通过！IMU 传感器模型正常。")
        return 0
    else:
        print("存在测试失败！请检查传感器模型。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
真值生成 MVP 自检脚本

验证内容：
1. math3d 函数正确性
2. 准静态工况生成
3. 摆动工况生成
4. 重力投影

运行方式：
    python scripts/smoke_truth_mvp.py
"""

import sys
from pathlib import Path
import numpy as np

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.math3d import (
    rpy_to_quat, quat_to_rpy, quat_to_R_nb, quat_to_R_bn,
    deg2rad, rad2deg, quat_normalize
)
from src.truth.frames import gravity_n, gravity_b, accel_measurement_static, GRAVITY_STANDARD
from src.truth.scenarios import generate_quasi_static, generate_swing


def test_math3d_roundtrip():
    """测试 1: math3d 欧拉角-四元数 round-trip"""
    print("=" * 60)
    print("测试 1: math3d 欧拉角-四元数 Round-Trip")
    print("=" * 60)
    
    test_cases = [
        (10.0, -5.0, 0.0),
        (0.0, 0.0, 0.0),
        (30.0, 20.0, 45.0),
        (-15.0, 10.0, -60.0),
    ]
    
    all_passed = True
    
    for roll_deg, pitch_deg, yaw_deg in test_cases:
        roll = deg2rad(roll_deg)
        pitch = deg2rad(pitch_deg)
        yaw = deg2rad(yaw_deg)
        
        q = rpy_to_quat(roll, pitch, yaw)
        roll_out, pitch_out, yaw_out = quat_to_rpy(q)
        
        roll_err = abs(rad2deg(roll_out - roll))
        pitch_err = abs(rad2deg(pitch_out - pitch))
        yaw_err = abs(rad2deg(yaw_out - yaw))
        
        tol = 1e-10
        passed = roll_err < tol and pitch_err < tol and yaw_err < tol
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"  ({roll_deg:+.1f}°, {pitch_deg:+.1f}°, {yaw_deg:+.1f}°) -> 误差: ({roll_err:.2e}°, {pitch_err:.2e}°, {yaw_err:.2e}°) {status}")
    
    return all_passed


def test_gravity_projection():
    """测试 2: 重力投影"""
    print("\n" + "=" * 60)
    print("测试 2: 重力投影")
    print("=" * 60)
    
    g = GRAVITY_STANDARD
    
    # 水平静止
    # 重力 g_n = [0, 0, g]，加速度计测量 a = -R_bn @ g_n = -g_b
    q = rpy_to_quat(0, 0, 0)
    a_meas = accel_measurement_static(q, g)
    expected = np.array([0, 0, -g])
    err = np.linalg.norm(a_meas - expected)
    passed = err < 1e-10
    print(f"  水平静止: a_meas = [{a_meas[0]:+.4f}, {a_meas[1]:+.4f}, {a_meas[2]:+.4f}], 误差: {err:.2e} {'✓' if passed else '✗'}")
    
    all_passed = passed
    
    # 抬头 90° (pitch = +90°)
    # 机体 X 轴指向天，Z 轴指向北
    # g_b = R_bn @ g_n，R_bn 将导航系 Z 轴（重力方向）映射到机体系 -X 轴
    # 所以 a_meas = -g_b = [+g, 0, 0]
    q = rpy_to_quat(0, deg2rad(90), 0)
    a_meas = accel_measurement_static(q, g)
    expected = np.array([g, 0, 0])  # 抬头时，重力在机体 -X 方向，比力在 +X 方向
    err = np.linalg.norm(a_meas - expected)
    passed = err < 1e-6
    print(f"  抬头 90°: a_meas = [{a_meas[0]:+.4f}, {a_meas[1]:+.4f}, {a_meas[2]:+.4f}], 期望: [{expected[0]:+.4f}, {expected[1]:+.4f}, {expected[2]:+.4f}], 误差: {err:.2e} {'✓' if passed else '✗'}")
    all_passed = all_passed and passed
    
    # 左翼下沉 90° (roll = +90°)
    # 机体 Y 轴指向地，Z 轴指向东
    # g_b = R_bn @ g_n，重力在机体 +Y 方向
    # 所以 a_meas = -g_b = [0, -g, 0]
    q = rpy_to_quat(deg2rad(90), 0, 0)
    a_meas = accel_measurement_static(q, g)
    expected = np.array([0, -g, 0])  # 左翼下沉时，重力在机体 +Y 方向，比力在 -Y 方向
    err = np.linalg.norm(a_meas - expected)
    passed = err < 1e-6
    print(f"  左翼下沉 90°: a_meas = [{a_meas[0]:+.4f}, {a_meas[1]:+.4f}, {a_meas[2]:+.4f}], 期望: [{expected[0]:+.4f}, {expected[1]:+.4f}, {expected[2]:+.4f}], 误差: {err:.2e} {'✓' if passed else '✗'}")
    all_passed = all_passed and passed
    
    return all_passed


def test_quasi_static():
    """测试 3: 准静态工况生成"""
    print("\n" + "=" * 60)
    print("测试 3: 准静态工况生成")
    print("=" * 60)
    
    truth = generate_quasi_static(
        fs=100.0,
        duration_s=1.0,
        roll_deg=10.0,
        pitch_deg=-5.0,
        yaw_deg=0.0,
        temp_C=25.0,
        seed=42
    )
    
    all_passed = True
    
    # 检查字段存在
    required_fields = ["t", "q_nb", "omega_b", "a_lin_n", "temp"]
    for field in required_fields:
        if field not in truth:
            print(f"  ✗ 缺少字段: {field}")
            all_passed = False
        else:
            print(f"  ✓ 字段 {field}: shape = {truth[field].shape}")
    
    # 检查样本数
    n_expected = 101  # ceil(100 * 1.0) + 1
    n_actual = len(truth["t"])
    if n_actual == n_expected:
        print(f"  ✓ 样本数正确: {n_actual}")
    else:
        print(f"  ✗ 样本数错误: 期望 {n_expected}, 实际 {n_actual}")
        all_passed = False
    
    # 检查四元数是单位四元数
    q_norms = np.linalg.norm(truth["q_nb"], axis=1)
    if np.allclose(q_norms, 1.0, atol=1e-10):
        print(f"  ✓ 四元数是单位四元数")
    else:
        print(f"  ✗ 四元数不是单位四元数，最大误差: {np.max(np.abs(q_norms - 1.0))}")
        all_passed = False
    
    # 检查四元数是常值
    q_diff = np.max(np.abs(truth["q_nb"] - truth["q_nb"][0]))
    if q_diff < 1e-10:
        print(f"  ✓ 四元数是常值")
    else:
        print(f"  ✗ 四元数不是常值，最大变化: {q_diff}")
        all_passed = False
    
    # 检查角速度为零
    omega_max = np.max(np.abs(truth["omega_b"]))
    if omega_max < 1e-10:
        print(f"  ✓ 角速度为零")
    else:
        print(f"  ✗ 角速度不为零，最大值: {omega_max}")
        all_passed = False
    
    # 检查姿态正确
    roll_out, pitch_out, yaw_out = quat_to_rpy(truth["q_nb"][0])
    roll_err = abs(rad2deg(roll_out) - 10.0)
    pitch_err = abs(rad2deg(pitch_out) - (-5.0))
    yaw_err = abs(rad2deg(yaw_out) - 0.0)
    if roll_err < 1e-10 and pitch_err < 1e-10 and yaw_err < 1e-10:
        print(f"  ✓ 姿态正确: roll={rad2deg(roll_out):.2f}°, pitch={rad2deg(pitch_out):.2f}°, yaw={rad2deg(yaw_out):.2f}°")
    else:
        print(f"  ✗ 姿态错误: roll={rad2deg(roll_out):.2f}°, pitch={rad2deg(pitch_out):.2f}°, yaw={rad2deg(yaw_out):.2f}°")
        all_passed = False
    
    return all_passed


def test_swing():
    """测试 4: 摆动工况生成"""
    print("\n" + "=" * 60)
    print("测试 4: 摆动工况生成")
    print("=" * 60)
    
    truth = generate_swing(
        fs=100.0,
        duration_s=2.0,
        roll_amp_deg=10.0,
        pitch_amp_deg=5.0,
        roll_freq_hz=0.5,
        pitch_freq_hz=0.3,
        roll_phase_deg=0.0,
        pitch_phase_deg=90.0,
        yaw_deg=0.0,
        temp_C=25.0,
        seed=42
    )
    
    all_passed = True
    
    # 检查字段存在
    required_fields = ["t", "q_nb", "omega_b", "a_lin_n", "temp"]
    for field in required_fields:
        if field not in truth:
            print(f"  ✗ 缺少字段: {field}")
            all_passed = False
        else:
            print(f"  ✓ 字段 {field}: shape = {truth[field].shape}")
    
    # 检查四元数是单位四元数
    q_norms = np.linalg.norm(truth["q_nb"], axis=1)
    if np.allclose(q_norms, 1.0, atol=1e-10):
        print(f"  ✓ 四元数是单位四元数")
    else:
        print(f"  ✗ 四元数不是单位四元数，最大误差: {np.max(np.abs(q_norms - 1.0))}")
        all_passed = False
    
    # 检查姿态变化范围
    rolls = []
    pitches = []
    for i in range(len(truth["t"])):
        r, p, y = quat_to_rpy(truth["q_nb"][i])
        rolls.append(rad2deg(r))
        pitches.append(rad2deg(p))
    
    roll_range = max(rolls) - min(rolls)
    pitch_range = max(pitches) - min(pitches)
    
    # 期望范围约为 2 * amplitude
    roll_range_expected = 2 * 10.0
    pitch_range_expected = 2 * 5.0
    
    if abs(roll_range - roll_range_expected) < 1.0:
        print(f"  ✓ Roll 范围正确: {roll_range:.2f}° (期望 ~{roll_range_expected}°)")
    else:
        print(f"  ✗ Roll 范围错误: {roll_range:.2f}° (期望 ~{roll_range_expected}°)")
        all_passed = False
    
    if abs(pitch_range - pitch_range_expected) < 1.0:
        print(f"  ✓ Pitch 范围正确: {pitch_range:.2f}° (期望 ~{pitch_range_expected}°)")
    else:
        print(f"  ✗ Pitch 范围错误: {pitch_range:.2f}° (期望 ~{pitch_range_expected}°)")
        all_passed = False
    
    # 检查角速度不为零
    omega_max = np.max(np.abs(truth["omega_b"]))
    if omega_max > 0.01:
        print(f"  ✓ 角速度非零，最大值: {rad2deg(omega_max):.2f} deg/s")
    else:
        print(f"  ✗ 角速度应该非零")
        all_passed = False
    
    return all_passed


def main():
    print("\n" + "=" * 60)
    print("真值生成 MVP 自检脚本")
    print("=" * 60)
    
    results = []
    
    results.append(("math3d Round-Trip", test_math3d_roundtrip()))
    results.append(("重力投影", test_gravity_projection()))
    results.append(("准静态工况", test_quasi_static()))
    results.append(("摆动工况", test_swing()))
    
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
        print("所有测试通过！真值生成 MVP 正常。")
        return 0
    else:
        print("存在测试失败！请检查真值生成模块。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
坐标系约定自检脚本

验证内容：
1. 欧拉角 → 四元数 → 欧拉角 round-trip
2. 四元数 → 旋转矩阵 → 四元数 round-trip
3. 重力向量在不同姿态下的投影
4. 符号正确性检验

运行方式：
    python scripts/smoke_frames.py
"""

import sys
import numpy as np
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    欧拉角转四元数 (ZYX 顺序)
    
    Args:
        roll: 横滚角 (rad)
        pitch: 俯仰角 (rad)
        yaw: 偏航角 (rad)
    
    Returns:
        q = [w, x, y, z]
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def quat_to_euler(q: np.ndarray) -> tuple:
    """
    四元数转欧拉角 (ZYX 顺序)
    
    Args:
        q: [w, x, y, z]
    
    Returns:
        (roll, pitch, yaw) in radians
    """
    w, x, y, z = q
    
    # Roll (φ)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (θ)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.arcsin(sinp)
    
    # Yaw (ψ)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return (roll, pitch, yaw)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    四元数转旋转矩阵 R_nb
    
    Args:
        q: [w, x, y, z]
    
    Returns:
        R_nb (3x3)
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵转四元数
    
    Args:
        R: 3x3 旋转矩阵
    
    Returns:
        q = [w, x, y, z]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    # 确保 w > 0（四元数符号一致性）
    if w < 0:
        q = -q
    return q


def deg2rad(deg: float) -> float:
    return deg * np.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / np.pi


def test_euler_quat_roundtrip():
    """测试 1: 欧拉角 → 四元数 → 欧拉角 round-trip"""
    print("=" * 60)
    print("测试 1: 欧拉角 → 四元数 → 欧拉角 Round-Trip")
    print("=" * 60)
    
    # 测试用例：roll=10°, pitch=-5°, yaw=0°
    test_cases = [
        (10.0, -5.0, 0.0),    # 指定测试用例
        (0.0, 0.0, 0.0),      # 零姿态
        (30.0, 0.0, 0.0),     # 纯 roll
        (0.0, 20.0, 0.0),     # 纯 pitch
        (0.0, 0.0, 45.0),     # 纯 yaw
        (-15.0, 10.0, 30.0),  # 混合姿态
    ]
    
    all_passed = True
    
    for roll_deg, pitch_deg, yaw_deg in test_cases:
        # 转换为弧度
        roll = deg2rad(roll_deg)
        pitch = deg2rad(pitch_deg)
        yaw = deg2rad(yaw_deg)
        
        # 欧拉角 → 四元数
        q = euler_to_quat(roll, pitch, yaw)
        
        # 四元数 → 欧拉角
        roll_out, pitch_out, yaw_out = quat_to_euler(q)
        
        # 计算误差
        roll_err = rad2deg(roll_out - roll)
        pitch_err = rad2deg(pitch_out - pitch)
        yaw_err = rad2deg(yaw_out - yaw)
        
        # 检查误差
        tol = 1e-10  # 度
        passed = (abs(roll_err) < tol and abs(pitch_err) < tol and abs(yaw_err) < tol)
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"\n输入: roll={roll_deg:+7.2f}°, pitch={pitch_deg:+7.2f}°, yaw={yaw_deg:+7.2f}°")
        print(f"四元数: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
        print(f"输出: roll={rad2deg(roll_out):+7.2f}°, pitch={rad2deg(pitch_out):+7.2f}°, yaw={rad2deg(yaw_out):+7.2f}°")
        print(f"误差: roll={roll_err:+.2e}°, pitch={pitch_err:+.2e}°, yaw={yaw_err:+.2e}°")
        print(f"状态: {status}")
    
    return all_passed


def test_quat_rotation_matrix_roundtrip():
    """测试 2: 四元数 → 旋转矩阵 → 四元数 round-trip"""
    print("\n" + "=" * 60)
    print("测试 2: 四元数 → 旋转矩阵 → 四元数 Round-Trip")
    print("=" * 60)
    
    test_cases = [
        (10.0, -5.0, 0.0),
        (0.0, 0.0, 0.0),
        (45.0, 30.0, -60.0),
    ]
    
    all_passed = True
    
    for roll_deg, pitch_deg, yaw_deg in test_cases:
        roll = deg2rad(roll_deg)
        pitch = deg2rad(pitch_deg)
        yaw = deg2rad(yaw_deg)
        
        # 欧拉角 → 四元数
        q_in = euler_to_quat(roll, pitch, yaw)
        
        # 四元数 → 旋转矩阵
        R = quat_to_rotation_matrix(q_in)
        
        # 旋转矩阵 → 四元数
        q_out = rotation_matrix_to_quat(R)
        
        # 计算误差（考虑四元数符号）
        if np.dot(q_in, q_out) < 0:
            q_out = -q_out
        q_err = np.linalg.norm(q_in - q_out)
        
        # 检查旋转矩阵正交性
        R_orth_err = np.linalg.norm(R @ R.T - np.eye(3))
        det_err = abs(np.linalg.det(R) - 1.0)
        
        tol = 1e-10
        passed = (q_err < tol and R_orth_err < tol and det_err < tol)
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"\n输入姿态: roll={roll_deg:+.1f}°, pitch={pitch_deg:+.1f}°, yaw={yaw_deg:+.1f}°")
        print(f"四元数误差: {q_err:.2e}")
        print(f"R 正交性误差: {R_orth_err:.2e}")
        print(f"det(R) 误差: {det_err:.2e}")
        print(f"状态: {status}")
    
    return all_passed


def test_gravity_projection():
    """测试 3: 重力向量在不同姿态下的投影"""
    print("\n" + "=" * 60)
    print("测试 3: 重力向量投影")
    print("=" * 60)
    
    g = 9.80665
    g_n = np.array([0.0, 0.0, g])  # NED 坐标系重力
    
    test_cases = [
        # (roll, pitch, yaw, expected_a_b)
        (0.0, 0.0, 0.0, np.array([0.0, 0.0, -g])),  # 水平静止
        (0.0, 90.0, 0.0, np.array([g, 0.0, 0.0])),  # 垂直向上（抬头90°）
        (0.0, -90.0, 0.0, np.array([-g, 0.0, 0.0])),  # 垂直向下（低头90°）
        (90.0, 0.0, 0.0, np.array([0.0, -g, 0.0])),  # 左翼下沉90°（roll正方向）
        (-90.0, 0.0, 0.0, np.array([0.0, g, 0.0])),  # 右翼下沉90°
    ]
    
    all_passed = True
    
    for roll_deg, pitch_deg, yaw_deg, expected in test_cases:
        roll = deg2rad(roll_deg)
        pitch = deg2rad(pitch_deg)
        yaw = deg2rad(yaw_deg)
        
        q = euler_to_quat(roll, pitch, yaw)
        R_nb = quat_to_rotation_matrix(q)
        R_bn = R_nb.T
        
        # 加速度计测量 = -R_bn @ g_n（静止时）
        a_b = -R_bn @ g_n
        
        err = np.linalg.norm(a_b - expected)
        tol = 1e-6
        passed = err < tol
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"\n姿态: roll={roll_deg:+.0f}°, pitch={pitch_deg:+.0f}°, yaw={yaw_deg:+.0f}°")
        print(f"期望 a_b: [{expected[0]:+.4f}, {expected[1]:+.4f}, {expected[2]:+.4f}]")
        print(f"实际 a_b: [{a_b[0]:+.4f}, {a_b[1]:+.4f}, {a_b[2]:+.4f}]")
        print(f"误差: {err:.2e}")
        print(f"状态: {status}")
    
    return all_passed


def test_sign_convention():
    """测试 4: 符号约定正确性"""
    print("\n" + "=" * 60)
    print("测试 4: 符号约定正确性")
    print("=" * 60)
    
    all_passed = True
    
    # 测试 roll 正方向（左翼下沉，绕X轴右手定则）
    roll_deg = 10.0
    q = euler_to_quat(deg2rad(roll_deg), 0, 0)
    R_nb = quat_to_rotation_matrix(q)
    R_bn = R_nb.T
    g_n = np.array([0.0, 0.0, 9.80665])
    a_b = -R_bn @ g_n
    
    # 左翼下沉时，Y 轴应该有负的加速度分量（重力在Y负方向有分量）
    roll_sign_correct = a_b[1] < 0
    status = "✓ PASS" if roll_sign_correct else "✗ FAIL"
    all_passed = all_passed and roll_sign_correct
    print(f"\nRoll = +{roll_deg}° (左翼下沉，绕X轴右手定则)")
    print(f"a_b = [{a_b[0]:+.4f}, {a_b[1]:+.4f}, {a_b[2]:+.4f}]")
    print(f"Y 分量应为负: {status}")
    
    # 测试 pitch 正方向（抬头，绕Y轴右手定则）
    pitch_deg = 10.0
    q = euler_to_quat(0, deg2rad(pitch_deg), 0)
    R_nb = quat_to_rotation_matrix(q)
    R_bn = R_nb.T
    a_b = -R_bn @ g_n
    
    # 抬头时，X 轴应该有正的加速度分量
    pitch_sign_correct = a_b[0] > 0
    status = "✓ PASS" if pitch_sign_correct else "✗ FAIL"
    all_passed = all_passed and pitch_sign_correct
    print(f"\nPitch = +{pitch_deg}° (抬头，绕Y轴右手定则)")
    print(f"a_b = [{a_b[0]:+.4f}, {a_b[1]:+.4f}, {a_b[2]:+.4f}]")
    print(f"X 分量应为正: {status}")
    
    return all_passed


def test_quaternion_sign_equivalence():
    """测试 5: 四元数符号等价性 (q 和 -q 表示同一旋转)"""
    print("\n" + "=" * 60)
    print("测试 5: 四元数符号等价性")
    print("=" * 60)
    
    all_passed = True
    
    test_cases = [
        (10.0, -5.0, 0.0),
        (30.0, 20.0, 45.0),
        (-15.0, 10.0, -60.0),
        (0.0, 0.0, 90.0),
    ]
    
    for roll_deg, pitch_deg, yaw_deg in test_cases:
        roll = deg2rad(roll_deg)
        pitch = deg2rad(pitch_deg)
        yaw = deg2rad(yaw_deg)
        
        q = euler_to_quat(roll, pitch, yaw)
        q_neg = -q  # 符号取反
        
        # 两个四元数应该产生相同的旋转矩阵
        R_q = quat_to_rotation_matrix(q)
        R_q_neg = quat_to_rotation_matrix(q_neg)
        
        R_diff = np.linalg.norm(R_q - R_q_neg)
        
        tol = 1e-10
        passed = R_diff < tol
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"\n姿态: roll={roll_deg:+.1f}°, pitch={pitch_deg:+.1f}°, yaw={yaw_deg:+.1f}°")
        print(f"q     = [{q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f}]")
        print(f"-q    = [{q_neg[0]:+.6f}, {q_neg[1]:+.6f}, {q_neg[2]:+.6f}, {q_neg[3]:+.6f}]")
        print(f"R(q) - R(-q) 范数: {R_diff:.2e}")
        print(f"状态: {status}")
    
    return all_passed


def test_gimbal_lock_handling():
    """测试 6: 万向锁（Gimbal Lock）处理 - pitch 接近 ±90°"""
    print("\n" + "=" * 60)
    print("测试 6: 万向锁处理 (pitch ≈ ±90°)")
    print("=" * 60)
    
    all_passed = True
    
    # 测试接近奇异点的情况
    test_cases = [
        # (roll, pitch, yaw) - pitch 接近 ±90°
        (0.0, 89.0, 0.0),
        (0.0, 89.9, 0.0),
        (0.0, 89.99, 0.0),
        (0.0, -89.0, 0.0),
        (0.0, -89.9, 0.0),
        (30.0, 89.5, 45.0),  # 混合姿态接近奇异点
    ]
    
    for roll_deg, pitch_deg, yaw_deg in test_cases:
        roll = deg2rad(roll_deg)
        pitch = deg2rad(pitch_deg)
        yaw = deg2rad(yaw_deg)
        
        # 欧拉角 → 四元数 → 欧拉角
        q = euler_to_quat(roll, pitch, yaw)
        roll_out, pitch_out, yaw_out = quat_to_euler(q)
        
        # 检查是否有 NaN
        has_nan = np.isnan(roll_out) or np.isnan(pitch_out) or np.isnan(yaw_out)
        
        # 检查 pitch 误差（pitch 应该准确）
        pitch_err = abs(rad2deg(pitch_out - pitch))
        
        # 注意：在万向锁附近，roll 和 yaw 会耦合，但 pitch 应该准确
        # 旋转矩阵应该一致
        R_in = quat_to_rotation_matrix(euler_to_quat(roll, pitch, yaw))
        R_out = quat_to_rotation_matrix(euler_to_quat(roll_out, pitch_out, yaw_out))
        R_diff = np.linalg.norm(R_in - R_out)
        
        tol = 1e-6
        passed = (not has_nan) and (R_diff < tol)
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"\n输入: roll={roll_deg:+.2f}°, pitch={pitch_deg:+.2f}°, yaw={yaw_deg:+.2f}°")
        print(f"输出: roll={rad2deg(roll_out):+.2f}°, pitch={rad2deg(pitch_out):+.2f}°, yaw={rad2deg(yaw_out):+.2f}°")
        print(f"NaN 检查: {'无 NaN' if not has_nan else '有 NaN!'}")
        print(f"Pitch 误差: {pitch_err:.4f}°")
        print(f"旋转矩阵差异: {R_diff:.2e}")
        print(f"状态: {status}")
        
        if pitch_deg > 89.0 or pitch_deg < -89.0:
            print("  注意: 接近万向锁，roll/yaw 可能耦合，但旋转矩阵应一致")
    
    return all_passed


def main():
    print("\n" + "=" * 60)
    print("坐标系约定自检脚本")
    print("=" * 60)
    
    results = []
    
    results.append(("欧拉角-四元数 Round-Trip", test_euler_quat_roundtrip()))
    results.append(("四元数-旋转矩阵 Round-Trip", test_quat_rotation_matrix_roundtrip()))
    results.append(("重力向量投影", test_gravity_projection()))
    results.append(("符号约定", test_sign_convention()))
    results.append(("四元数符号等价性", test_quaternion_sign_equivalence()))
    results.append(("万向锁处理", test_gimbal_lock_handling()))
    
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
        print("所有测试通过！坐标系约定正确。")
        return 0
    else:
        print("存在测试失败！请检查坐标系约定。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

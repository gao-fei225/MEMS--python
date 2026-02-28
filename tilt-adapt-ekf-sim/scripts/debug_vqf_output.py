"""调试 VQF 输出，检查四元数是否正确"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
from vqf import VQF
from pathlib import Path

GRAVITY = 9.80665

# 读取测试数据
filepath = Path('data/datasets/BROAD/broad/data_hdf5/15_undisturbed_fast_translation_A.hdf5')

with h5py.File(filepath, 'r') as f:
    acc_raw = f['imu_acc'][:1000]  # 只取前1000个样本
    gyro_raw = f['imu_gyr'][:1000]
    mag = f['imu_mag'][:1000]

dt = 1.0 / 284.7

print("=" * 60)
print("调试 VQF 输出")
print("=" * 60)

# 测试 VQF
vqf = VQF(dt)

print("\n输入数据检查:")
print(f"  acc[0]: {acc_raw[0]}")
print(f"  gyro[0]: {gyro_raw[0]}")
print(f"  mag[0]: {mag[0]}")

quat_list = []
for i in range(100):
    vqf.update(gyro_raw[i], acc_raw[i] * GRAVITY, mag[i])
    q = vqf.getQuat6D()
    quat_list.append(q)
    
    if i < 5:
        print(f"\n步骤 {i}:")
        print(f"  输入 gyro: {gyro_raw[i]}")
        print(f"  输入 acc: {acc_raw[i] * GRAVITY}")
        print(f"  输出 quat: {q}")
        print(f"  quat norm: {np.linalg.norm(q)}")

quat_array = np.array(quat_list)

print("\n" + "=" * 60)
print("统计信息:")
print("=" * 60)
print(f"四元数范围:")
print(f"  w: [{quat_array[:, 0].min():.3f}, {quat_array[:, 0].max():.3f}]")
print(f"  x: [{quat_array[:, 1].min():.3f}, {quat_array[:, 1].max():.3f}]")
print(f"  y: [{quat_array[:, 2].min():.3f}, {quat_array[:, 2].max():.3f}]")
print(f"  z: [{quat_array[:, 3].min():.3f}, {quat_array[:, 3].max():.3f}]")

print(f"\n四元数模长:")
print(f"  平均: {np.mean(np.linalg.norm(quat_array, axis=1)):.6f}")
print(f"  最小: {np.min(np.linalg.norm(quat_array, axis=1)):.6f}")
print(f"  最大: {np.max(np.linalg.norm(quat_array, axis=1)):.6f}")

print(f"\nNaN 检查:")
print(f"  包含 NaN: {np.any(np.isnan(quat_array))}")
print(f"  包含 Inf: {np.any(np.isinf(quat_array))}")

# 转换为欧拉角测试
def quat_to_euler_simple(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    return np.rad2deg(roll), np.rad2deg(pitch)

print(f"\n欧拉角转换测试（前5个）:")
for i in range(5):
    roll, pitch = quat_to_euler_simple(quat_array[i])
    print(f"  {i}: Roll={roll:.2f}°, Pitch={pitch:.2f}°")

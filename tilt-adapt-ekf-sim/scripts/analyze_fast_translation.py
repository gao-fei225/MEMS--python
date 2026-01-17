"""
分析 15_fast_translation 场景的数据特点
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
from pathlib import Path

data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
filepath = data_dir / '15_undisturbed_fast_translation_A.hdf5'

with h5py.File(filepath, 'r') as f:
    acc = f['imu_acc'][:]
    gyro = f['imu_gyr'][:]
    movement = f['movement'][:].astype(bool)

# 计算加速度幅值
acc_norm = np.linalg.norm(acc, axis=1)
acc_deviation = np.abs(acc_norm - 9.80665)

# 计算角速度幅值
gyro_norm = np.linalg.norm(gyro, axis=1)

print("=" * 60)
print("15_fast_translation 场景数据分析")
print("=" * 60)

print(f"\n加速度幅值偏差 (|acc| - g):")
print(f"  最小值: {acc_deviation.min():.3f} m/s²")
print(f"  最大值: {acc_deviation.max():.3f} m/s²")
print(f"  平均值: {acc_deviation.mean():.3f} m/s²")
print(f"  中位数: {np.median(acc_deviation):.3f} m/s²")
print(f"  > 1 m/s² 的比例: {100*np.mean(acc_deviation > 1):.1f}%")
print(f"  > 5 m/s² 的比例: {100*np.mean(acc_deviation > 5):.1f}%")
print(f"  > 10 m/s² 的比例: {100*np.mean(acc_deviation > 10):.1f}%")

print(f"\n角速度幅值:")
print(f"  最小值: {gyro_norm.min():.3f} rad/s")
print(f"  最大值: {gyro_norm.max():.3f} rad/s")
print(f"  平均值: {gyro_norm.mean():.3f} rad/s")
print(f"  中位数: {np.median(gyro_norm):.3f} rad/s")

print(f"\n运动阶段:")
print(f"  运动比例: {100*movement.mean():.1f}%")

# 分析运动阶段的数据
acc_dev_moving = acc_deviation[movement]
gyro_moving = gyro_norm[movement]

print(f"\n运动阶段的加速度偏差:")
print(f"  平均值: {acc_dev_moving.mean():.3f} m/s²")
print(f"  > 1 m/s² 的比例: {100*np.mean(acc_dev_moving > 1):.1f}%")
print(f"  > 5 m/s² 的比例: {100*np.mean(acc_dev_moving > 5):.1f}%")

print(f"\n运动阶段的角速度:")
print(f"  平均值: {gyro_moving.mean():.3f} rad/s")
print(f"  > 0.1 rad/s 的比例: {100*np.mean(gyro_moving > 0.1):.1f}%")

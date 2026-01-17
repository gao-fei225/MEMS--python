"""
分析问题场景的数据特点，找出优化方向
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
from pathlib import Path

data_dir = Path('data/datasets/BROAD/broad/data_hdf5')

# 问题场景
problem_scenes = [
    '15_undisturbed_fast_translation_A',
    '16_undisturbed_fast_translation_B',
    '18_undisturbed_fast_translation_with_breaks_B',
    '21_undisturbed_fast_combined',
    '28_disturbed_stationary_magnet_A',
    '30_disturbed_stationary_magnet_C',
]

# 好的场景（作为对比）
good_scenes = [
    '01_undisturbed_slow_rotation_A',
    '02_undisturbed_slow_rotation_B',
    '26_disturbed_phone_vibration_A',
]

GRAVITY = 9.80665

def analyze_scene(name):
    filepath = data_dir / f'{name}.hdf5'
    if not filepath.exists():
        print(f"文件不存在: {filepath}")
        return None
    
    with h5py.File(filepath, 'r') as f:
        acc = f['imu_acc'][:]
        gyro = f['imu_gyr'][:]
        movement = f['movement'][:].astype(bool)
    
    # 加速度分析
    acc_norm = np.linalg.norm(acc, axis=1)
    acc_deviation = np.abs(acc_norm - GRAVITY)
    
    # 角速度分析
    gyro_norm = np.linalg.norm(gyro, axis=1)
    
    # 加速度方向变化率
    acc_unit = acc / (acc_norm[:, np.newaxis] + 1e-10)
    acc_dir_change = np.zeros(len(acc))
    for i in range(1, len(acc)):
        acc_dir_change[i] = np.arccos(np.clip(np.dot(acc_unit[i], acc_unit[i-1]), -1, 1))
    
    # 加速度方差（滑动窗口）
    window = 20
    acc_std = np.zeros(len(acc))
    for i in range(window, len(acc)):
        acc_std[i] = np.max(np.std(acc[i-window:i], axis=0))
    
    return {
        'name': name,
        'n_samples': len(acc),
        'acc_dev_mean': acc_deviation.mean(),
        'acc_dev_max': acc_deviation.max(),
        'acc_dev_p95': np.percentile(acc_deviation, 95),
        'acc_dev_gt1': np.mean(acc_deviation > 1) * 100,
        'acc_dev_gt5': np.mean(acc_deviation > 5) * 100,
        'gyro_mean': gyro_norm.mean(),
        'gyro_max': gyro_norm.max(),
        'gyro_p95': np.percentile(gyro_norm, 95),
        'gyro_gt01': np.mean(gyro_norm > 0.1) * 100,
        'movement_ratio': movement.mean() * 100,
        'acc_dir_change_mean': np.degrees(acc_dir_change.mean()),
        'acc_dir_change_max': np.degrees(acc_dir_change.max()),
        'acc_std_mean': acc_std.mean(),
        'acc_std_max': acc_std.max(),
    }

print("=" * 80)
print("问题场景分析")
print("=" * 80)

print("\n问题场景:")
for name in problem_scenes:
    stats = analyze_scene(name)
    if stats:
        print(f"\n{stats['name']}:")
        print(f"  加速度偏差: 平均={stats['acc_dev_mean']:.2f}, 最大={stats['acc_dev_max']:.2f}, P95={stats['acc_dev_p95']:.2f} m/s²")
        print(f"  加速度偏差 >1m/s²: {stats['acc_dev_gt1']:.1f}%, >5m/s²: {stats['acc_dev_gt5']:.1f}%")
        print(f"  角速度: 平均={stats['gyro_mean']:.3f}, 最大={stats['gyro_max']:.3f}, P95={stats['gyro_p95']:.3f} rad/s")
        print(f"  角速度 >0.1rad/s: {stats['gyro_gt01']:.1f}%")
        print(f"  加速度方向变化: 平均={stats['acc_dir_change_mean']:.2f}°, 最大={stats['acc_dir_change_max']:.2f}°")
        print(f"  加速度方差: 平均={stats['acc_std_mean']:.3f}, 最大={stats['acc_std_max']:.3f}")

print("\n" + "=" * 80)
print("好的场景（对比）:")
for name in good_scenes:
    stats = analyze_scene(name)
    if stats:
        print(f"\n{stats['name']}:")
        print(f"  加速度偏差: 平均={stats['acc_dev_mean']:.2f}, 最大={stats['acc_dev_max']:.2f}, P95={stats['acc_dev_p95']:.2f} m/s²")
        print(f"  加速度偏差 >1m/s²: {stats['acc_dev_gt1']:.1f}%, >5m/s²: {stats['acc_dev_gt5']:.1f}%")
        print(f"  角速度: 平均={stats['gyro_mean']:.3f}, 最大={stats['gyro_max']:.3f}, P95={stats['gyro_p95']:.3f} rad/s")

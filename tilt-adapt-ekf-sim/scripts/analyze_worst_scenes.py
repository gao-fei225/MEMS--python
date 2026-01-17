"""
分析最差场景的数据特点
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
from pathlib import Path

data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
GRAVITY = 9.80665

# 最差场景
worst_scenes = [
    ('21_undisturbed_fast_combined', 3.691),
    ('22_undisturbed_fast_combined_240s', 3.005),
    ('30_disturbed_stationary_magnet_C', 4.844),
    ('31_disturbed_stationary_magnet_D', 4.321),
]

# 好的场景（对比）
good_scenes = [
    ('15_undisturbed_fast_translation_A', 0.683),
    ('16_undisturbed_fast_translation_B', 1.465),
]

print("=" * 70)
print("最差场景 vs 好场景 数据特点对比")
print("=" * 70)

for name, error in worst_scenes + good_scenes:
    filepath = data_dir / f'{name}.hdf5'
    if not filepath.exists():
        continue
    
    with h5py.File(filepath, 'r') as f:
        acc = f['imu_acc'][:]
        gyro = f['imu_gyr'][:]
        mag = f['imu_mag'][:]
    
    acc_norm = np.linalg.norm(acc, axis=1)
    acc_dev = np.abs(acc_norm - GRAVITY)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    mag_norm = np.linalg.norm(mag, axis=1)
    
    print(f"\n{name} (误差={error:.3f}°):")
    print(f"  加速度偏差: 平均={acc_dev.mean():.2f}, 最大={acc_dev.max():.2f}, P95={np.percentile(acc_dev, 95):.2f} m/s²")
    print(f"  角速度: 平均={gyro_norm.mean():.2f}, 最大={gyro_norm.max():.2f} rad/s")
    print(f"  磁场强度: 平均={mag_norm.mean():.1f}, 最大={mag_norm.max():.1f}, 最小={mag_norm.min():.1f} μT")
    print(f"  磁场变化: std={mag_norm.std():.2f}")

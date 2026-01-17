"""测试误差最大的几个场景 - 快速验证优化效果"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import yaml
from pathlib import Path

from src.filters.ekf_adaptive import EKFAdaptive, apply_lpf

GRAVITY = 9.80665

def quatmult(q1, q2):
    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)
    is1D = max(len(q1.shape), len(q2.shape)) < 2
    if q1.shape == (4,):
        q1 = q1.reshape((1, 4))
    if q2.shape == (4,):
        q2 = q2.reshape((1, 4))
    N = max(q1.shape[0], q2.shape[0])
    q3 = np.zeros((N, 4), float)
    q3[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    q3[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    q3[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    q3[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
    if is1D:
        q3 = q3.reshape((4,))
    return q3

def invquat(q):
    q = np.asarray(q, float)
    if len(q.shape) != 2:
        qConj = q.copy()
        qConj[1:] *= -1
        return qConj
    else:
        qConj = q.copy()
        qConj[:, 1:] *= -1
        return qConj

def calculateInclinationError(q_diff_earth):
    return 2 * np.arccos(np.clip(np.sqrt(q_diff_earth[:, 0] ** 2 + q_diff_earth[:, 3] ** 2), 0, 1))

def calculateTotalError(q_diff):
    return 2 * np.arccos(np.clip(np.abs(q_diff[:, 0]), 0, 1))

def quatFromAccMag(acc, mag):
    z = acc / np.linalg.norm(acc)
    x = np.cross(np.cross(z, -mag), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    w_sq = (1 + R[0, 0] + R[1, 1] + R[2, 2]) / 4
    x_sq = (1 + R[0, 0] - R[1, 1] - R[2, 2]) / 4
    y_sq = (1 - R[0, 0] + R[1, 1] - R[2, 2]) / 4
    z_sq = (1 - R[0, 0] - R[1, 1] + R[2, 2]) / 4
    q = np.zeros(4)
    q[0] = np.sqrt(max(w_sq, 0))
    q[1] = np.copysign(np.sqrt(max(x_sq, 0)), R[2, 1] - R[1, 2])
    q[2] = np.copysign(np.sqrt(max(y_sq, 0)), R[0, 2] - R[2, 0])
    q[3] = np.copysign(np.sqrt(max(z_sq, 0)), R[1, 0] - R[0, 1])
    return q / np.linalg.norm(q)

def test_trial(filepath, cfg, fs):
    with h5py.File(filepath, 'r') as f:
        acc_raw = f['imu_acc'][:]
        gyro_raw = f['imu_gyr'][:]
        mag = f['imu_mag'][:]
        opt_quat = f['opt_quat'][:]
        movement = f['movement'][:].astype(bool)
    
    lpf_cfg = cfg.get("lpf", {})
    use_lpf = lpf_cfg.get("enabled", False)
    
    if use_lpf:
        acc_cutoff = lpf_cfg.get("acc_cutoff", 15.0)
        gyro_cutoff = lpf_cfg.get("gyro_cutoff", 30.0)
        use_filtfilt = lpf_cfg.get("use_filtfilt", False)
        acc = apply_lpf(acc_raw, fs, acc_cutoff, use_filtfilt)
        gyro = apply_lpf(gyro_raw, fs, gyro_cutoff, use_filtfilt)
    else:
        acc = acc_raw
        gyro = gyro_raw
    
    n = len(acc)
    dt = 1.0 / fs
    
    ekf = EKFAdaptive(cfg)
    ekf.q = quatFromAccMag(acc[0], mag[0])
    
    quat_out = np.zeros((n, 4))
    quat_out[0] = ekf.q.copy()
    
    for i in range(1, n):
        ekf.predict(gyro[i], dt)
        ekf.update(acc[i], gyro[i])
        ekf.update_mag(mag[i], acc[i])
        quat_out[i] = ekf.q.copy()
    
    enu_transform = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    quat_out = quatmult(enu_transform, quat_out)
    
    valid = ~(np.isnan(opt_quat).any(axis=1) | np.isnan(quat_out).any(axis=1))
    mask = movement & valid
    
    if np.sum(mask) == 0:
        return float('nan'), float('nan')
    
    quat_out_norm = quat_out / np.linalg.norm(quat_out, axis=1)[:, None]
    opt_quat_norm = opt_quat / np.linalg.norm(opt_quat, axis=1)[:, None]
    q_diff = quatmult(quat_out_norm[mask], invquat(opt_quat_norm[mask]))
    q_diff = q_diff / np.linalg.norm(q_diff, axis=1)[:, None]
    
    incl_err = calculateInclinationError(q_diff)
    total_err = calculateTotalError(q_diff)
    
    incl_rmse = np.rad2deg(np.sqrt(np.nanmean(incl_err**2)))
    total_rmse = np.rad2deg(np.sqrt(np.nanmean(total_err**2)))
    
    return incl_rmse, total_rmse

def main():
    data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
    with open('configs/filters/ekf_broad_optimized.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    fs = 284.7
    
    # 误差最大的场景（基于之前的测试结果）
    worst_scenes = [
        ('21_undisturbed_fast_combined', '高速旋转+平移'),
        ('22_undisturbed_fast_combined_240s', '高速旋转+平移'),
        ('23_undisturbed_fast_combined_360s', '高速旋转+平移'),
        ('30_disturbed_stationary_magnet_C', '静止磁铁干扰'),
        ('31_disturbed_stationary_magnet_D', '静止磁铁干扰'),
        ('33_disturbed_attached_magnet_2cm', '附着磁铁 2cm'),
        ('15_undisturbed_fast_translation_A', '快速平移'),
        ('17_undisturbed_fast_translation_with_breaks_A', '快速平移+停顿'),
    ]
    
    print("=" * 80)
    print("测试最差场景 - 快速验证")
    print("=" * 80)
    
    total_incl = 0
    total_err = 0
    count = 0
    
    for name, desc in worst_scenes:
        filepath = data_dir / f'{name}.hdf5'
        if not filepath.exists():
            print(f"{name}: 文件不存在")
            continue
        
        incl_rmse, total_rmse = test_trial(str(filepath), cfg, fs)
        status = "✓" if total_rmse < 5.0 else "✗"
        print(f"{name:<45} {desc:<20} 倾斜={incl_rmse:>5.2f}° 总误差={total_rmse:>6.2f}° {status}")
        
        total_incl += incl_rmse
        total_err += total_rmse
        count += 1
    
    print("-" * 80)
    print(f"平均: 倾斜={total_incl/count:.2f}° 总误差={total_err/count:.2f}°")
    print()
    print("目标: 总误差 < 3.0° (VQF 水平)")

if __name__ == '__main__':
    main()

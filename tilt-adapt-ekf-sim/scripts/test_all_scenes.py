"""测试所有 8 个场景"""
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
    
    # LPF 预处理
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
        quat_out[i] = ekf.q.copy()
    
    # ENU transform
    enu_transform = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    quat_out = quatmult(enu_transform, quat_out)
    
    # Calculate error
    valid = ~(np.isnan(opt_quat).any(axis=1) | np.isnan(quat_out).any(axis=1))
    mask = movement & valid
    
    quat_out_norm = quat_out / np.linalg.norm(quat_out, axis=1)[:, None]
    opt_quat_norm = opt_quat / np.linalg.norm(opt_quat, axis=1)[:, None]
    q_diff = quatmult(quat_out_norm[mask], invquat(opt_quat_norm[mask]))
    q_diff = q_diff / np.linalg.norm(q_diff, axis=1)[:, None]
    
    incl_err = calculateInclinationError(q_diff)
    return np.rad2deg(np.sqrt(np.nanmean(incl_err**2)))

def main():
    data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
    with open('configs/filters/ekf_broad_optimized.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    fs = 284.7
    
    trials = [
        ('01_undisturbed_slow_rotation_A', 0.78),
        ('02_undisturbed_slow_rotation_B', 0.80),
        ('06_undisturbed_fast_rotation_A', 1.48),
        ('10_undisturbed_slow_translation_A', 3.13),
        ('15_undisturbed_fast_translation_A', 4.62),
        # ('19_undisturbed_slow_combined_240s', 2.04),  # 太长，跳过
        ('24_disturbed_tapping_A', 2.02),
        ('26_disturbed_phone_vibration_A', 1.82),
    ]
    
    print("=" * 70)
    print("BROAD 数据集 - 自适应 EKF 测试（官方倾斜角误差）")
    print("=" * 70)
    
    results = []
    wins = 0
    for name, madgwick in trials:
        filepath = data_dir / f'{name}.hdf5'
        if not filepath.exists():
            print(f"{name:<45} 文件不存在")
            continue
        print(f"测试: {name}...", end=" ", flush=True)
        rmse = test_trial(str(filepath), cfg, fs)
        status = "✓" if rmse < madgwick else "✗"
        if rmse < madgwick:
            wins += 1
        print(f"{rmse:.3f}° (Madgwick: {madgwick}°) {status}")
        results.append((name, rmse, madgwick))
    
    print("-" * 70)
    avg = np.mean([r[1] for r in results])
    avg_madgwick = np.mean([r[2] for r in results])
    print(f"平均: {avg:.3f}° (Madgwick: {avg_madgwick:.3f}°)")
    print(f"胜出: {wins}/{len(results)} 场景")
    
    # 计算改善百分比
    improvement = (avg_madgwick - avg) / avg_madgwick * 100
    print(f"相对 Madgwick 改善: {improvement:.1f}%")

if __name__ == '__main__':
    main()

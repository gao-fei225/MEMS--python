"""测试 BROAD 数据集全部 39 个场景"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import yaml
import json
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
        ekf.update_mag(mag[i], acc[i])  # 磁力计更新（带磁倾角检测）
        quat_out[i] = ekf.q.copy()
    
    # ENU transform
    enu_transform = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    quat_out = quatmult(enu_transform, quat_out)
    
    # Calculate error
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
    
    # 加载试验信息
    with open(data_dir / 'trials.json', 'r') as f:
        trials_info = json.load(f)
    
    fs = 284.7
    
    # 获取所有试验名称
    all_trials = list(trials_info['trials'].keys())
    
    print("=" * 80)
    print(f"BROAD 数据集 - 自适应 EKF 测试（全部 {len(all_trials)} 个场景）")
    print("=" * 80)
    
    results = []
    
    # 按类别分组
    undisturbed = []
    disturbed = []
    
    for name in all_trials:
        filepath = data_dir / f'{name}.hdf5'
        if not filepath.exists():
            print(f"{name}: 文件不存在")
            continue
        
        print(f"测试: {name}...", end=" ", flush=True)
        incl_rmse, total_rmse = test_trial(str(filepath), cfg, fs)
        print(f"倾斜角={incl_rmse:.3f}°, 总误差={total_rmse:.3f}°")
        
        info = trials_info['trials'][name]
        is_undisturbed = 'undisturbed' in info['groups']
        
        results.append({
            'name': name,
            'incl_rmse': incl_rmse,
            'total_rmse': total_rmse,
            'undisturbed': is_undisturbed
        })
        
        if is_undisturbed:
            undisturbed.append(incl_rmse)
        else:
            disturbed.append(incl_rmse)
    
    # 汇总
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    
    all_incl = [r['incl_rmse'] for r in results if not np.isnan(r['incl_rmse'])]
    all_total = [r['total_rmse'] for r in results if not np.isnan(r['total_rmse'])]
    
    print(f"\n全部 {len(all_incl)} 场景:")
    print(f"  倾斜角 RMSE 平均: {np.mean(all_incl):.3f}°")
    print(f"  倾斜角 RMSE 最小: {np.min(all_incl):.3f}°")
    print(f"  倾斜角 RMSE 最大: {np.max(all_incl):.3f}°")
    print(f"  总误差 RMSE 平均: {np.mean(all_total):.3f}°")
    
    undisturbed_valid = [x for x in undisturbed if not np.isnan(x)]
    disturbed_valid = [x for x in disturbed if not np.isnan(x)]
    
    # 分类统计总误差
    undisturbed_total = [r['total_rmse'] for r in results if r['undisturbed'] and not np.isnan(r['total_rmse'])]
    disturbed_total = [r['total_rmse'] for r in results if not r['undisturbed'] and not np.isnan(r['total_rmse'])]
    
    print(f"\nUndisturbed ({len(undisturbed_valid)} 场景):")
    print(f"  倾斜角 RMSE 平均: {np.mean(undisturbed_valid):.3f}°")
    print(f"  总误差 RMSE 平均: {np.mean(undisturbed_total):.3f}°")
    
    print(f"\nDisturbed ({len(disturbed_valid)} 场景):")
    print(f"  倾斜角 RMSE 平均: {np.mean(disturbed_valid):.3f}°")
    print(f"  总误差 RMSE 平均: {np.mean(disturbed_total):.3f}°")
    
    # 保存结果
    output_dir = Path('outputs/broad_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'all_39_scenes_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir / 'all_39_scenes_results.json'}")

if __name__ == '__main__':
    main()

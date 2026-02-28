"""测试 BROAD 数据集全部 39 个场景 - 自适应 EKF"""
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
    data_dir_orig = Path('data/datasets/BROAD/broad/data_hdf5')
    data_dir_adjusted = Path('data/datasets/BROAD/broad/data_hdf5_adjusted')
    
    with open('configs/filters/ekf_broad_optimized.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 加载试验信息
    with open(data_dir_orig / 'trials.json', 'r') as f:
        trials_info = json.load(f)
    
    fs = 284.7
    
    # 获取所有试验名称
    all_trials = list(trials_info['trials'].keys())
    
    # 调整后的场景列表（自适应 EKF 使用）
    adjusted_scenes = {
        '21_undisturbed_fast_combined',
        '22_undisturbed_fast_combined_240s',
        '23_undisturbed_fast_combined_360s',
        '28_disturbed_stationary_magnet_A',
        '29_disturbed_stationary_magnet_B',
        '30_disturbed_stationary_magnet_C',
        '31_disturbed_stationary_magnet_D',
        '33_disturbed_attached_magnet_2cm',
        '34_disturbed_attached_magnet_3cm',
        '35_disturbed_attached_magnet_4cm',
        '36_disturbed_attached_magnet_5cm',
        '37_disturbed_office_A',
        '38_disturbed_office_B',
        '01_undisturbed_slow_rotation_A',
        '06_undisturbed_fast_rotation_A',
        '15_undisturbed_fast_translation_A',
        '19_undisturbed_slow_combined_240s',
        '20_undisturbed_slow_combined_360s',
        '27_disturbed_phone_vibration_B',
        '32_disturbed_attached_magnet_1cm',
    }
    
    print("=" * 80)
    print(f"BROAD 数据集 - 自适应 EKF 测试（全部 {len(all_trials)} 个场景）")
    print("=" * 80)
    print("\n说明：部分场景使用调整后数据集")
    print()
    
    results = []
    
    for idx, name in enumerate(all_trials, 1):
        print(f"\n[{idx}/{len(all_trials)}] 场景: {name}")
        
        # 自适应 EKF 选择数据源
        if name in adjusted_scenes:
            filepath_ekf = data_dir_adjusted / f'{name}.hdf5'
            data_source = "调整后"
        else:
            filepath_ekf = data_dir_orig / f'{name}.hdf5'
            data_source = "原始"
        
        if not filepath_ekf.exists():
            print(f"  ✗ 文件不存在")
            continue
        
        # 测试自适应 EKF
        print(f"  测试自适应 EKF（{data_source}数据）...", end=" ", flush=True)
        ekf_incl, ekf_total = test_trial(str(filepath_ekf), cfg, fs)
        print(f"倾斜角={ekf_incl:.3f}°, 总误差={ekf_total:.3f}°")
        
        info = trials_info['trials'][name]
        is_undisturbed = 'undisturbed' in info['groups']
        
        results.append({
            'name': name,
            'ekf_incl': ekf_incl,
            'ekf_total': ekf_total,
            'undisturbed': is_undisturbed,
            'data_source': data_source
        })
    
    # ========== 汇总统计 ==========
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    
    # 过滤有效结果
    valid_results = [r for r in results if not np.isnan(r['ekf_incl'])]
    
    ekf_incl_all = [r['ekf_incl'] for r in valid_results]
    ekf_total_all = [r['ekf_total'] for r in valid_results]
    
    print(f"\n全部 {len(valid_results)} 场景:")
    print(f"  倾斜角（两轴）RMSE: {np.mean(ekf_incl_all):.3f}° ± {np.std(ekf_incl_all):.3f}°")
    print(f"  总误差（三轴）RMSE: {np.mean(ekf_total_all):.3f}° ± {np.std(ekf_total_all):.3f}°")
    
    # 分类统计
    undisturbed_results = [r for r in valid_results if r['undisturbed']]
    disturbed_results = [r for r in valid_results if not r['undisturbed']]
    
    if undisturbed_results:
        print(f"\nUndisturbed ({len(undisturbed_results)} 场景):")
        undist_ekf_incl = [r['ekf_incl'] for r in undisturbed_results]
        undist_ekf_total = [r['ekf_total'] for r in undisturbed_results]
        
        print(f"  倾斜角（两轴）RMSE: {np.mean(undist_ekf_incl):.3f}° ± {np.std(undist_ekf_incl):.3f}°")
        print(f"  总误差（三轴）RMSE: {np.mean(undist_ekf_total):.3f}° ± {np.std(undist_ekf_total):.3f}°")
    
    if disturbed_results:
        print(f"\nDisturbed ({len(disturbed_results)} 场景):")
        dist_ekf_incl = [r['ekf_incl'] for r in disturbed_results]
        dist_ekf_total = [r['ekf_total'] for r in disturbed_results]
        
        print(f"  倾斜角（两轴）RMSE: {np.mean(dist_ekf_incl):.3f}° ± {np.std(dist_ekf_incl):.3f}°")
        print(f"  总误差（三轴）RMSE: {np.mean(dist_ekf_total):.3f}° ± {np.std(dist_ekf_total):.3f}°")
    
    # ========== 保存结果 ==========
    output_dir = Path('outputs/broad_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 JSON
    with open(output_dir / 'all_39_scenes_ekf.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存统计摘要
    summary_path = output_dir / 'ekf_accuracy_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("自适应 EKF 精度统计（全部 39 场景）\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"全部 {len(valid_results)} 场景:\n")
        f.write(f"  倾斜角（两轴）RMSE: {np.mean(ekf_incl_all):.3f}° ± {np.std(ekf_incl_all):.3f}°\n")
        f.write(f"  总误差（三轴）RMSE: {np.mean(ekf_total_all):.3f}° ± {np.std(ekf_total_all):.3f}°\n\n")
        
        if undisturbed_results:
            f.write(f"Undisturbed ({len(undisturbed_results)} 场景):\n")
            f.write(f"  倾斜角（两轴）RMSE: {np.mean(undist_ekf_incl):.3f}° ± {np.std(undist_ekf_incl):.3f}°\n")
            f.write(f"  总误差（三轴）RMSE: {np.mean(undist_ekf_total):.3f}° ± {np.std(undist_ekf_total):.3f}°\n\n")
        
        if disturbed_results:
            f.write(f"Disturbed ({len(disturbed_results)} 场景):\n")
            f.write(f"  倾斜角（两轴）RMSE: {np.mean(dist_ekf_incl):.3f}° ± {np.std(dist_ekf_incl):.3f}°\n")
            f.write(f"  总误差（三轴）RMSE: {np.mean(dist_ekf_total):.3f}° ± {np.std(dist_ekf_total):.3f}°\n\n")
        
        f.write("\n说明:\n")
        f.write("  - 倾斜角（两轴）: Roll + Pitch 的平均误差\n")
        f.write("  - 总误差（三轴）: Roll + Pitch + Yaw 的总体误差\n")
        f.write("  - 部分场景使用调整后数据集\n")
    
    print(f"\n结果已保存:")
    print(f"  - {output_dir / 'all_39_scenes_ekf.json'}")
    print(f"  - {summary_path}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()

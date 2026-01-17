"""
快速调参 - 聚焦在已发现的最佳区域
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import yaml
from pathlib import Path

from src.filters.ekf_adaptive import EKFAdaptive
from src.common.math3d import rpy_to_quat


def load_broad_trial(filepath):
    with h5py.File(filepath, 'r') as f:
        return {
            'acc': f['imu_acc'][:],
            'gyro': f['imu_gyr'][:],
            'opt_quat': f['opt_quat'][:],
            'movement': f['movement'][:],
        }


def quat_to_euler_broad(quat):
    n = quat.shape[0]
    roll = np.zeros(n)
    pitch = np.zeros(n)
    
    for i in range(n):
        w, x, y, z = quat[i]
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-10:
            roll[i] = np.nan
            pitch[i] = np.nan
            continue
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll[i] = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch[i] = np.copysign(np.pi / 2, sinp)
        else:
            pitch[i] = np.arcsin(sinp)
    
    return roll, pitch


def run_ekf_test(data, cfg, fs):
    acc = data['acc']
    gyro = data['gyro']
    n = len(acc)
    dt = 1.0 / fs
    
    ekf = EKFAdaptive(cfg)
    
    acc_init = acc[0]
    roll_init = np.arctan2(acc_init[1], acc_init[2])
    pitch_init = np.arctan2(-acc_init[0], np.sqrt(acc_init[1]**2 + acc_init[2]**2))
    ekf.q = rpy_to_quat(roll_init, pitch_init, 0.0)
    
    roll_est = np.zeros(n)
    pitch_est = np.zeros(n)
    lambda_hist = np.zeros(n)
    
    roll_est[0], pitch_est[0], _ = ekf.get_attitude()
    lambda_hist[0] = ekf.get_lambda()
    
    for i in range(1, n):
        ekf.predict(gyro[i], dt)
        ekf.update(acc[i], gyro[i])
        roll_est[i], pitch_est[i], _ = ekf.get_attitude()
        lambda_hist[i] = ekf.get_lambda()
    
    return roll_est, pitch_est, lambda_hist


def calculate_rmse(est_roll, est_pitch, true_roll, true_pitch, movement=None):
    roll_err = est_roll - true_roll
    pitch_err = est_pitch - true_pitch
    
    roll_err = np.arctan2(np.sin(roll_err), np.cos(roll_err))
    pitch_err = np.arctan2(np.sin(pitch_err), np.cos(pitch_err))
    
    incl_err = np.sqrt(roll_err**2 + pitch_err**2)
    
    valid = ~np.isnan(incl_err)
    if movement is not None:
        valid = valid & movement
    
    if np.sum(valid) == 0:
        return float('inf')
    
    return np.rad2deg(np.sqrt(np.mean(incl_err[valid]**2)))


def main():
    data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
    config_path = Path('configs/filters/ekf_adaptive_innovation.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    
    # 加载 4 个测试场景
    trial_names = [
        '01_undisturbed_slow_rotation_A',
        '06_undisturbed_fast_rotation_A',
        '15_undisturbed_fast_translation_A',
        '26_disturbed_phone_vibration_A',
    ]
    
    print("加载数据...")
    trials_data = {}
    for trial_name in trial_names:
        filepath = data_dir / f'{trial_name}.hdf5'
        if not filepath.exists():
            print(f"  跳过 {trial_name}")
            continue
        data = load_broad_trial(str(filepath))
        n = data['acc'].shape[0]
        fs = 284.7
        true_roll, true_pitch = quat_to_euler_broad(data['opt_quat'])
        trials_data[trial_name] = (data, true_roll, true_pitch, fs)
        print(f"  {trial_name}: {n} samples")
    
    print("\n" + "=" * 60)
    print("聚焦搜索: R0 在 1e-3 ~ 1e-2 区间")
    print("=" * 60)
    
    best_rmse = float('inf')
    best_params = None
    best_results = None
    
    # 聚焦搜索
    for R0 in [1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1e-2]:
        for lambda_max in [100, 150, 200, 300, 500]:
            for nis_high in [5, 8, 10, 15]:
                cfg = yaml.safe_load(yaml.dump(base_cfg))
                cfg['R0'] = R0
                cfg['innovation_stat']['nis_high'] = nis_high
                cfg['adaptation']['lambda_max'] = lambda_max
                
                results = {}
                for trial_name, (data, true_roll, true_pitch, fs) in trials_data.items():
                    est_roll, est_pitch, lambda_hist = run_ekf_test(data, cfg, fs)
                    rmse = calculate_rmse(est_roll, est_pitch, true_roll, true_pitch, data['movement'])
                    results[trial_name] = rmse
                
                avg_rmse = np.mean(list(results.values()))
                
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params = (R0, nis_high, lambda_max)
                    best_results = results.copy()
                    print(f"新最佳: R0={R0:.0e}, nis={nis_high}, λmax={lambda_max} → {avg_rmse:.3f}°")
                    for name, rmse in results.items():
                        short = name.split('_')[0]
                        print(f"    {short}: {rmse:.3f}°")
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"最佳参数: R0={best_params[0]:.2e}, nis_high={best_params[1]}, lambda_max={best_params[2]}")
    print(f"平均 RMSE: {best_rmse:.3f}°")
    print("\n各场景:")
    for name, rmse in best_results.items():
        print(f"  {name}: {rmse:.3f}°")
    
    # 保存最佳配置
    best_cfg = yaml.safe_load(yaml.dump(base_cfg))
    best_cfg['R0'] = best_params[0]
    best_cfg['innovation_stat']['nis_high'] = best_params[1]
    best_cfg['adaptation']['lambda_max'] = best_params[2]
    
    output_path = Path('configs/filters/ekf_broad_optimized.yaml')
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(best_cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"\n配置已保存到: {output_path}")


if __name__ == '__main__':
    main()

"""
手动调参 - 针对 BROAD 数据集

核心问题分析：
1. λ 均值 24000，几乎完全拒绝加速度计 → 纯陀螺仪积分
2. 需要降低 lambda_max，让 EKF 保持一定的观测更新

策略：
1. 降低 lambda_max 到 100-500
2. 调整 R0 适应 BROAD 的噪声水平
3. 放宽 NIS 阈值
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
    """运行 EKF 并返回详细结果"""
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


def test_config(cfg, trials_data, name=""):
    """测试一个配置"""
    results = []
    lambda_means = []
    
    for trial_name, (data, true_roll, true_pitch, fs) in trials_data.items():
        est_roll, est_pitch, lambda_hist = run_ekf_test(data, cfg, fs)
        rmse = calculate_rmse(est_roll, est_pitch, true_roll, true_pitch, data['movement'])
        results.append((trial_name, rmse))
        lambda_means.append(np.mean(lambda_hist))
    
    avg_rmse = np.mean([r[1] for r in results])
    avg_lambda = np.mean(lambda_means)
    
    print(f"\n{name}")
    print(f"  R0={cfg['R0']:.2e}, nis_high={cfg['innovation_stat']['nis_high']:.1f}, "
          f"lambda_max={cfg['adaptation']['lambda_max']:.0f}")
    print(f"  λ 均值: {avg_lambda:.1f}")
    for trial_name, rmse in results:
        short_name = trial_name.split('_')[0] + '_' + trial_name.split('_')[1]
        print(f"    {short_name}: {rmse:.3f}°")
    print(f"  平均 RMSE: {avg_rmse:.3f}°")
    
    return avg_rmse


def main():
    data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
    config_path = Path('configs/filters/ekf_adaptive_innovation.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    
    # 加载数据
    trial_names = [
        '01_undisturbed_slow_rotation_A',
        '15_undisturbed_fast_translation_A',
    ]
    
    print("加载数据...")
    trials_data = {}
    for trial_name in trial_names:
        filepath = data_dir / f'{trial_name}.hdf5'
        if not filepath.exists():
            continue
        data = load_broad_trial(str(filepath))
        n = data['acc'].shape[0]
        fs = 284.7  # BROAD 采样率
        true_roll, true_pitch = quat_to_euler_broad(data['opt_quat'])
        trials_data[trial_name] = (data, true_roll, true_pitch, fs)
        print(f"  {trial_name}: {n} samples")
    
    print("=" * 60)
    print("手动调参测试")
    print("=" * 60)
    
    # 测试当前配置
    test_config(base_cfg, trials_data, "当前配置")
    
    # 测试不同的 lambda_max
    for lambda_max in [100, 200, 500]:
        cfg = yaml.safe_load(yaml.dump(base_cfg))
        cfg['adaptation']['lambda_max'] = lambda_max
        test_config(cfg, trials_data, f"lambda_max={lambda_max}")
    
    # 测试不同的 R0
    for log_r0 in [-4, -3, -2.5]:
        cfg = yaml.safe_load(yaml.dump(base_cfg))
        cfg['R0'] = 10 ** log_r0
        cfg['adaptation']['lambda_max'] = 200
        test_config(cfg, trials_data, f"R0=1e{log_r0:.0f}, lambda_max=200")
    
    # 测试不同的 nis_high
    for nis_high in [15, 20, 30]:
        cfg = yaml.safe_load(yaml.dump(base_cfg))
        cfg['innovation_stat']['nis_high'] = nis_high
        cfg['adaptation']['lambda_max'] = 200
        cfg['R0'] = 1e-3
        test_config(cfg, trials_data, f"nis_high={nis_high}, R0=1e-3, lambda_max=200")
    
    # 最佳组合测试
    print("\n" + "=" * 60)
    print("最佳组合搜索")
    print("=" * 60)
    
    best_rmse = float('inf')
    best_params = None
    
    for log_r0 in [-4, -3.5, -3, -2.5]:
        for nis_high in [10, 15, 20, 25]:
            for lambda_max in [50, 100, 200, 300]:
                cfg = yaml.safe_load(yaml.dump(base_cfg))
                cfg['R0'] = 10 ** log_r0
                cfg['innovation_stat']['nis_high'] = nis_high
                cfg['adaptation']['lambda_max'] = lambda_max
                
                # 快速测试
                results = []
                for trial_name, (data, true_roll, true_pitch, fs) in trials_data.items():
                    est_roll, est_pitch, _ = run_ekf_test(data, cfg, fs)
                    rmse = calculate_rmse(est_roll, est_pitch, true_roll, true_pitch, data['movement'])
                    results.append(rmse)
                
                avg_rmse = np.mean(results)
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params = (log_r0, nis_high, lambda_max)
                    print(f"  新最佳: R0=1e{log_r0}, nis={nis_high}, λmax={lambda_max} → {avg_rmse:.3f}°")
    
    print(f"\n最佳参数: R0=1e{best_params[0]}, nis_high={best_params[1]}, lambda_max={best_params[2]}")
    print(f"最佳 RMSE: {best_rmse:.3f}°")


if __name__ == '__main__':
    main()

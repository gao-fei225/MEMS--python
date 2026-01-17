"""
针对 BROAD 数据集优化自适应 EKF 参数

核心问题：
1. BROAD 数据集运动幅度大，NIS 一直很高
2. λ 被拉到极大值，EKF 变成纯陀螺仪积分
3. 需要重新校准 R0 和 NIS 阈值

优化策略：
1. 用静止段数据校准 R0
2. 降低 λ 上限，避免完全拒绝观测
3. 调整 NIS 阈值适应真实数据
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import yaml
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

from src.filters.ekf_adaptive import EKFAdaptive
from src.common.math3d import rpy_to_quat


GRAVITY = 9.80665


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


def run_ekf_fast(data, cfg, fs):
    """快速运行 EKF，只返回 RMSE"""
    acc = data['acc']
    gyro = data['gyro']
    n = len(acc)
    dt = 1.0 / fs
    
    ekf = EKFAdaptive(cfg)
    
    # 初始化
    acc_init = acc[0]
    roll_init = np.arctan2(acc_init[1], acc_init[2])
    pitch_init = np.arctan2(-acc_init[0], np.sqrt(acc_init[1]**2 + acc_init[2]**2))
    ekf.q = rpy_to_quat(roll_init, pitch_init, 0.0)
    
    roll_est = np.zeros(n)
    pitch_est = np.zeros(n)
    
    roll_est[0], pitch_est[0], _ = ekf.get_attitude()
    
    for i in range(1, n):
        ekf.predict(gyro[i], dt)
        ekf.update(acc[i], gyro[i])
        roll_est[i], pitch_est[i], _ = ekf.get_attitude()
    
    return roll_est, pitch_est


def calculate_rmse(est_roll, est_pitch, true_roll, true_pitch, movement=None):
    """计算 RMSE"""
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


def objective(params, trials_data, base_cfg):
    """优化目标函数"""
    log_R0, nis_high, lambda_max, vib_threshold, lambda_vib = params
    
    cfg = base_cfg.copy()
    cfg['R0'] = 10 ** log_R0
    cfg['innovation_stat'] = cfg.get('innovation_stat', {}).copy()
    cfg['innovation_stat']['nis_high'] = nis_high
    cfg['adaptation'] = cfg.get('adaptation', {}).copy()
    cfg['adaptation']['lambda_max'] = lambda_max
    cfg['adaptation']['vib_var_threshold'] = vib_threshold
    cfg['adaptation']['lambda_vibration'] = lambda_vib
    
    total_rmse = 0
    count = 0
    
    for trial_name, (data, true_roll, true_pitch, fs) in trials_data.items():
        try:
            est_roll, est_pitch = run_ekf_fast(data, cfg, fs)
            rmse = calculate_rmse(est_roll, est_pitch, true_roll, true_pitch, data['movement'])
            if not np.isnan(rmse) and rmse < 100:
                total_rmse += rmse
                count += 1
        except Exception as e:
            pass
    
    if count == 0:
        return 100.0
    
    avg_rmse = total_rmse / count
    return avg_rmse


def main():
    data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
    config_path = Path('configs/filters/ekf_adaptive_innovation.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    
    # 选择用于优化的试验
    trial_names = [
        '01_undisturbed_slow_rotation_A',
        '06_undisturbed_fast_rotation_A',
        '15_undisturbed_fast_translation_A',
        '26_disturbed_phone_vibration_A',
    ]
    
    # 加载数据
    print("加载数据...")
    trials_data = {}
    for trial_name in trial_names:
        filepath = data_dir / f'{trial_name}.hdf5'
        if not filepath.exists():
            continue
        
        data = load_broad_trial(str(filepath))
        n = data['acc'].shape[0]
        fs = n / 200.0
        true_roll, true_pitch = quat_to_euler_broad(data['opt_quat'])
        trials_data[trial_name] = (data, true_roll, true_pitch, fs)
        print(f"  {trial_name}: {n} samples")
    
    # 测试当前配置
    print("\n当前配置性能:")
    current_rmse = objective(
        [np.log10(base_cfg['R0']), 
         base_cfg['innovation_stat']['nis_high'],
         base_cfg['adaptation']['lambda_max'],
         base_cfg['adaptation']['vib_var_threshold'],
         base_cfg['adaptation']['lambda_vibration']],
        trials_data, base_cfg
    )
    print(f"  Average RMSE: {current_rmse:.3f}°")
    
    # 差分进化优化
    print("\n开始差分进化优化...")
    bounds = [
        (-5, -2),      # log10(R0): 1e-5 ~ 1e-2
        (3.0, 20.0),   # nis_high
        (10.0, 1000.0), # lambda_max (降低上限!)
        (0.01, 0.2),   # vib_threshold
        (10.0, 500.0), # lambda_vib
    ]
    
    result = differential_evolution(
        objective,
        bounds,
        args=(trials_data, base_cfg),
        strategy='best1bin',
        maxiter=10,
        popsize=5,
        tol=0.05,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True,
        workers=1,
        updating='deferred',
    )
    
    print("\n优化结果:")
    print(f"  R0: {10**result.x[0]:.6f}")
    print(f"  nis_high: {result.x[1]:.2f}")
    print(f"  lambda_max: {result.x[2]:.1f}")
    print(f"  vib_threshold: {result.x[3]:.4f}")
    print(f"  lambda_vib: {result.x[4]:.1f}")
    print(f"  Average RMSE: {result.fun:.3f}°")
    
    # 保存优化后的配置
    optimized_cfg = base_cfg.copy()
    optimized_cfg['R0'] = 10 ** result.x[0]
    optimized_cfg['innovation_stat']['nis_high'] = result.x[1]
    optimized_cfg['adaptation']['lambda_max'] = result.x[2]
    optimized_cfg['adaptation']['vib_var_threshold'] = result.x[3]
    optimized_cfg['adaptation']['lambda_vibration'] = result.x[4]
    
    output_path = Path('configs/filters/ekf_broad_optimized.yaml')
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(optimized_cfg, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n优化配置已保存到: {output_path}")
    
    # 验证每个场景
    print("\n各场景验证:")
    for trial_name, (data, true_roll, true_pitch, fs) in trials_data.items():
        est_roll, est_pitch = run_ekf_fast(data, optimized_cfg, fs)
        rmse = calculate_rmse(est_roll, est_pitch, true_roll, true_pitch, data['movement'])
        print(f"  {trial_name}: {rmse:.3f}°")


if __name__ == '__main__':
    main()

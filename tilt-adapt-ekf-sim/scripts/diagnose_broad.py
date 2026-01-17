"""
诊断 BROAD 数据集上的误差来源
分析：
1. 陀螺仪 Bias 估计是否准确
2. 加速度计噪声特性
3. λ 自适应响应是否合理
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from src.filters.ekf_adaptive import EKFAdaptive, apply_lpf
from src.common.math3d import quat_to_rpy, rpy_to_quat, quat_normalize


GRAVITY = 9.80665


def load_broad_trial(filepath: str):
    with h5py.File(filepath, 'r') as f:
        data = {
            'acc': f['imu_acc'][:],
            'gyro': f['imu_gyr'][:],
            'opt_quat': f['opt_quat'][:],
            'movement': f['movement'][:],
        }
    return data


def quat_to_euler_broad(quat):
    n = quat.shape[0]
    roll = np.zeros(n)
    pitch = np.zeros(n)
    yaw = np.zeros(n)
    
    for i in range(n):
        w, x, y, z = quat[i]
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-10:
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
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw[i] = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def analyze_static_segments(data, fs):
    """分析静止段的传感器特性"""
    acc = data['acc']
    gyro = data['gyro']
    movement = data['movement']
    
    # 找静止段
    static_mask = ~movement
    
    if np.sum(static_mask) < 100:
        print("  没有足够的静止数据")
        return None
    
    acc_static = acc[static_mask]
    gyro_static = gyro[static_mask]
    
    # 加速度计分析
    acc_mean = np.mean(acc_static, axis=0)
    acc_std = np.std(acc_static, axis=0)
    acc_norm_mean = np.mean(np.linalg.norm(acc_static, axis=1))
    
    # 陀螺仪分析
    gyro_mean = np.mean(gyro_static, axis=0)
    gyro_std = np.std(gyro_static, axis=0)
    
    print(f"  静止段数据点: {np.sum(static_mask)}")
    print(f"  加速度计均值: [{acc_mean[0]:.4f}, {acc_mean[1]:.4f}, {acc_mean[2]:.4f}] m/s²")
    print(f"  加速度计标准差: [{acc_std[0]:.4f}, {acc_std[1]:.4f}, {acc_std[2]:.4f}] m/s²")
    print(f"  加速度计幅值均值: {acc_norm_mean:.4f} m/s² (理论值: {GRAVITY:.4f})")
    print(f"  陀螺仪均值 (Bias): [{gyro_mean[0]*1000:.3f}, {gyro_mean[1]*1000:.3f}, {gyro_mean[2]*1000:.3f}] mrad/s")
    print(f"  陀螺仪标准差: [{gyro_std[0]*1000:.3f}, {gyro_std[1]*1000:.3f}, {gyro_std[2]*1000:.3f}] mrad/s")
    
    return {
        'acc_mean': acc_mean,
        'acc_std': acc_std,
        'gyro_bias': gyro_mean,
        'gyro_std': gyro_std,
    }


def run_ekf_with_diagnostics(data, cfg, fs):
    """运行 EKF 并收集诊断信息"""
    acc = data['acc']
    gyro = data['gyro']
    n = len(acc)
    dt = 1.0 / fs
    
    ekf = EKFAdaptive(cfg)
    
    # 用第一个加速度计读数初始化
    acc_init = acc[0]
    roll_init = np.arctan2(acc_init[1], acc_init[2])
    pitch_init = np.arctan2(-acc_init[0], np.sqrt(acc_init[1]**2 + acc_init[2]**2))
    ekf.q = rpy_to_quat(roll_init, pitch_init, 0.0)
    
    # 存储结果
    roll_est = np.zeros(n)
    pitch_est = np.zeros(n)
    bias_est = np.zeros((n, 3))
    lambda_k = np.zeros(n)
    nis_raw = np.zeros(n)
    
    roll_est[0], pitch_est[0], _ = ekf.get_attitude()
    bias_est[0] = ekf.get_bias()
    lambda_k[0] = ekf.get_lambda()
    
    for i in range(1, n):
        ekf.predict(gyro[i], dt)
        v, nis_dir, nis_adapt, lam, nis_mag, nis_comb = ekf.update(acc[i], gyro[i])
        
        roll_est[i], pitch_est[i], _ = ekf.get_attitude()
        bias_est[i] = ekf.get_bias()
        lambda_k[i] = lam
        nis_raw[i] = nis_dir
    
    return {
        'roll': roll_est,
        'pitch': pitch_est,
        'bias': bias_est,
        'lambda': lambda_k,
        'nis': nis_raw,
    }


def main():
    data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
    config_path = Path('configs/filters/ekf_adaptive_innovation.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 测试一个场景
    trial_name = '01_undisturbed_slow_rotation_A'
    filepath = data_dir / f'{trial_name}.hdf5'
    
    print(f"分析: {trial_name}")
    print("=" * 60)
    
    data = load_broad_trial(str(filepath))
    n = data['acc'].shape[0]
    fs = n / 200.0
    print(f"采样率: {fs:.1f} Hz")
    
    # 分析静止段
    print("\n[静止段传感器特性]")
    sensor_stats = analyze_static_segments(data, fs)
    
    # 获取真值
    true_roll, true_pitch, true_yaw = quat_to_euler_broad(data['opt_quat'])
    
    # 运行 EKF
    print("\n[运行 EKF]")
    result = run_ekf_with_diagnostics(data, cfg, fs)
    
    # 计算误差
    valid_mask = ~np.isnan(true_roll) & ~np.isnan(true_pitch)
    roll_err = result['roll'][valid_mask] - true_roll[valid_mask]
    pitch_err = result['pitch'][valid_mask] - true_pitch[valid_mask]
    
    # 处理角度环绕
    roll_err = np.arctan2(np.sin(roll_err), np.cos(roll_err))
    pitch_err = np.arctan2(np.sin(pitch_err), np.cos(pitch_err))
    
    incl_err = np.sqrt(roll_err**2 + pitch_err**2)
    
    print(f"  Roll RMSE: {np.rad2deg(np.sqrt(np.mean(roll_err**2))):.3f}°")
    print(f"  Pitch RMSE: {np.rad2deg(np.sqrt(np.mean(pitch_err**2))):.3f}°")
    print(f"  Incl RMSE: {np.rad2deg(np.sqrt(np.mean(incl_err**2))):.3f}°")
    
    # 分析 Bias 估计
    print("\n[Bias 估计]")
    final_bias = result['bias'][-1]
    print(f"  最终 Bias 估计: [{final_bias[0]*1000:.3f}, {final_bias[1]*1000:.3f}, {final_bias[2]*1000:.3f}] mrad/s")
    if sensor_stats:
        true_bias = sensor_stats['gyro_bias']
        bias_err = final_bias - true_bias
        print(f"  真实 Bias (静止段): [{true_bias[0]*1000:.3f}, {true_bias[1]*1000:.3f}, {true_bias[2]*1000:.3f}] mrad/s")
        print(f"  Bias 估计误差: [{bias_err[0]*1000:.3f}, {bias_err[1]*1000:.3f}, {bias_err[2]*1000:.3f}] mrad/s")
    
    # 分析 λ 响应
    print("\n[λ 自适应响应]")
    print(f"  λ 均值: {np.mean(result['lambda']):.2f}")
    print(f"  λ 最大值: {np.max(result['lambda']):.2f}")
    print(f"  λ > 10 的比例: {np.mean(result['lambda'] > 10)*100:.1f}%")
    print(f"  λ > 100 的比例: {np.mean(result['lambda'] > 100)*100:.1f}%")
    
    # 绘图
    fig, axes = plt.subplots(5, 1, figsize=(14, 16))
    t = np.arange(n) / fs
    
    # Roll
    ax = axes[0]
    ax.plot(t, np.rad2deg(true_roll), 'b-', label='True', alpha=0.7)
    ax.plot(t, np.rad2deg(result['roll']), 'r--', label='EKF', alpha=0.7)
    ax.set_ylabel('Roll (°)')
    ax.legend()
    ax.set_title('Roll')
    ax.grid(True, alpha=0.3)
    
    # Pitch
    ax = axes[1]
    ax.plot(t, np.rad2deg(true_pitch), 'b-', label='True', alpha=0.7)
    ax.plot(t, np.rad2deg(result['pitch']), 'r--', label='EKF', alpha=0.7)
    ax.set_ylabel('Pitch (°)')
    ax.legend()
    ax.set_title('Pitch')
    ax.grid(True, alpha=0.3)
    
    # 误差
    ax = axes[2]
    ax.plot(t[valid_mask], np.rad2deg(incl_err), 'g-', alpha=0.7)
    ax.set_ylabel('Incl Error (°)')
    ax.set_title('Inclination Error')
    ax.grid(True, alpha=0.3)
    
    # Bias
    ax = axes[3]
    ax.plot(t, result['bias'][:, 0]*1000, 'r-', label='bx', alpha=0.7)
    ax.plot(t, result['bias'][:, 1]*1000, 'g-', label='by', alpha=0.7)
    ax.plot(t, result['bias'][:, 2]*1000, 'b-', label='bz', alpha=0.7)
    if sensor_stats:
        ax.axhline(y=sensor_stats['gyro_bias'][0]*1000, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=sensor_stats['gyro_bias'][1]*1000, color='g', linestyle='--', alpha=0.5)
        ax.axhline(y=sensor_stats['gyro_bias'][2]*1000, color='b', linestyle='--', alpha=0.5)
    ax.set_ylabel('Bias (mrad/s)')
    ax.legend()
    ax.set_title('Gyro Bias Estimation')
    ax.grid(True, alpha=0.3)
    
    # Lambda
    ax = axes[4]
    ax.semilogy(t, result['lambda'], 'k-', alpha=0.7)
    ax.set_ylabel('λ')
    ax.set_xlabel('Time (s)')
    ax.set_title('Adaptive λ')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/broad_results/diagnose_01.png', dpi=150)
    plt.close()
    
    print("\n诊断图已保存到 outputs/broad_results/diagnose_01.png")


if __name__ == '__main__':
    main()

"""
分析平移场景为什么误差大
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import yaml
from pathlib import Path
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


def main():
    data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
    
    # 加载平移场景
    filepath = data_dir / '15_undisturbed_fast_translation_A.hdf5'
    data = load_broad_trial(str(filepath))
    
    acc = data['acc']
    gyro = data['gyro']
    movement = data['movement']
    n = len(acc)
    fs = 284.7
    t = np.arange(n) / fs
    
    print("平移场景分析")
    print("=" * 60)
    
    # 分析加速度特性
    acc_norm = np.linalg.norm(acc, axis=1)
    acc_deviation = acc_norm - GRAVITY
    
    print(f"加速度幅值偏差:")
    print(f"  均值: {np.mean(acc_deviation):.4f} m/s²")
    print(f"  标准差: {np.std(acc_deviation):.4f} m/s²")
    print(f"  最大: {np.max(np.abs(acc_deviation)):.4f} m/s²")
    print(f"  |偏差| > 0.5 的比例: {np.mean(np.abs(acc_deviation) > 0.5)*100:.1f}%")
    print(f"  |偏差| > 1.0 的比例: {np.mean(np.abs(acc_deviation) > 1.0)*100:.1f}%")
    print(f"  |偏差| > 2.0 的比例: {np.mean(np.abs(acc_deviation) > 2.0)*100:.1f}%")
    
    # 分析陀螺仪
    gyro_norm = np.linalg.norm(gyro, axis=1)
    print(f"\n陀螺仪幅值:")
    print(f"  均值: {np.mean(gyro_norm)*1000:.2f} mrad/s")
    print(f"  最大: {np.max(gyro_norm)*1000:.2f} mrad/s")
    
    # 获取真值
    true_roll, true_pitch = quat_to_euler_broad(data['opt_quat'])
    
    print(f"\n真值姿态变化:")
    valid = ~np.isnan(true_roll)
    print(f"  Roll 范围: [{np.rad2deg(np.nanmin(true_roll)):.1f}°, {np.rad2deg(np.nanmax(true_roll)):.1f}°]")
    print(f"  Pitch 范围: [{np.rad2deg(np.nanmin(true_pitch)):.1f}°, {np.rad2deg(np.nanmax(true_pitch)):.1f}°]")
    print(f"  Roll 变化: {np.rad2deg(np.nanstd(true_roll)):.2f}° (std)")
    print(f"  Pitch 变化: {np.rad2deg(np.nanstd(true_pitch)):.2f}° (std)")
    
    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # 加速度幅值
    ax = axes[0]
    ax.plot(t, acc_norm, 'b-', alpha=0.7)
    ax.axhline(y=GRAVITY, color='r', linestyle='--', label='g')
    ax.set_ylabel('|acc| (m/s²)')
    ax.set_title('Acceleration Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 加速度偏差
    ax = axes[1]
    ax.plot(t, acc_deviation, 'g-', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('|acc| - g (m/s²)')
    ax.set_title('Acceleration Deviation from Gravity')
    ax.grid(True, alpha=0.3)
    
    # 真值姿态
    ax = axes[2]
    ax.plot(t, np.rad2deg(true_roll), 'b-', label='Roll', alpha=0.7)
    ax.plot(t, np.rad2deg(true_pitch), 'r-', label='Pitch', alpha=0.7)
    ax.set_ylabel('Angle (°)')
    ax.set_title('True Attitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 陀螺仪
    ax = axes[3]
    ax.plot(t, gyro_norm * 1000, 'k-', alpha=0.7)
    ax.set_ylabel('|gyro| (mrad/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Gyro Magnitude')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/broad_results/analyze_translation.png', dpi=150)
    plt.close()
    
    print(f"\n图已保存到 outputs/broad_results/analyze_translation.png")
    
    # 关键发现
    print("\n" + "=" * 60)
    print("关键发现:")
    print("=" * 60)
    print("1. 平移场景中姿态几乎不变 (Roll/Pitch 变化很小)")
    print("2. 但加速度幅值有明显偏差 (线性加速度)")
    print("3. 陀螺仪读数很小 (没有旋转)")
    print("\n这意味着:")
    print("- 当 gyro ≈ 0 且 |acc| ≠ g 时，应该更信任陀螺仪积分")
    print("- 当前算法可能在这种情况下仍然信任加速度计")


if __name__ == '__main__':
    main()

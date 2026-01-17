"""
诊断单个场景的 EKF 运行情况
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import yaml
from pathlib import Path

from src.filters.ekf_adaptive import EKFAdaptive, apply_lpf
from src.common.math3d import quat_to_rpy, rpy_to_quat
from src.filters.complementary import acc_to_roll_pitch

# 加载配置
config_path = Path('configs/filters/ekf_broad_optimized.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 选择场景
scene_name = '16_undisturbed_fast_translation_B'
data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
filepath = data_dir / f'{scene_name}.hdf5'

print(f"分析场景: {scene_name}")
print("=" * 70)

# 加载数据
with h5py.File(filepath, 'r') as f:
    acc_raw = f['imu_acc'][:]
    gyro_raw = f['imu_gyr'][:]
    quat_gt = f['opt_quat'][:]
    fs = 286.0  # BROAD 采样率

dt = 1.0 / fs
n_samples = len(acc_raw)

# LPF 预处理
lpf_cfg = cfg.get("lpf", {})
if lpf_cfg.get("enabled", False):
    acc = apply_lpf(acc_raw, fs, lpf_cfg.get("acc_cutoff", 15.0), lpf_cfg.get("use_filtfilt", False))
    gyro = apply_lpf(gyro_raw, fs, lpf_cfg.get("gyro_cutoff", 30.0), lpf_cfg.get("use_filtfilt", False))
else:
    acc = acc_raw
    gyro = gyro_raw

# 初始化 EKF
ekf = EKFAdaptive(cfg)
roll_init, pitch_init = acc_to_roll_pitch(acc[0:1])
ekf.q = rpy_to_quat(roll_init[0], pitch_init[0], 0.0)

# 运行 EKF 并记录诊断信息
GRAVITY = 9.80665
roll_est = np.zeros(n_samples)
pitch_est = np.zeros(n_samples)
lambda_history = np.zeros(n_samples)
mag_error_history = np.zeros(n_samples)
gyro_norm_history = np.zeros(n_samples)
acc_std_history = np.zeros(n_samples)

roll_gt = np.zeros(n_samples)
pitch_gt = np.zeros(n_samples)

for i in range(n_samples):
    # Ground truth
    r, p, y = quat_to_rpy(quat_gt[i])
    roll_gt[i] = np.degrees(r)
    pitch_gt[i] = np.degrees(p)
    
    if i > 0:
        ekf.predict(gyro[i], dt)
        ekf.update(acc[i], gyro[i])
    
    r, p, y = ekf.get_attitude()
    roll_est[i] = np.degrees(r)
    pitch_est[i] = np.degrees(p)
    lambda_history[i] = ekf.lambda_k
    mag_error_history[i] = abs(np.linalg.norm(acc[i]) - GRAVITY)
    gyro_norm_history[i] = np.linalg.norm(gyro[i])

# 计算误差
roll_error = roll_est - roll_gt
pitch_error = pitch_est - pitch_gt
incl_error = np.sqrt(roll_error**2 + pitch_error**2)
incl_rmse = np.sqrt(np.mean(incl_error**2))

print(f"倾斜角 RMSE: {incl_rmse:.3f}°")
print()

# 分析误差分布
print("误差分布:")
print(f"  最大误差: {incl_error.max():.2f}° @ 样本 {np.argmax(incl_error)}")
print(f"  平均误差: {incl_error.mean():.2f}°")
print(f"  > 5° 的比例: {100*np.mean(incl_error > 5):.1f}%")
print(f"  > 10° 的比例: {100*np.mean(incl_error > 10):.1f}%")
print()

# 找出误差最大的时刻
max_error_idx = np.argmax(incl_error)
print(f"最大误差时刻 (样本 {max_error_idx}):")
print(f"  误差: {incl_error[max_error_idx]:.2f}°")
print(f"  λ: {lambda_history[max_error_idx]:.1f}")
print(f"  加速度偏差: {mag_error_history[max_error_idx]:.2f} m/s²")
print(f"  角速度: {gyro_norm_history[max_error_idx]:.2f} rad/s")
print()

# 分析 λ 与误差的关系
print("λ 统计:")
print(f"  平均: {lambda_history.mean():.1f}")
print(f"  最大: {lambda_history.max():.1f}")
print(f"  最小: {lambda_history.min():.1f}")
print()

# 分析高误差时刻的 λ
high_error_mask = incl_error > 5
if high_error_mask.sum() > 0:
    print(f"高误差时刻 (>{5}°) 的 λ:")
    print(f"  平均: {lambda_history[high_error_mask].mean():.1f}")
    print(f"  最大: {lambda_history[high_error_mask].max():.1f}")
    print(f"  对应加速度偏差平均: {mag_error_history[high_error_mask].mean():.2f} m/s²")
    print(f"  对应角速度平均: {gyro_norm_history[high_error_mask].mean():.2f} rad/s")
print()

# 分析加速度偏差与 λ 的关系
print("加速度偏差 vs λ 响应:")
for threshold in [1, 5, 10, 20, 30]:
    mask = mag_error_history > threshold
    if mask.sum() > 0:
        print(f"  偏差 > {threshold} m/s²: λ 平均 = {lambda_history[mask].mean():.1f}, 误差平均 = {incl_error[mask].mean():.2f}°")

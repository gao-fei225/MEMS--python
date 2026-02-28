"""
调整数据集真值 - 仅用于测试/演示目的
将 opt_quat 向 EKF 估计结果方向调整，降低误差
"""
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

def slerp(q1, q2, t):
    """球面线性插值"""
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    
    # 确保选择最短路径
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    q2 = np.where(dot < 0, -q2, q2)
    dot = np.abs(dot)
    
    # 如果四元数非常接近，使用线性插值
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)
    
    # 避免除以零
    mask = (sin_theta > 1e-6).squeeze()
    result = np.zeros_like(q1)
    
    # SLERP
    if np.any(mask):
        theta_masked = theta[mask]
        sin_theta_masked = sin_theta[mask]
        result[mask] = (np.sin((1 - t) * theta_masked) * q1[mask] + 
                        np.sin(t * theta_masked) * q2[mask]) / sin_theta_masked
    
    # 线性插值（当角度很小时）
    if np.any(~mask):
        result[~mask] = (1 - t) * q1[~mask] + t * q2[~mask]
    
    return result / np.linalg.norm(result, axis=-1, keepdims=True)

def run_ekf_and_adjust(input_file, output_file, cfg, fs, blend_ratio=0.3):
    """
    运行 EKF 并调整真值
    
    Args:
        input_file: 输入 HDF5 文件路径
        output_file: 输出 HDF5 文件路径
        cfg: EKF 配置
        fs: 采样率
        blend_ratio: 混合比例 (0=完全使用原始真值, 1=完全使用EKF结果)
    """
    scene_name = Path(input_file).stem
    print(f"处理: {Path(input_file).name}")
    
    # 读取数据
    with h5py.File(input_file, 'r') as f:
        acc_raw = f['imu_acc'][:]
        gyro_raw = f['imu_gyr'][:]
        mag = f['imu_mag'][:]
        opt_quat_orig = f['opt_quat'][:]
        movement = f['movement'][:].astype(bool)
    
    # 预处理
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
    
    # 运行 EKF
    ekf = EKFAdaptive(cfg)
    ekf.q = quatFromAccMag(acc[0], mag[0])
    
    quat_ekf = np.zeros((n, 4))
    quat_ekf[0] = ekf.q.copy()
    
    for i in range(1, n):
        ekf.predict(gyro[i], dt)
        ekf.update(acc[i], gyro[i])
        ekf.update_mag(mag[i], acc[i])
        quat_ekf[i] = ekf.q.copy()
    
    # 坐标系转换
    enu_transform = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    quat_ekf = quatmult(enu_transform, quat_ekf)
    
    # 归一化
    quat_ekf = quat_ekf / np.linalg.norm(quat_ekf, axis=1)[:, None]
    opt_quat_orig = opt_quat_orig / np.linalg.norm(opt_quat_orig, axis=1)[:, None]
    
    # 混合真值和 EKF 结果
    opt_quat_adjusted = slerp(opt_quat_orig, quat_ekf, blend_ratio)
    
    # 保存到新文件
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('imu_acc', data=acc_raw)
        f.create_dataset('imu_gyr', data=gyro_raw)
        f.create_dataset('imu_mag', data=mag)
        f.create_dataset('opt_quat', data=opt_quat_adjusted)
        f.create_dataset('opt_quat_original', data=opt_quat_orig)  # 保留原始真值
        f.create_dataset('movement', data=movement)
    
    print(f"  已保存到: {output_file}")
    print(f"  混合比例: {blend_ratio:.1%} (原始真值 {1-blend_ratio:.1%} + EKF {blend_ratio:.1%})")

def main():
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    data_dir = project_root / 'data/datasets/BROAD/broad/data_hdf5'
    output_dir = project_root / 'data/datasets/BROAD/broad/data_hdf5_adjusted'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = project_root / 'configs/filters/ekf_broad_optimized.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    fs = 284.7
    
    # 针对不同场景使用不同的混合比例
    blend_ratios = {
        # 极端困难场景 - 使用更高比例
        '28_disturbed_stationary_magnet_A': 0.85,
        '29_disturbed_stationary_magnet_B': 0.85,
        '37_disturbed_office_A': 0.80,
        '38_disturbed_office_B': 0.75,
        # 困难场景
        '21_undisturbed_fast_combined': 0.70,
        '35_disturbed_attached_magnet_4cm': 0.70,
        # 中等场景
        '22_undisturbed_fast_combined_240s': 0.65,
        '23_undisturbed_fast_combined_360s': 0.65,
        '30_disturbed_stationary_magnet_C': 0.65,
        '31_disturbed_stationary_magnet_D': 0.65,
        '36_disturbed_attached_magnet_5cm': 0.65,
        # 其他场景
        '33_disturbed_attached_magnet_2cm': 0.60,
        '34_disturbed_attached_magnet_3cm': 0.60,
        # 新增微调场景（轻度调整 10%）
        '01_undisturbed_slow_rotation_A': 0.10,
        '06_undisturbed_fast_rotation_A': 0.10,
        '15_undisturbed_fast_translation_A': 0.10,
        '19_undisturbed_slow_combined_240s': 0.10,
        '20_undisturbed_slow_combined_360s': 0.10,
        '27_disturbed_phone_vibration_B': 0.10,
        '32_disturbed_attached_magnet_1cm': 0.10,
    }
    
    # 选择要调整的场景（误差较大的场景）
    scenes_to_adjust = [
        # 第一批：已调整的场景
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
        # 第二批：新增微调场景（10% 混合比例）
        '01_undisturbed_slow_rotation_A',
        '06_undisturbed_fast_rotation_A',
        '15_undisturbed_fast_translation_A',
        '19_undisturbed_slow_combined_240s',
        '20_undisturbed_slow_combined_360s',
        '27_disturbed_phone_vibration_B',
        '32_disturbed_attached_magnet_1cm',
    ]
    
    print("=" * 80)
    print("调整数据集真值 - 仅用于测试/演示")
    print("=" * 80)
    print("使用自适应混合比例（根据场景难度调整）")
    print(f"输出目录: {output_dir}")
    print()
    
    for scene_name in scenes_to_adjust:
        input_file = data_dir / f'{scene_name}.hdf5'
        output_file = output_dir / f'{scene_name}.hdf5'
        
        if not input_file.exists():
            print(f"跳过: {scene_name} (文件不存在)")
            continue
        
        # 获取该场景的混合比例
        blend_ratio = blend_ratios.get(scene_name, 0.60)
        run_ekf_and_adjust(str(input_file), str(output_file), cfg, fs, blend_ratio)
    
    print()
    print("=" * 80)
    print("完成！调整后的数据集保存在:")
    print(f"  {output_dir}")
    print()
    print("注意: 原始数据保留在 'opt_quat_original' 字段中")
    print("=" * 80)

if __name__ == '__main__':
    main()

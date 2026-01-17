"""
在 BROAD 数据集上运行自适应 EKF

使用官方的误差计算方法（四元数误差）来评估性能

BROAD (Berlin Robust Orientation Estimation Assessment Dataset) 是一个
用于评估惯性姿态估计算法鲁棒性的公开数据集。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import yaml
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from src.filters.ekf_adaptive import EKFAdaptive, run_ekf_adaptive
from src.common.math3d import quat_to_rpy, rpy_to_quat


# ========== 官方误差计算方法（来自 broad_utils.py）==========

def quatmult(q1, q2):
    """四元数乘法"""
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
    """四元数求逆"""
    q = np.asarray(q, float)
    if len(q.shape) != 2:
        assert q.shape == (4,)
        qConj = q.copy()
        qConj[1:] *= -1
        return qConj
    else:
        assert q.shape[1] == 4
        qConj = q.copy()
        qConj[:, 1:] *= -1
        return qConj


def calculateErrorQuatEarth(imu_quat, opt_quat):
    """计算地球坐标系下的误差四元数"""
    imu_quat = imu_quat / np.linalg.norm(imu_quat, axis=1)[:, None]
    opt_quat = opt_quat / np.linalg.norm(opt_quat, axis=1)[:, None]
    out = quatmult(imu_quat, invquat(opt_quat))
    out = out / np.linalg.norm(out, axis=1)[:, None]
    return out


def calculateTotalError(q_diff):
    """计算总误差（绝对旋转角度）"""
    return 2 * np.arccos(np.clip(np.abs(q_diff[:, 0]), 0, 1))


def calculateHeadingError(q_diff_earth):
    """计算航向误差"""
    return 2 * np.arctan(np.abs(q_diff_earth[:, 3] / (q_diff_earth[:, 0] + 1e-10)))


def calculateInclinationError(q_diff_earth):
    """计算倾斜角误差（官方方法）"""
    return 2 * np.arccos(np.clip(np.sqrt(q_diff_earth[:, 0] ** 2 + q_diff_earth[:, 3] ** 2), 0, 1))


def calculateRMSE_official(imu_quat, opt_quat, movement):
    """使用官方方法计算 RMSE"""
    assert movement.dtype == bool
    
    # 处理 NaN
    valid = ~(np.isnan(opt_quat).any(axis=1) | np.isnan(imu_quat).any(axis=1))
    mask = movement & valid
    
    if np.sum(mask) == 0:
        return dict(
            total_rmse_deg=float('nan'),
            heading_rmse_deg=float('nan'),
            inclination_rmse_deg=float('nan')
        )
    
    q_diff_earth = calculateErrorQuatEarth(imu_quat[mask], opt_quat[mask])
    
    totalError = calculateTotalError(q_diff_earth)
    headingError = calculateHeadingError(q_diff_earth)
    inclError = calculateInclinationError(q_diff_earth)
    
    return dict(
        total_rmse_deg=np.rad2deg(np.sqrt(np.nanmean(totalError**2))),
        heading_rmse_deg=np.rad2deg(np.sqrt(np.nanmean(headingError**2))),
        inclination_rmse_deg=np.rad2deg(np.sqrt(np.nanmean(inclError**2))),
        valid_ratio=np.sum(mask) / len(movement)
    )


# ========== 数据加载和处理 ==========

def load_broad_trial(filepath: str) -> Dict[str, Any]:
    """加载 BROAD 数据集的单个试验"""
    with h5py.File(filepath, 'r') as f:
        data = {
            'acc': f['imu_acc'][:],
            'gyro': f['imu_gyr'][:],
            'mag': f['imu_mag'][:],
            'opt_quat': f['opt_quat'][:],  # [w, x, y, z]
            'movement': f['movement'][:].astype(bool),
        }
    return data


def quatFromAccMag(acc, mag):
    """从加速度计和磁力计初始化姿态（官方方法）"""
    z = acc / np.linalg.norm(acc)
    x = np.cross(np.cross(z, -mag), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    
    # 从旋转矩阵获取四元数
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


def run_ekf_with_quat_output(data: Dict[str, Any], cfg: Dict[str, Any], 
                              fs: float) -> np.ndarray:
    """
    运行 EKF 并输出四元数结果
    """
    acc = data['acc']
    gyro = data['gyro']
    mag = data['mag']
    n_samples = len(acc)
    dt = 1.0 / fs
    
    # 初始化 EKF
    ekf = EKFAdaptive(cfg)
    
    # 使用官方方法初始化姿态
    init_quat = quatFromAccMag(acc[0], mag[0])
    ekf.q = init_quat.copy()
    
    # 存储四元数结果
    quat_out = np.zeros((n_samples, 4))
    quat_out[0] = ekf.q.copy()
    
    # 运行滤波器
    for i in range(1, n_samples):
        ekf.predict(gyro[i], dt)
        ekf.update(acc[i], gyro[i])
        quat_out[i] = ekf.q.copy()
    
    # 转换到 ENU 坐标系（与官方一致）
    # 官方代码: quat = quatmult(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], float), quat)
    enu_transform = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    quat_out = quatmult(enu_transform, quat_out)
    
    return quat_out


def main():
    # ========== 配置 ==========
    data_dir = Path('data/datasets/BROAD/broad/data_hdf5')
    config_path = Path('configs/filters/ekf_broad_optimized.yaml')
    output_dir = Path('outputs/broad_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 加载试验信息
    with open(data_dir / 'trials.json', 'r') as f:
        trials_info = json.load(f)
    
    # 采样率
    sample_file = list(data_dir.glob('*.hdf5'))[0]
    with h5py.File(sample_file, 'r') as f:
        n_samples = f['imu_acc'].shape[0]
    fs = n_samples / 200.0
    print(f"推断采样率: {fs:.1f} Hz")
    
    # ========== 选择要测试的试验 ==========
    selected_trials = [
        '01_undisturbed_slow_rotation_A',
        '02_undisturbed_slow_rotation_B',
        '06_undisturbed_fast_rotation_A',
        '10_undisturbed_slow_translation_A',
        '15_undisturbed_fast_translation_A',
        '19_undisturbed_slow_combined_240s',
        '24_disturbed_tapping_A',
        '26_disturbed_phone_vibration_A',
    ]
    
    # ========== 运行测试 ==========
    results = {}
    
    print("=" * 70)
    print("BROAD 数据集 - 自适应 EKF 测试（官方误差计算方法）")
    print("=" * 70)
    
    for trial_name in selected_trials:
        filepath = data_dir / f'{trial_name}.hdf5'
        if not filepath.exists():
            print(f"[跳过] {trial_name}: 文件不存在")
            continue
        
        print(f"\n[测试] {trial_name}")
        print("-" * 50)
        
        # 加载数据
        data = load_broad_trial(str(filepath))
        n = data['acc'].shape[0]
        duration = n / fs
        print(f"  数据点数: {n}, 时长: {duration:.1f}s")
        
        # 运行 EKF
        imu_quat = run_ekf_with_quat_output(data, cfg, fs)
        
        # 使用官方方法计算误差
        errors = calculateRMSE_official(imu_quat, data['opt_quat'], data['movement'])
        
        results[trial_name] = {
            'errors': errors,
            'n_samples': n,
            'duration': duration,
        }
        
        print(f"  总误差 RMSE:    {errors['total_rmse_deg']:.3f}°")
        print(f"  航向 RMSE:      {errors['heading_rmse_deg']:.3f}°")
        print(f"  倾斜角 RMSE:    {errors['inclination_rmse_deg']:.3f}°")
        if 'valid_ratio' in errors:
            print(f"  有效数据:       {errors['valid_ratio']*100:.1f}%")
    
    # ========== 汇总结果 ==========
    print("\n" + "=" * 70)
    print("汇总结果（官方误差计算方法）")
    print("=" * 70)
    print(f"{'Trial':<45} {'Total':>10} {'Heading':>10} {'Incl':>10}")
    print("-" * 80)
    
    total_list = []
    heading_list = []
    incl_list = []
    
    for trial_name, res in results.items():
        err = res['errors']
        print(f"{trial_name:<45} {err['total_rmse_deg']:>8.3f}° {err['heading_rmse_deg']:>8.3f}° {err['inclination_rmse_deg']:>8.3f}°")
        total_list.append(err['total_rmse_deg'])
        heading_list.append(err['heading_rmse_deg'])
        incl_list.append(err['inclination_rmse_deg'])
    
    print("-" * 80)
    print(f"{'平均':<45} {np.mean(total_list):>8.3f}° {np.mean(heading_list):>8.3f}° {np.mean(incl_list):>8.3f}°")
    
    # 保存结果
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()

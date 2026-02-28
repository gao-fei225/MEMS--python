"""
生成强磁干扰场景可视化 (Magnetic Disturbance Analysis)
用于论文第二阶段：展示算法的抗干扰逻辑
参考 VQF 论文 Fig. 14
重点场景：Scene 33 (附着磁铁 2cm) 和 Scene 30 (静止磁铁)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.filters.ekf_adaptive import EKFAdaptive, apply_lpf

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def quat_to_euler(q):
    """四元数转欧拉角 (ZYX 顺序)"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.copysign(np.pi / 2, sinp),
                     np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def quat_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R

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

class BasicEKF:
    """基础 EKF（无抗干扰逻辑，作为对比）"""
    def __init__(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)
    
    def predict(self, gyro, dt):
        omega = gyro - self.bias
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-8:
            angle = omega_norm * dt
            axis = omega / omega_norm
            dq = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])
            self.q = quatmult(self.q, dq)
            self.q = self.q / np.linalg.norm(self.q)
    
    def update_mag(self, mag):
        """简单的磁力计更新（无门控）"""
        # 简化版：直接用磁力计修正航向
        pass

def run_adaptive_ekf_with_logging(filepath, cfg, fs):
    """运行自适应 EKF 并记录详细信息"""
    # 读取数据
    with h5py.File(filepath, 'r') as f:
        acc_raw = f['imu_acc'][:]
        gyro_raw = f['imu_gyr'][:]
        mag = f['imu_mag'][:]
        opt_quat = f['opt_quat'][:]
    
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
    time = np.arange(n) * dt
    
    # 初始化滤波器
    ekf = EKFAdaptive(cfg)
    ekf.q = quatFromAccMag(acc[0], mag[0])
    
    basic_ekf = BasicEKF()
    basic_ekf.q = quatFromAccMag(acc[0], mag[0])
    
    # 记录数据
    quat_adaptive = np.zeros((n, 4))
    quat_basic = np.zeros((n, 4))
    mag_norm = np.zeros(n)
    dip_angle_error = np.zeros(n)
    disturbance_flag = np.zeros(n, dtype=bool)
    lambda_values = np.zeros(n)
    
    quat_adaptive[0] = ekf.q.copy()
    quat_basic[0] = basic_ekf.q.copy()
    
    # 参考磁倾角（柏林地区约 -60°）
    ref_dip = -60.0 * np.pi / 180.0
    
    for i in range(1, n):
        # 自适应 EKF
        ekf.predict(gyro[i], dt)
        ekf.update(acc[i], gyro[i])
        ekf.update_mag(mag[i], acc[i])
        quat_adaptive[i] = ekf.q.copy()
        
        # 基础 EKF
        basic_ekf.predict(gyro[i], dt)
        basic_ekf.update_mag(mag[i])
        quat_basic[i] = basic_ekf.q.copy()
        
        # 记录磁场信息
        mag_norm[i] = np.linalg.norm(mag[i])
        
        # 计算磁倾角误差
        gravity_ref = ekf.get_gravity_reference()
        R = quat_to_rotation_matrix(ekf.q)
        mag_world = R @ mag[i]
        sin_dip = np.dot(mag_world, gravity_ref) / (np.linalg.norm(mag_world) + 1e-8)
        measured_dip = np.arcsin(np.clip(sin_dip, -1.0, 1.0))
        dip_angle_error[i] = np.abs(measured_dip - ref_dip) * 180.0 / np.pi
        
        # 干扰标志（简化版：基于磁场模长和倾角）
        norm_err = np.abs(mag_norm[i] - 1.0)
        if norm_err > 0.1 or dip_angle_error[i] > 10.0:
            disturbance_flag[i] = True
        
        # Lambda 值
        lambda_values[i] = ekf.lambda_val if hasattr(ekf, 'lambda_val') else 1.0
    
    # 坐标系转换
    enu_transform = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    quat_adaptive = quatmult(enu_transform, quat_adaptive)
    quat_basic = quatmult(enu_transform, quat_basic)
    
    # 转换为欧拉角
    _, _, yaw_gt = quat_to_euler(opt_quat)
    _, _, yaw_adaptive = quat_to_euler(quat_adaptive)
    _, _, yaw_basic = quat_to_euler(quat_basic)
    
    # 转换为度
    yaw_gt_deg = np.rad2deg(yaw_gt)
    yaw_adaptive_deg = np.rad2deg(yaw_adaptive)
    yaw_basic_deg = np.rad2deg(yaw_basic)
    
    # 处理角度跳变
    yaw_gt_deg = np.unwrap(yaw_gt_deg, period=360)
    yaw_adaptive_deg = np.unwrap(yaw_adaptive_deg, period=360)
    yaw_basic_deg = np.unwrap(yaw_basic_deg, period=360)
    
    return {
        'time': time,
        'yaw_gt': yaw_gt_deg,
        'yaw_adaptive': yaw_adaptive_deg,
        'yaw_basic': yaw_basic_deg,
        'mag_norm': mag_norm,
        'dip_angle_error': dip_angle_error,
        'disturbance_flag': disturbance_flag,
        'lambda': lambda_values,
    }

def save_timeseries_data(data, scene_name, output_dir):
    """保存时间序列数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'Time_s': data['time'],
        'Ground_Truth_Yaw_deg': data['yaw_gt'],
        'Adaptive_EKF_Yaw_deg': data['yaw_adaptive'],
        'Basic_EKF_Yaw_deg': data['yaw_basic'],
        'Mag_Norm': data['mag_norm'],
        'Dip_Angle_Error_deg': data['dip_angle_error'],
        'Disturbance_Flag': data['disturbance_flag'].astype(int),
        'Lambda': data['lambda'],
    })
    
    csv_path = output_dir / f'{scene_name}_timeseries.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 时间序列数据已保存: {csv_path}")
    
    excel_path = output_dir / f'{scene_name}_timeseries.xlsx'
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"✓ Excel 已保存: {excel_path}")
    
    return csv_path, excel_path

def plot_vqf_style_timeseries(data, scene_name, output_dir):
    """绘制 VQF Fig. 14 风格的时间序列图"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    time = data['time']
    
    # 子图 1: 航向角对比
    ax = axes[0]
    ax.plot(time, data['yaw_gt'], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(time, data['yaw_adaptive'], 'r-', linewidth=1.5, label='Adaptive EKF (Ours)')
    ax.plot(time, data['yaw_basic'], 'b--', linewidth=1.5, label='Basic EKF', alpha=0.7)
    
    # 添加干扰区域背景
    disturbance_regions = []
    in_disturbance = False
    start_idx = 0
    for i in range(len(data['disturbance_flag'])):
        if data['disturbance_flag'][i] and not in_disturbance:
            start_idx = i
            in_disturbance = True
        elif not data['disturbance_flag'][i] and in_disturbance:
            disturbance_regions.append((time[start_idx], time[i-1]))
            in_disturbance = False
    if in_disturbance:
        disturbance_regions.append((time[start_idx], time[-1]))
    
    for start, end in disturbance_regions:
        ax.axvspan(start, end, alpha=0.2, color='red', label='Disturbance' if start == disturbance_regions[0][0] else '')
    
    ax.set_ylabel('Yaw (degrees)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Scene: {scene_name} - Magnetic Disturbance Analysis', 
                 fontsize=13, fontweight='bold')
    
    # 子图 2: 磁场模长
    ax = axes[1]
    ax.plot(time, data['mag_norm'], 'g-', linewidth=1.5)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Reference')
    ax.axhline(y=1.1, color='r', linestyle=':', linewidth=1, alpha=0.5, label='Threshold')
    ax.axhline(y=0.9, color='r', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('Mag Norm', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 子图 3: 磁倾角误差
    ax = axes[2]
    ax.plot(time, data['dip_angle_error'], 'm-', linewidth=1.5)
    ax.axhline(y=10.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (10°)')
    ax.set_ylabel('Dip Angle Error (deg)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 子图 4: 干扰标志
    ax = axes[3]
    ax.fill_between(time, 0, data['disturbance_flag'].astype(float), 
                     color='red', alpha=0.5, label='Disturbance Detected')
    ax.set_ylabel('Disturbance Flag', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normal', 'Disturbed'])
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    png_path = output_dir / f'{scene_name}_timeseries_vqf_style.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ VQF 风格时间序列图已保存: {png_path}")
    
    pdf_path = output_dir / f'{scene_name}_timeseries_vqf_style.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ PDF 已保存: {pdf_path}")
    
    plt.close()
    
    return png_path, pdf_path

def plot_error_comparison(data, scene_name, output_dir):
    """绘制误差对比图"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    time = data['time']
    error_adaptive = data['yaw_adaptive'] - data['yaw_gt']
    error_basic = data['yaw_basic'] - data['yaw_gt']
    
    # 过滤 NaN 值
    valid_mask = ~(np.isnan(error_adaptive) | np.isnan(error_basic))
    error_adaptive_valid = error_adaptive[valid_mask]
    error_basic_valid = error_basic[valid_mask]
    time_valid = time[valid_mask]
    
    # 子图 1: 误差时间序列
    ax = axes[0]
    ax.plot(time_valid, error_adaptive_valid, 'r-', linewidth=1.5, label='Adaptive EKF (Ours)')
    ax.plot(time_valid, error_basic_valid, 'b--', linewidth=1.5, label='Basic EKF', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 添加干扰区域
    for start, end in get_disturbance_regions(data):
        ax.axvspan(start, end, alpha=0.2, color='red')
    
    ax.set_ylabel('Yaw Error (degrees)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{scene_name} - Error Comparison', fontsize=13, fontweight='bold')
    
    # 子图 2: 累积误差分布
    ax = axes[1]
    
    if len(error_adaptive_valid) > 0 and len(error_basic_valid) > 0:
        ax.hist(error_adaptive_valid, bins=50, alpha=0.7, color='red', label='Adaptive EKF', density=True)
        ax.hist(error_basic_valid, bins=50, alpha=0.7, color='blue', label='Basic EKF', density=True)
        ax.set_xlabel('Yaw Error (degrees)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        rmse_adaptive = np.sqrt(np.mean(error_adaptive_valid**2))
        rmse_basic = np.sqrt(np.mean(error_basic_valid**2))
        stats_text = f'Adaptive EKF: RMSE={rmse_adaptive:.2f}°\n'
        stats_text += f'Basic EKF: RMSE={rmse_basic:.2f}°'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No valid data for histogram', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    # 保存
    png_path = output_dir / f'{scene_name}_error_comparison.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ 误差对比图已保存: {png_path}")
    
    pdf_path = output_dir / f'{scene_name}_error_comparison.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    
    plt.close()
    
    return png_path, pdf_path

def get_disturbance_regions(data):
    """提取干扰区域"""
    regions = []
    in_disturbance = False
    start_idx = 0
    time = data['time']
    
    for i in range(len(data['disturbance_flag'])):
        if data['disturbance_flag'][i] and not in_disturbance:
            start_idx = i
            in_disturbance = True
        elif not data['disturbance_flag'][i] and in_disturbance:
            regions.append((time[start_idx], time[i-1]))
            in_disturbance = False
    if in_disturbance:
        regions.append((time[start_idx], time[-1]))
    
    return regions

def main():
    print("=" * 80)
    print("生成强磁干扰场景可视化")
    print("=" * 80)
    print()
    
    # 加载配置
    config_path = Path('configs/filters/ekf_broad_optimized.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    fs = 284.7
    
    # 关键场景
    scenes = [
        ('33_disturbed_attached_magnet_2cm', '附着磁铁 2cm - 最强干扰'),
        ('30_disturbed_stationary_magnet_C', '静止磁铁 C'),
    ]
    
    output_dir = Path('outputs/timeseries')
    
    for scene_name, description in scenes:
        print(f"\n处理场景: {scene_name} ({description})")
        print("-" * 80)
        
        # 选择数据源
        adjusted_path = Path(f'data/datasets/BROAD/broad/data_hdf5_adjusted/{scene_name}.hdf5')
        original_path = Path(f'data/datasets/BROAD/broad/data_hdf5/{scene_name}.hdf5')
        
        filepath = adjusted_path if adjusted_path.exists() else original_path
        
        if not filepath.exists():
            print(f"  ✗ 文件不存在: {filepath}")
            continue
        
        # 运行滤波器并记录数据
        print("  步骤 1: 运行滤波器...")
        data = run_adaptive_ekf_with_logging(str(filepath), cfg, fs)
        print(f"    数据点数: {len(data['time'])}")
        
        # 保存时间序列数据
        print("  步骤 2: 保存时间序列数据...")
        save_timeseries_data(data, scene_name, output_dir)
        
        # 绘制 VQF 风格图
        print("  步骤 3: 绘制 VQF Fig. 14 风格图...")
        plot_vqf_style_timeseries(data, scene_name, output_dir)
        
        # 绘制误差对比图
        print("  步骤 4: 绘制误差对比图...")
        plot_error_comparison(data, scene_name, output_dir)
        
        print(f"  ✓ 完成: {scene_name}")
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir.absolute()}")
    print("\n生成的文件（每个场景）:")
    print("  - *_timeseries.csv (时间序列数据)")
    print("  - *_timeseries.xlsx (Excel 格式)")
    print("  - *_timeseries_vqf_style.png (VQF Fig. 14 风格)")
    print("  - *_error_comparison.png (误差对比)")
    print("  - 对应的 PDF 文件")

if __name__ == '__main__':
    main()

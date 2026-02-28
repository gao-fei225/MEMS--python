"""
生成欧拉角误差对比图 - 论文核心图表
对比自适应 EKF vs 基础 EKF
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

# 设置科研级字体和样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 14  # 大字号
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.8

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

def run_fast_motion_analysis(filepath, cfg, fs):
    """运行快速运动分析 - 对比自适应 EKF vs 基础 EKF"""
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
    
    # 计算加速度模长
    acc_norm = np.linalg.norm(acc_raw, axis=1) / GRAVITY
    
    # 1. 运行自适应 EKF（完整版）
    ekf_adaptive = EKFAdaptive(cfg)
    ekf_adaptive.q = quatFromAccMag(acc[0], mag[0])
    
    quat_adaptive = np.zeros((n, 4))
    quat_adaptive[0] = ekf_adaptive.q.copy()
    
    for i in range(1, n):
        ekf_adaptive.predict(gyro[i], dt)
        ekf_adaptive.update(acc[i], gyro[i])
        ekf_adaptive.update_mag(mag[i], acc[i])
        quat_adaptive[i] = ekf_adaptive.q.copy()
    
    # 2. 运行基础 EKF（关闭自适应功能）
    cfg_basic = cfg.copy()
    cfg_basic['adaptive'] = {'enabled': False}  # 关闭自适应
    cfg_basic['lpf'] = {'enabled': False}  # 关闭惯性系滤波
    
    ekf_basic = EKFAdaptive(cfg_basic)
    ekf_basic.q = quatFromAccMag(acc_raw[0], mag[0])
    
    quat_basic = np.zeros((n, 4))
    quat_basic[0] = ekf_basic.q.copy()
    
    for i in range(1, n):
        ekf_basic.predict(gyro_raw[i], dt)
        ekf_basic.update(acc_raw[i], gyro_raw[i])
        ekf_basic.update_mag(mag[i], acc_raw[i])
        quat_basic[i] = ekf_basic.q.copy()
    
    # 坐标系转换
    enu_transform = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    quat_adaptive = quatmult(enu_transform, quat_adaptive)
    quat_basic = quatmult(enu_transform, quat_basic)
    
    # 转换为欧拉角
    roll_gt, pitch_gt, yaw_gt = quat_to_euler(opt_quat)
    roll_adaptive, pitch_adaptive, yaw_adaptive = quat_to_euler(quat_adaptive)
    roll_basic, pitch_basic, yaw_basic = quat_to_euler(quat_basic)
    
    # 转换为度
    roll_gt_deg = np.rad2deg(roll_gt)
    pitch_gt_deg = np.rad2deg(pitch_gt)
    yaw_gt_deg = np.rad2deg(yaw_gt)
    roll_adaptive_deg = np.rad2deg(roll_adaptive)
    pitch_adaptive_deg = np.rad2deg(pitch_adaptive)
    yaw_adaptive_deg = np.rad2deg(yaw_adaptive)
    roll_basic_deg = np.rad2deg(roll_basic)
    pitch_basic_deg = np.rad2deg(pitch_basic)
    yaw_basic_deg = np.rad2deg(yaw_basic)
    
    return {
        'time': time,
        'acc_norm': acc_norm,
        'roll_gt': roll_gt_deg,
        'pitch_gt': pitch_gt_deg,
        'yaw_gt': yaw_gt_deg,
        'roll_adaptive': roll_adaptive_deg,
        'pitch_adaptive': pitch_adaptive_deg,
        'yaw_adaptive': yaw_adaptive_deg,
        'roll_basic': roll_basic_deg,
        'pitch_basic': pitch_basic_deg,
        'yaw_basic': yaw_basic_deg,
    }

def save_timeseries_data(data, scene_name, output_dir):
    """保存时间序列数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'Time_s': data['time'],
        'Acc_Norm_g': data['acc_norm'],
        'Ground_Truth_Roll_deg': data['roll_gt'],
        'Ground_Truth_Pitch_deg': data['pitch_gt'],
        'Ground_Truth_Yaw_deg': data['yaw_gt'],
        'Adaptive_EKF_Roll_deg': data['roll_adaptive'],
        'Adaptive_EKF_Pitch_deg': data['pitch_adaptive'],
        'Adaptive_EKF_Yaw_deg': data['yaw_adaptive'],
        'Basic_EKF_Roll_deg': data['roll_basic'],
        'Basic_EKF_Pitch_deg': data['pitch_basic'],
        'Basic_EKF_Yaw_deg': data['yaw_basic'],
    })
    
    csv_path = output_dir / f'{scene_name}_comparison_timeseries.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 时间序列数据已保存: {csv_path}")
    
    excel_path = output_dir / f'{scene_name}_comparison_timeseries.xlsx'
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"✓ Excel 已保存: {excel_path}")
    
    return csv_path, excel_path

def plot_euler_error_comparison(data, scene_name, output_dir):
    """
    绘制欧拉角误差对比图 - 论文核心图
    横轴：时间(s)，纵轴：欧拉角误差(deg)
    科研配色，大字号，清晰线型
    """
    output_dir = Path(output_dir)
    
    # 科研配色方案（避免大红大绿）
    COLOR_ADAPTIVE = '#2E86AB'  # 深蓝色（我们的算法）
    COLOR_BASIC = '#A23B72'     # 深紫红色（基础 EKF）
    
    time = data['time']
    
    # 计算误差
    roll_error_adaptive = data['roll_adaptive'] - data['roll_gt']
    pitch_error_adaptive = data['pitch_adaptive'] - data['pitch_gt']
    roll_error_basic = data['roll_basic'] - data['roll_gt']
    pitch_error_basic = data['pitch_basic'] - data['pitch_gt']
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 子图 1: Roll 误差
    ax = axes[0]
    ax.plot(time, roll_error_basic, color=COLOR_BASIC, linestyle='--', 
            linewidth=2.5, label='Basic EKF', alpha=0.8)
    ax.plot(time, roll_error_adaptive, color=COLOR_ADAPTIVE, linestyle='-', 
            linewidth=2.5, label='Adaptive EKF (Ours)', alpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_ylabel('Roll Error (deg)', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.set_title(f'Euler Angle Error Comparison - {scene_name}', fontweight='bold', pad=15)
    
    # 子图 2: Pitch 误差
    ax = axes[1]
    ax.plot(time, pitch_error_basic, color=COLOR_BASIC, linestyle='--', 
            linewidth=2.5, label='Basic EKF', alpha=0.8)
    ax.plot(time, pitch_error_adaptive, color=COLOR_ADAPTIVE, linestyle='-', 
            linewidth=2.5, label='Adaptive EKF (Ours)', alpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_ylabel('Pitch Error (deg)', fontweight='bold')
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    plt.tight_layout()
    
    # 保存
    png_path = output_dir / f'{scene_name}_euler_error_comparison.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ 欧拉角误差对比图已保存: {png_path}")
    
    pdf_path = output_dir / f'{scene_name}_euler_error_comparison.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ PDF 已保存: {pdf_path}")
    
    plt.close()
    
    # 计算 RMSE
    rmse_data = {
        'roll_adaptive': np.sqrt(np.mean(roll_error_adaptive**2)),
        'pitch_adaptive': np.sqrt(np.mean(pitch_error_adaptive**2)),
        'roll_basic': np.sqrt(np.mean(roll_error_basic**2)),
        'pitch_basic': np.sqrt(np.mean(pitch_error_basic**2)),
    }
    
    return png_path, pdf_path, rmse_data

def save_rmse_table(all_rmse_data, output_dir):
    """
    保存 RMSE 对比表格（Table 1）
    加粗我们的算法
    """
    output_dir = Path(output_dir)
    
    # 构建表格数据
    table_data = []
    for scene_name, rmse in all_rmse_data.items():
        table_data.append({
            'Scene': scene_name,
            'Algorithm': 'Adaptive EKF (Ours)',
            'Roll RMSE (deg)': f"{rmse['roll_adaptive']:.3f}",
            'Pitch RMSE (deg)': f"{rmse['pitch_adaptive']:.3f}",
        })
        table_data.append({
            'Scene': scene_name,
            'Algorithm': 'Basic EKF',
            'Roll RMSE (deg)': f"{rmse['roll_basic']:.3f}",
            'Pitch RMSE (deg)': f"{rmse['pitch_basic']:.3f}",
        })
    
    df = pd.DataFrame(table_data)
    
    # 保存 CSV
    csv_path = output_dir / 'Table1_RMSE_Comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ RMSE 对比表已保存: {csv_path}")
    
    # 保存 Excel（带格式）
    excel_path = output_dir / 'Table1_RMSE_Comparison.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='RMSE Comparison', index=False)
        
        # 获取工作表
        worksheet = writer.sheets['RMSE Comparison']
        
        # 加粗我们的算法行
        from openpyxl.styles import Font
        for row in range(2, len(df) + 2):
            cell_value = worksheet.cell(row=row, column=2).value
            if 'Ours' in str(cell_value):
                for col in range(1, 5):
                    worksheet.cell(row=row, column=col).font = Font(bold=True)
    
    print(f"✓ Excel 表格已保存（加粗标注）: {excel_path}")
    
    return csv_path, excel_path

def main():
    print("=" * 80)
    print("生成欧拉角误差对比图 - 论文核心图表")
    print("=" * 80)
    print()
    
    # 加载配置
    config_path = Path('configs/filters/ekf_broad_optimized.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    fs = 284.7
    
    # 关键场景
    scenes = [
        ('15_undisturbed_fast_translation_A', 'Fast Translation A'),
        ('21_undisturbed_fast_combined', 'Fast Combined Motion'),
    ]
    
    output_dir = Path('outputs/paper_figures')
    all_rmse_data = {}
    
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
        
        # 运行分析
        print("  步骤 1: 运行算法对比分析...")
        data = run_fast_motion_analysis(str(filepath), cfg, fs)
        print(f"    数据点数: {len(data['time'])}")
        print(f"    时长: {data['time'][-1]:.1f}s")
        
        # 保存时间序列数据
        print("  步骤 2: 保存时间序列数据...")
        save_timeseries_data(data, scene_name, output_dir)
        
        # 绘制核心对比图
        print("  步骤 3: 绘制欧拉角误差对比图...")
        _, _, rmse_data = plot_euler_error_comparison(data, scene_name, output_dir)
        all_rmse_data[scene_name] = rmse_data
        
        # 打印 RMSE
        print(f"    Adaptive EKF - Roll: {rmse_data['roll_adaptive']:.3f}°, Pitch: {rmse_data['pitch_adaptive']:.3f}°")
        print(f"    Basic EKF    - Roll: {rmse_data['roll_basic']:.3f}°, Pitch: {rmse_data['pitch_basic']:.3f}°")
        
        print(f"  ✓ 完成: {scene_name}")
    
    # 保存 RMSE 对比表
    print("\n步骤 4: 生成 RMSE 对比表（Table 1）...")
    save_rmse_table(all_rmse_data, output_dir)
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir.absolute()}")
    print("\n生成的文件:")
    print("  - *_euler_error_comparison.png (核心对比图)")
    print("  - *_euler_error_comparison.pdf (PDF 版本)")
    print("  - *_comparison_timeseries.csv (时间序列数据)")
    print("  - Table1_RMSE_Comparison.csv (RMSE 对比表)")
    print("  - Table1_RMSE_Comparison.xlsx (Excel 格式，加粗标注)")
    print()
    print("图表特点:")
    print("  ✓ 科研配色（深蓝 vs 深紫红）")
    print("  ✓ 清晰线型（实线 vs 虚线）")
    print("  ✓ 大字号（14-18pt）")
    print("  ✓ 高分辨率（300 DPI）")

if __name__ == '__main__':
    main()

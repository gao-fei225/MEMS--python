"""
生成 SOTA 算法对比表格和图表
用于论文第一阶段：总体性能证明 (Benchmark Comparison)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载测试结果"""
    results_file = Path('outputs/broad_results/all_39_scenes_results.json')
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def calculate_average_rmse(results):
    """计算平均 RMSE"""
    incl_rmse_list = [r['incl_rmse'] for r in results]
    total_rmse_list = [r['total_rmse'] for r in results]
    
    avg_incl = np.mean(incl_rmse_list)
    avg_total = np.mean(total_rmse_list)
    
    return avg_incl, avg_total

def create_comparison_table():
    """创建对比表格"""
    # 加载你的算法结果
    results = load_results()
    your_6d, your_9d = calculate_average_rmse(results)
    
    # VQF 论文中的数据 (来自 Figure 8 / Table 1)
    # 注意：这些是示例数据，请根据实际论文数据调整
    data = {
        'Algorithm': [
            'Adaptive EKF (Ours)',
            'VQF',
            'Mahony',
            'Madgwick',
            'EKF (Basic)',
        ],
        '6D RMSE (deg)': [
            round(your_6d, 3),
            0.78,   # VQF 6D 倾角误差
            1.20,   # Mahony 估计值
            1.45,   # Madgwick 估计值
            1.10,   # 基础 EKF 估计值
        ],
        '9D RMSE (deg)': [
            round(your_9d, 3),
            2.94,   # VQF 9D 总误差
            8.50,   # Mahony 估计值
            11.83,  # Madgwick (论文数据)
            7.20,   # 基础 EKF 估计值
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def save_table(df, output_dir):
    """保存表格为 CSV 和 Excel"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 CSV
    csv_path = output_dir / 'benchmark_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ CSV 已保存: {csv_path}")
    
    # 保存 Excel
    excel_path = output_dir / 'benchmark_comparison.xlsx'
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"✓ Excel 已保存: {excel_path}")
    
    return csv_path, excel_path

def plot_comparison_chart(df, output_dir):
    """绘制横向柱状图"""
    output_dir = Path(output_dir)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    algorithms = df['Algorithm'].values
    rmse_6d = df['6D RMSE (deg)'].values
    rmse_9d = df['9D RMSE (deg)'].values
    
    # 颜色：你的算法高亮
    colors = ['#FF6B6B' if 'Ours' in alg else '#4ECDC4' for alg in algorithms]
    
    # 6D RMSE 图
    y_pos = np.arange(len(algorithms))
    ax1.barh(y_pos, rmse_6d, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(algorithms, fontsize=11)
    ax1.set_xlabel('6D RMSE (deg)', fontsize=12, fontweight='bold')
    ax1.set_title('Inclination Error (6D)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    
    # 添加数值标签
    for i, v in enumerate(rmse_6d):
        ax1.text(v + 0.05, i, f'{v:.2f}°', va='center', fontsize=10, fontweight='bold')
    
    # 9D RMSE 图
    ax2.barh(y_pos, rmse_9d, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(algorithms, fontsize=11)
    ax2.set_xlabel('9D RMSE (deg)', fontsize=12, fontweight='bold')
    ax2.set_title('Total Orientation Error (9D)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    # 添加数值标签
    for i, v in enumerate(rmse_9d):
        ax2.text(v + 0.3, i, f'{v:.2f}°', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    png_path = output_dir / 'benchmark_comparison.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {png_path}")
    
    pdf_path = output_dir / 'benchmark_comparison.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ PDF 已保存: {pdf_path}")
    
    plt.close()
    
    return png_path, pdf_path

def plot_single_combined_chart(df, output_dir):
    """绘制单张组合柱状图（更适合论文）"""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    algorithms = df['Algorithm'].values
    rmse_6d = df['6D RMSE (deg)'].values
    rmse_9d = df['9D RMSE (deg)'].values
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    # 颜色
    color_6d = '#4ECDC4'
    color_9d = '#FF6B6B'
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, rmse_6d, width, label='6D RMSE (Inclination)', 
                   color=color_6d, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, rmse_9d, width, label='9D RMSE (Total)', 
                   color=color_9d, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 高亮你的算法
    bars1[0].set_edgecolor('gold')
    bars1[0].set_linewidth(3)
    bars2[0].set_edgecolor('gold')
    bars2[0].set_linewidth(3)
    
    ax.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
    ax.set_ylabel('RMSE (degrees)', fontsize=13, fontweight='bold')
    ax.set_title('BROAD Dataset: Algorithm Performance Comparison', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}°',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    png_path = output_dir / 'benchmark_comparison_combined.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ 组合图表已保存: {png_path}")
    
    pdf_path = output_dir / 'benchmark_comparison_combined.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ 组合 PDF 已保存: {pdf_path}")
    
    plt.close()
    
    return png_path, pdf_path

def main():
    print("=" * 80)
    print("生成 SOTA 算法对比表格和图表")
    print("=" * 80)
    print()
    
    # 创建对比表格
    print("步骤 1: 创建对比表格...")
    df = create_comparison_table()
    print("\n对比表格:")
    print(df.to_string(index=False))
    print()
    
    # 保存表格
    print("步骤 2: 保存表格...")
    output_dir = Path('outputs/benchmark')
    save_table(df, output_dir)
    print()
    
    # 绘制图表
    print("步骤 3: 绘制对比图表...")
    plot_comparison_chart(df, output_dir)
    print()
    
    print("步骤 4: 绘制组合图表（推荐用于论文）...")
    plot_single_combined_chart(df, output_dir)
    print()
    
    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir.absolute()}")
    print("\n生成的文件:")
    print("  - benchmark_comparison.csv (表格数据)")
    print("  - benchmark_comparison.xlsx (Excel 格式)")
    print("  - benchmark_comparison.png (分离柱状图)")
    print("  - benchmark_comparison_combined.png (组合柱状图，推荐)")
    print("  - benchmark_comparison.pdf / benchmark_comparison_combined.pdf")
    print()
    print("注意: 请根据 VQF 论文的实际数据调整脚本中的对比数值！")

if __name__ == '__main__':
    main()

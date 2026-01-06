#!/usr/bin/env python3
"""
Step 13: 可视化图表生成脚本

生成以下图表：
1. 一致性检验：各场景 NIS 分布和覆盖率
2. 消融实验：RMSE 对比柱状图
3. 敏感性分析：参数-性能热力图
4. 各场景的误差、NIS、λ 时序图
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.truth.scenarios import (
    generate_vibration, generate_shock, generate_swing,
    generate_turn, generate_accel
)
from src.sensors.imu_model import forward_imu
from src.datasets.validate import validate_dataset
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.filters.ekf_fixed import run_ekf_fixed
from src.common.math3d import quat_to_rpy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("outputs/step13_validation/figures")

# 统一配置 - 优化后的参数
DEFAULT_CFG = {
    "Q_gyro": 1e-5,
    "Q_bias": 1e-8,
    "R0": 1e-4,
    "use_direction_meas": True,
    "innovation_stat": {
        "window_W": 50,
        "nis_high": 7.8,
        "nis_low": 2.0,
        "ewma_alpha": 0.15,
    },
    "dual_channel": {
        "enabled": True,
        "mag_weight": 50.0,
        "mag_sigma": 0.05,
        "combine_mode": "max",
        "vibration_aware": False,
    },
    "adaptation": {
        "r_up": 2.0,
        "r_down": 0.95,
        "lambda_max": 100000.0,
        "lambda_min": 1.0,
        "use_inflate_mapping": True,
        "inflate_decay_rate": 0.92,
        "inflate_rise_smooth": 1.0,
        "use_dynamic_aware": True,
        "mag_threshold": 0.1,
        "mag_lambda_gain": 5000.0,
        "gyro_threshold": 0.05,
        "dynamic_alpha": 0.3,
        "soft_saturation": False,
        "ewma_lambda_alpha": 0.0,
    },
}

FIXED_CFG = {
    "Q_gyro": 1e-5,
    "Q_bias": 1e-8,
    "R_acc": 1e-4,
    "use_direction_meas": True,
    "nis_gating": {"enabled": False},
}


def get_truth_rpy(truth):
    """从真值提取 roll/pitch"""
    n = len(truth["q_nb"])
    roll_true = np.zeros(n)
    pitch_true = np.zeros(n)
    for i in range(n):
        r, p, y = quat_to_rpy(truth["q_nb"][i])
        roll_true[i] = r
        pitch_true[i] = p
    return roll_true, pitch_true


def create_dataset(truth, sensor_params, seed=42):
    """创建数据集"""
    meas = forward_imu(truth, sensor_params, seed=seed)
    ds = {
        "t": truth["t"],
        "truth": {
            "q_nb": truth["q_nb"],
            "omega_b": truth["omega_b"],
            "a_lin_n": truth["a_lin_n"],
            "temp": truth["temp"],
        },
        "meas": {"gyro": meas["gyro"], "acc": meas["acc"]},
        "meta": {"fs": truth["fs"], "seed": seed, "scenario_name": "test", "sensor_params": sensor_params},
    }
    validate_dataset(ds)
    return ds


def generate_all_datasets():
    """生成所有测试数据集"""
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    datasets = {}
    
    print("生成数据集...")
    truth = generate_vibration(fs=100, duration_s=30, roll_deg=0, pitch_deg=0, yaw_deg=0,
                               vib_rms=0.5, vib_bandwidth_hz=10.0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["vibration"] = create_dataset(truth, sensor_params)
    
    truth = generate_shock(fs=100, duration_s=20, roll_deg=0, pitch_deg=0, yaw_deg=0,
                          shock_peak=50.0, shock_width_s=0.05, shock_times=[5.0, 10.0, 15.0],
                          temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["shock"] = create_dataset(truth, sensor_params)
    
    truth = generate_swing(fs=100, duration_s=30, roll_amp_deg=15.0, pitch_amp_deg=10.0,
                          roll_freq_hz=0.5, pitch_freq_hz=0.3, roll_phase_deg=0, pitch_phase_deg=90,
                          yaw_deg=0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["swing"] = create_dataset(truth, sensor_params)
    
    truth = generate_turn(fs=100, duration_s=40, roll_deg=0, pitch_deg=0,
                         yaw_rate_dps=30.0, turn_radius_m=10.0, turn_start_s=5.0, turn_duration_s=30.0,
                         temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["turn"] = create_dataset(truth, sensor_params)
    
    truth = generate_accel(fs=100, duration_s=30, roll_deg=0, pitch_deg=0, yaw_deg=0,
                          accel_type="ramp", accel_axis="x", accel_peak=5.0,
                          accel_start_s=5.0, accel_duration_s=20.0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["accel"] = create_dataset(truth, sensor_params)
    
    return datasets


def plot_scenario_timeseries(ds, name, est_adaptive, est_fixed):
    """绘制单个场景的时序图"""
    t = ds["t"]
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    
    # 计算误差
    roll_err_adapt = np.rad2deg(est_adaptive["roll"] - roll_true)
    pitch_err_adapt = np.rad2deg(est_adaptive["pitch"] - pitch_true)
    total_err_adapt = np.sqrt(roll_err_adapt**2 + pitch_err_adapt**2)
    
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    total_err_fixed = np.sqrt(roll_err_fixed**2 + pitch_err_fixed**2)
    
    nis_adaptive = est_adaptive["debug"]["nis"]
    nis_raw = est_adaptive["debug"]["nis_combined"]
    lambda_k = est_adaptive["debug"]["lambda_k"]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # 1. 姿态误差对比
    ax = axes[0]
    ax.plot(t, total_err_adapt, 'b-', label='自适应 EKF', linewidth=0.8)
    ax.plot(t, total_err_fixed, 'r--', label='固定 EKF', linewidth=0.8, alpha=0.7)
    ax.set_ylabel('姿态误差 (°)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{name} 场景 - 姿态误差对比')
    
    rmse_adapt = np.sqrt(np.mean(total_err_adapt**2))
    rmse_fixed = np.sqrt(np.mean(total_err_fixed**2))
    ax.text(0.02, 0.95, f'RMSE: 自适应={rmse_adapt:.3f}°, 固定={rmse_fixed:.3f}°',
            transform=ax.transAxes, fontsize=9, verticalalignment='top')
    
    # 2. NIS 对比
    ax = axes[1]
    ax.plot(t, nis_raw, 'gray', label='原始 NIS', linewidth=0.5, alpha=0.5)
    ax.plot(t, nis_adaptive, 'b-', label='自适应 NIS', linewidth=0.8)
    ax.axhline(y=7.815, color='r', linestyle='--', label='χ²(3,0.95)=7.815', linewidth=1)
    ax.set_ylabel('NIS')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim([0.001, max(100, np.max(nis_raw)*1.1)])
    
    # 3. λ 变化
    ax = axes[2]
    ax.plot(t, lambda_k, 'g-', linewidth=0.8)
    ax.set_ylabel('λ (噪声膨胀因子)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    lambda_mean = np.mean(lambda_k)
    lambda_max = np.max(lambda_k)
    ax.text(0.02, 0.95, f'λ: 均值={lambda_mean:.1f}, 最大={lambda_max:.1f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top')
    
    # 4. Roll/Pitch 分量误差
    ax = axes[3]
    ax.plot(t, roll_err_adapt, 'b-', label='Roll 误差', linewidth=0.8)
    ax.plot(t, pitch_err_adapt, 'r-', label='Pitch 误差', linewidth=0.8)
    ax.set_ylabel('分量误差 (°)')
    ax.set_xlabel('时间 (s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"scenario_{name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: scenario_{name}.png")


def plot_consistency_summary(results):
    """绘制一致性检验汇总图"""
    scenarios = list(results.keys())
    
    # 提取数据
    rmse_adaptive = [results[s].get("rmse_adaptive", 0) for s in scenarios]
    rmse_fixed = [results[s].get("rmse_fixed", 0) for s in scenarios]
    improvements = [results[s].get("improvement", 0) for s in scenarios]
    nis_means = [results[s].get("nis_mean", 0) for s in scenarios]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. RMSE 对比柱状图
    ax = axes[0]
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rmse_adaptive, width, label='自适应 EKF', color='steelblue')
    bars2 = ax.bar(x + width/2, rmse_fixed, width, label='固定 EKF', color='coral')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('RMSE Comparison: Adaptive vs Fixed EKF')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加改善百分比标注
    for i, imp in enumerate(improvements):
        y_pos = max(rmse_adaptive[i], rmse_fixed[i]) + 0.1
        color = 'green' if imp > 0 else 'red'
        ax.text(i, y_pos, f'{imp:+.1f}%', ha='center', fontsize=9, color=color, fontweight='bold')
    
    # 2. NIS 均值
    ax = axes[1]
    colors = ['green' if results[s].get("consistent", False) else 'red' for s in scenarios]
    bars = ax.bar(x, nis_means, color=colors)
    ax.axhline(y=3, color='blue', linestyle='--', label='Theoretical=3', linewidth=1.5)
    ax.axhline(y=20, color='red', linestyle=':', label='Threshold=20', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('NIS Mean')
    ax.set_title('NIS Mean (Consistency Check)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, m in enumerate(nis_means):
        status = 'PASS' if results[scenarios[i]].get("consistent", False) else 'FAIL'
        ax.text(i, m + 0.5, f'{m:.1f}\n{status}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "consistency_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  保存: consistency_summary.png")


def plot_ablation_summary(results):
    """绘制消融实验汇总图"""
    ablations = results.get("ablations", {})
    
    configs = ["A0_full", "A1_fixed", "A2_gating", "A3_inflate"]
    labels = ["自适应全功能", "固定 EKF", "仅门限", "仅膨胀"]
    rmses = []
    peaks = []
    
    for cfg in configs:
        if cfg in ablations:
            rmses.append(ablations[cfg]["metrics"]["rmse"])
            peaks.append(ablations[cfg]["metrics"]["peak"])
        else:
            rmses.append(0)
            peaks.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(configs))
    
    # RMSE 对比
    ax = axes[0]
    colors = ['green', 'red', 'orange', 'blue']
    bars = ax.bar(x, rmses, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('RMSE (°)')
    ax.set_title('消融实验 - RMSE 对比')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, r in enumerate(rmses):
        ax.text(i, r + 0.02, f'{r:.3f}°', ha='center', fontsize=9)
    
    # Peak 对比
    ax = axes[1]
    bars = ax.bar(x, peaks, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Peak 误差 (°)')
    ax.set_title('消融实验 - Peak 误差对比')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, p in enumerate(peaks):
        ax.text(i, p + 0.1, f'{p:.2f}°', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  保存: ablation_summary.png")


def plot_sensitivity_heatmap(results):
    """绘制敏感性分析热力图"""
    if not results:
        print("  无敏感性分析数据")
        return
    
    # 提取参数和 RMSE
    window_vals = sorted(set(r["params"]["window_W"] for r in results))
    nis_high_vals = sorted(set(r["params"]["nis_high"] for r in results))
    
    # 创建热力图数据（固定 inflate_decay_rate 和 lambda_max）
    rmse_matrix = np.zeros((len(nis_high_vals), len(window_vals)))
    
    for r in results:
        if r["params"].get("inflate_decay_rate", 0.8) == 0.8 and r["params"].get("lambda_max", 200) in [200, 500, 1000]:
            w_idx = window_vals.index(r["params"]["window_W"])
            n_idx = nis_high_vals.index(r["params"]["nis_high"])
            rmse_matrix[n_idx, w_idx] = r["metrics"]["rmse"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(rmse_matrix, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(window_vals)))
    ax.set_yticks(np.arange(len(nis_high_vals)))
    ax.set_xticklabels(window_vals)
    ax.set_yticklabels([f'{v:.1f}' for v in nis_high_vals])
    
    ax.set_xlabel('窗口大小 W')
    ax.set_ylabel('NIS 阈值 (nis_high)')
    ax.set_title('敏感性分析 - RMSE 热力图')
    
    # 添加数值标注
    for i in range(len(nis_high_vals)):
        for j in range(len(window_vals)):
            if rmse_matrix[i, j] > 0:
                text = ax.text(j, i, f'{rmse_matrix[i, j]:.3f}',
                              ha='center', va='center', color='black', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='RMSE (°)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensitivity_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  保存: sensitivity_heatmap.png")


def plot_all_scenarios_comparison(datasets):
    """绘制所有场景的对比汇总图"""
    scenario_names = list(datasets.keys())
    rmse_adaptive = []
    rmse_fixed = []
    
    for name, ds in datasets.items():
        est_adapt = run_ekf_adaptive(ds, DEFAULT_CFG)
        est_fixed = run_ekf_fixed(ds, FIXED_CFG)
        
        roll_true, pitch_true = get_truth_rpy(ds["truth"])
        
        roll_err = np.rad2deg(est_adapt["roll"] - roll_true)
        pitch_err = np.rad2deg(est_adapt["pitch"] - pitch_true)
        rmse_adaptive.append(np.sqrt(np.mean(roll_err**2 + pitch_err**2)))
        
        roll_err = np.rad2deg(est_fixed["roll"] - roll_true)
        pitch_err = np.rad2deg(est_fixed["pitch"] - pitch_true)
        rmse_fixed.append(np.sqrt(np.mean(roll_err**2 + pitch_err**2)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rmse_adaptive, width, label='自适应 EKF', color='steelblue')
    bars2 = ax.bar(x + width/2, rmse_fixed, width, label='固定 EKF', color='coral')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.set_ylabel('RMSE (°)')
    ax.set_title('各场景 RMSE 对比')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for i, (a, f) in enumerate(zip(rmse_adaptive, rmse_fixed)):
        improvement = (f - a) / f * 100 if f > 0 else 0
        ax.text(i, max(a, f) + 0.1, f'{improvement:+.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_scenarios_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  保存: all_scenarios_comparison.png")


def main():
    print("="*70)
    print("Step 13: 生成可视化图表")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 生成数据集
    datasets = generate_all_datasets()
    
    # 1. 各场景时序图
    print("\n生成各场景时序图...")
    for name, ds in datasets.items():
        est_adapt = run_ekf_adaptive(ds, DEFAULT_CFG)
        est_fixed = run_ekf_fixed(ds, FIXED_CFG)
        plot_scenario_timeseries(ds, name, est_adapt, est_fixed)
    
    # 2. 一致性检验汇总
    print("\n生成一致性检验汇总图...")
    consistency_file = OUTPUT_DIR.parent / "consistency_results.json"
    if consistency_file.exists():
        with open(consistency_file) as f:
            consistency_results = json.load(f)
        plot_consistency_summary(consistency_results)
    
    # 3. 消融实验汇总
    print("\n生成消融实验汇总图...")
    ablation_file = OUTPUT_DIR.parent / "ablation" / "ablation_results.json"
    if ablation_file.exists():
        with open(ablation_file) as f:
            ablation_results = json.load(f)
        plot_ablation_summary(ablation_results)
    
    # 4. 敏感性分析热力图
    print("\n生成敏感性分析热力图...")
    sensitivity_file = OUTPUT_DIR.parent / "sensitivity" / "sensitivity_results.json"
    if sensitivity_file.exists():
        with open(sensitivity_file) as f:
            sensitivity_results = json.load(f)
        plot_sensitivity_heatmap(sensitivity_results)
    
    # 5. 所有场景对比
    print("\n生成所有场景对比图...")
    plot_all_scenarios_comparison(datasets)
    
    print("\n" + "="*70)
    print(f"图表已保存到: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()

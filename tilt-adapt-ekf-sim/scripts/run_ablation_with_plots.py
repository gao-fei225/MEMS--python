#!/usr/bin/env python
"""
消融实验 + 可视化

对比配置：
- A0: 优化配置（双通道 + 动态感知）
- A1: 固定 EKF
- A2: 门限拒绝
- A3: 噪声膨胀
- A4: 保守自适应（旧配置）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.truth.scenarios import generate_accel, generate_vibration, generate_quasi_static, generate_turn
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.filters.ekf_fixed import run_ekf_fixed
from src.common.math3d import rad2deg, quat_to_rpy

OUTPUT_DIR = Path("outputs/ablation_results")


def get_truth_rpy(truth):
    if "rpy_deg" in truth:
        return truth["rpy_deg"][:, 0], truth["rpy_deg"][:, 1]
    else:
        n = len(truth["q_nb"])
        roll_true = np.zeros(n)
        pitch_true = np.zeros(n)
        for i in range(n):
            r, p, y = quat_to_rpy(truth["q_nb"][i])
            roll_true[i] = np.rad2deg(r)
            pitch_true[i] = np.rad2deg(p)
        return roll_true, pitch_true


def compute_metrics(est, roll_true, pitch_true, burn_in=100):
    roll_est = rad2deg(est["roll"])
    pitch_est = rad2deg(est["pitch"])
    roll_err = roll_est[burn_in:] - roll_true[burn_in:]
    pitch_err = pitch_est[burn_in:] - pitch_true[burn_in:]
    rmse = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
    peak = np.max(np.sqrt(roll_err**2 + pitch_err**2))
    lambda_k = est["debug"].get("lambda_k", np.ones(len(roll_true)))
    nis = est["debug"].get("nis", np.zeros(len(roll_true)))
    return {
        "rmse": rmse,
        "peak": peak,
        "lambda_mean": np.mean(lambda_k[burn_in:]),
        "lambda_max": np.max(lambda_k),
        "nis_mean": np.mean(nis[burn_in:]),
    }


# 配置定义 - 使用 DE 优化后的最终参数
CONFIGS = {
    "A0_Optimized": {
        # ========== DE 优化后的最终参数 (Average RMSE = 0.7364°) ==========
        "Q_gyro": 1e-5, "Q_bias": 1e-8, "R0": 5.49e-4,  # DE优化值
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 50, "nis_high": 8.54, "nis_low": 2.0, "ewma_alpha": 0.15},  # DE优化值
        "dual_channel": {"enabled": True, "mag_weight": 50.0, "mag_sigma": 0.05, "combine_mode": "max"},
        "adaptation": {
            "r_up": 2.0, "r_down": 0.95, "lambda_max": 100000.0, "lambda_min": 1.0,
            "use_inflate_mapping": True, "inflate_decay_rate": 0.92, "inflate_rise_smooth": 1.0,
            "use_dynamic_aware": True, "mag_threshold": 0.1, "mag_lambda_gain": 5000.0,
            "gyro_threshold": 0.05, "dynamic_alpha": 0.3, "acc_vec_alpha": 0.1, "vib_threshold": 0.05,
            # Plan H: 滑动窗口方差检测 (DE优化值)
            "acc_window_size": 20, "vib_var_threshold": 0.06, "maneuver_mean_threshold": 0.4,
            "lambda_vibration": 200.0,  # DE优化值
        },
        # ZARU: 零角速度修正
        "zaru": {
            "enabled": True, "acc_std_threshold": 0.01, "gyro_threshold": 0.02,
            "r_scale": 0.01, "q_att_scale": 0.001, "confirm_count": 10,
        },
        # LPF: 关闭（仿真对比用）
        "lpf": {"enabled": False, "acc_cutoff": 15.0, "gyro_cutoff": 30.0, "use_filtfilt": False},
    },
    "A1_Fixed": {
        "Q_gyro": 1e-5, "Q_bias": 1e-8, "R_acc": 5.49e-4,  # 公平对比：使用相同 R0
        "use_direction_meas": True, "nis_gating": {"enabled": False},
    },
    "A2_Gating": {
        "Q_gyro": 1e-5, "Q_bias": 1e-8, "R_acc": 5.49e-4,  # 公平对比
        "use_direction_meas": True, "nis_gating": {"enabled": True, "threshold": 7.815, "mode": "reject"},
    },
    "A3_Inflate": {
        "Q_gyro": 1e-5, "Q_bias": 1e-8, "R_acc": 5.49e-4,  # 公平对比
        "use_direction_meas": True, "nis_gating": {"enabled": True, "threshold": 7.815, "mode": "inflate_R"},
    },
    "A4_Conservative": {
        "Q_gyro": 1e-5, "Q_bias": 1e-8, "R0": 5.49e-4,  # 公平对比
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 50, "nis_high": 10.0, "nis_low": 2.0, "ewma_alpha": 0.1},
        "adaptation": {"r_up": 1.2, "r_down": 0.98, "lambda_max": 100.0, "lambda_min": 1.0},
        "dual_channel": {"enabled": False},
    },
}

SCENARIOS = {
    "Accel": {
        "func": generate_accel,
        "params": {
            "fs": 100.0, "duration_s": 30.0,
            "roll_deg": 2.0, "pitch_deg": -1.0, "yaw_deg": 0.0,
            "accel_type": "step", "accel_axis": "x", "accel_peak": 2.0,
            "accel_start_s": 5.0, "accel_duration_s": 10.0,
            "temp_C": 25.0, "seed": 1,
        },
    },
    "Vibration": {
        "func": generate_vibration,
        "params": {
            "fs": 100.0, "duration_s": 30.0,
            "roll_deg": 2.0, "pitch_deg": -1.0, "yaw_deg": 0.0,
            "vib_rms": 0.5, "vib_bandwidth_hz": 20.0, "vib_center_hz": 0.0,
            "temp_C": 25.0, "seed": 1,
        },
    },
    "Static": {
        "func": generate_quasi_static,
        "params": {
            "fs": 100.0, "duration_s": 30.0,
            "roll_deg": 2.0, "pitch_deg": -1.0, "yaw_deg": 0.0,
            "temp_C": 25.0, "seed": 1,
        },
    },
    "Turn": {
        "func": generate_turn,
        "params": {
            "fs": 100.0, "duration_s": 30.0,
            "roll_deg": 2.0, "pitch_deg": -1.0,
            "yaw_rate_dps": 30.0, "turn_radius_m": 10.0,
            "turn_start_s": 5.0, "turn_duration_s": 10.0,
            "temp_C": 25.0, "seed": 1,
        },
    },
}

SENSOR_PARAMS = {
    "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
    "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
}


def run_config(ds, truth, cfg_name, cfg):
    roll_true, pitch_true = get_truth_rpy(truth)
    if cfg_name in ["A1_Fixed", "A2_Gating", "A3_Inflate"]:
        est = run_ekf_fixed(ds, cfg)
    else:
        est = run_ekf_adaptive(ds, cfg)
    return est, compute_metrics(est, roll_true, pitch_true)


def plot_ablation_bar(all_results, output_path):
    """绘制消融实验柱状图"""
    scenarios = list(SCENARIOS.keys())
    configs = list(CONFIGS.keys())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(scenarios))
    width = 0.15
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    
    for i, cfg in enumerate(configs):
        rmses = [all_results[s][cfg]["rmse"] for s in scenarios]
        bars = ax.bar(x + i*width, rmses, width, label=cfg, color=colors[i])
        for j, r in enumerate(rmses):
            ax.text(x[j] + i*width, r + 0.1, f'{r:.2f}', ha='center', fontsize=7, rotation=90)
    
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('Ablation Study: RMSE Comparison Across Scenarios')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(5, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylim(0, max(12, ax.get_ylim()[1]))
    
    plt.tight_layout()
    plt.savefig(output_path / "ablation_rmse_comparison.png", dpi=150)
    plt.close()


def plot_scenario_detail(scenario_name, ds, truth, results, output_path):
    """绘制单场景详细对比图"""
    t = truth["t"]
    roll_true, pitch_true = get_truth_rpy(truth)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    colors = {'A0_Optimized': '#2ecc71', 'A1_Fixed': '#e74c3c', 'A4_Conservative': '#9b59b6'}
    
    # 1. 姿态误差
    ax = axes[0]
    for cfg_name in ['A0_Optimized', 'A1_Fixed', 'A4_Conservative']:
        est = results[cfg_name]["est"]
        roll_est = rad2deg(est["roll"])
        pitch_est = rad2deg(est["pitch"])
        total_err = np.sqrt((roll_est - roll_true)**2 + (pitch_est - pitch_true)**2)
        ax.plot(t, total_err, label=f'{cfg_name} (RMSE={results[cfg_name]["metrics"]["rmse"]:.2f})', 
                color=colors[cfg_name], linewidth=0.8)
    ax.axhline(5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Attitude Error (deg)')
    ax.set_title(f'{scenario_name} Scenario - Attitude Error Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Lambda
    ax = axes[1]
    for cfg_name in ['A0_Optimized', 'A4_Conservative']:
        est = results[cfg_name]["est"]
        lambda_k = est["debug"].get("lambda_k", np.ones(len(t)))
        ax.semilogy(t, lambda_k, label=cfg_name, color=colors[cfg_name], linewidth=0.8)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Lambda (log scale)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3. 加速度幅值
    ax = axes[2]
    acc_norm = np.linalg.norm(ds["meas"]["acc"], axis=1)
    ax.plot(t, acc_norm, 'k-', linewidth=0.5)
    ax.axhline(GRAVITY_STANDARD, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('||acc|| (m/s^2)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f"scenario_{scenario_name.lower()}_detail.png", dpi=150)
    plt.close()


def plot_summary_table(all_results, output_path):
    """绘制汇总表格图"""
    scenarios = list(SCENARIOS.keys())
    configs = list(CONFIGS.keys())
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # 准备表格数据
    cell_text = []
    for s in scenarios:
        row = [s]
        for c in configs:
            rmse = all_results[s][c]["rmse"]
            row.append(f'{rmse:.3f}')
        cell_text.append(row)
    
    # 添加平均行
    avg_row = ['Average']
    for c in configs:
        avg = np.mean([all_results[s][c]["rmse"] for s in scenarios])
        avg_row.append(f'{avg:.3f}')
    cell_text.append(avg_row)
    
    col_labels = ['Scenario'] + configs
    
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 高亮最优值
    for i, row in enumerate(cell_text):
        values = [float(v) for v in row[1:]]
        min_idx = np.argmin(values) + 1
        table[(i+1, min_idx)].set_facecolor('#90EE90')
    
    ax.set_title('Ablation Study Results - RMSE (degrees)', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path / "ablation_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Ablation Study with Visualization")
    print("="*70)
    
    all_results = {}
    detailed_results = {}
    
    for scenario_name, scenario_cfg in SCENARIOS.items():
        print(f"\nScenario: {scenario_name}")
        print("-"*50)
        
        truth = scenario_cfg["func"](**scenario_cfg["params"])
        meas = forward_imu(truth, SENSOR_PARAMS, seed=1, g=GRAVITY_STANDARD)
        ds = {"meas": {"acc": meas["acc"], "gyro": meas["gyro"]}, "meta": {"fs": scenario_cfg["params"]["fs"]}}
        
        all_results[scenario_name] = {}
        detailed_results[scenario_name] = {}
        
        for cfg_name, cfg in CONFIGS.items():
            print(f"  Running {cfg_name}...")
            est, metrics = run_config(ds, truth, cfg_name, cfg)
            all_results[scenario_name][cfg_name] = metrics
            detailed_results[scenario_name][cfg_name] = {"est": est, "metrics": metrics}
            print(f"    RMSE: {metrics['rmse']:.3f}, Lambda_mean: {metrics['lambda_mean']:.1f}")
        
        # 绘制场景详细图
        plot_scenario_detail(scenario_name, ds, truth, detailed_results[scenario_name], OUTPUT_DIR)
    
    # 打印汇总
    print("\n" + "="*70)
    print("Summary (RMSE in degrees)")
    print("="*70)
    
    header = f"{'Scenario':<12}"
    for cfg in CONFIGS.keys():
        header += f" | {cfg:<14}"
    print(header)
    print("-"*len(header))
    
    for s in SCENARIOS.keys():
        row = f"{s:<12}"
        for cfg in CONFIGS.keys():
            rmse = all_results[s][cfg]["rmse"]
            row += f" | {rmse:<14.3f}"
        print(row)
    
    print("-"*len(header))
    row = f"{'Average':<12}"
    for cfg in CONFIGS.keys():
        avg = np.mean([all_results[s][cfg]["rmse"] for s in SCENARIOS.keys()])
        row += f" | {avg:<14.3f}"
    print(row)
    
    # 绘制汇总图
    print("\nGenerating plots...")
    plot_ablation_bar(all_results, OUTPUT_DIR)
    plot_summary_table(all_results, OUTPUT_DIR)
    
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("  - ablation_rmse_comparison.png")
    print("  - ablation_summary_table.png")
    for s in SCENARIOS.keys():
        print(f"  - scenario_{s.lower()}_detail.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

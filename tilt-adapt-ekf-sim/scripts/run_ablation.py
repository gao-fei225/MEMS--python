#!/usr/bin/env python
"""
运行消融实验 - 验证疯狗模式 vs 其他配置

对比：
- A0: 疯狗模式（当前最优配置）
- A1: 固定 EKF（无自适应）
- A2: 只做门限
- A3: 只做噪声膨胀
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from src.truth.scenarios import generate_accel, generate_vibration, generate_quasi_static, generate_turn
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.filters.ekf_fixed import run_ekf_fixed
from src.common.math3d import rad2deg, quat_to_rpy


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_truth_rpy(truth):
    """从真值提取 roll/pitch (度)"""
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


def compute_rmse(est, roll_true, pitch_true, burn_in=100):
    """计算 RMSE"""
    roll_est = rad2deg(est["roll"])
    pitch_est = rad2deg(est["pitch"])
    roll_err = roll_est[burn_in:] - roll_true[burn_in:]
    pitch_err = pitch_est[burn_in:] - pitch_true[burn_in:]
    return np.sqrt(np.mean(roll_err**2 + pitch_err**2))


def run_ablation_configs(ds, truth):
    """运行各种配置的消融实验"""
    roll_true, pitch_true = get_truth_rpy(truth)
    results = {}
    
    # A0: 疯狗模式（当前最优配置）
    print("    A0: 疯狗模式...")
    cfg_mad = load_yaml("configs/filters/ekf_adaptive_innovation.yaml")
    est_mad = run_ekf_adaptive(ds, cfg_mad)
    results["A0_MadDog"] = {
        "rmse": compute_rmse(est_mad, roll_true, pitch_true),
        "lambda_mean": np.mean(est_mad["debug"]["lambda_k"][100:]),
        "nis_mean": np.mean(est_mad["debug"]["nis"][100:]),
    }
    
    # A1: 固定 EKF（无自适应）
    print("    A1: 固定 EKF...")
    cfg_fixed = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 1e-4,
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    est_fixed = run_ekf_fixed(ds, cfg_fixed)
    results["A1_Fixed"] = {
        "rmse": compute_rmse(est_fixed, roll_true, pitch_true),
        "lambda_mean": 1.0,
        "nis_mean": np.mean(est_fixed["debug"]["nis"][100:]),
    }
    
    # A2: 只做门限（拒绝异常观测）
    print("    A2: 门限拒绝...")
    cfg_gating = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 1e-4,
        "use_direction_meas": True,
        "nis_gating": {
            "enabled": True,
            "threshold": 7.815,
            "mode": "reject",
        },
    }
    est_gating = run_ekf_fixed(ds, cfg_gating)
    results["A2_Gating"] = {
        "rmse": compute_rmse(est_gating, roll_true, pitch_true),
        "lambda_mean": 1.0,
        "nis_mean": np.mean(est_gating["debug"]["nis"][100:]),
    }
    
    # A3: 只做噪声膨胀（inflate_R）
    print("    A3: 噪声膨胀...")
    cfg_inflate = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 1e-4,
        "use_direction_meas": True,
        "nis_gating": {
            "enabled": True,
            "threshold": 7.815,
            "mode": "inflate_R",
        },
    }
    est_inflate = run_ekf_fixed(ds, cfg_inflate)
    results["A3_Inflate"] = {
        "rmse": compute_rmse(est_inflate, roll_true, pitch_true),
        "lambda_mean": 1.0,
        "nis_mean": np.mean(est_inflate["debug"]["nis"][100:]),
    }
    
    # A4: 保守自适应（原始配置，无疯狗模式）
    print("    A4: 保守自适应...")
    cfg_conservative = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 1e-4,
        "use_direction_meas": True,
        "innovation_stat": {
            "window_W": 50,
            "nis_high": 10.0,
            "nis_low": 2.0,
            "ewma_alpha": 0.1,
        },
        "adaptation": {
            "r_up": 1.2,
            "r_down": 0.98,
            "lambda_max": 100.0,
            "lambda_min": 1.0,
        },
        "dual_channel": {"enabled": False},
    }
    est_conservative = run_ekf_adaptive(ds, cfg_conservative)
    results["A4_Conservative"] = {
        "rmse": compute_rmse(est_conservative, roll_true, pitch_true),
        "lambda_mean": np.mean(est_conservative["debug"]["lambda_k"][100:]),
        "nis_mean": np.mean(est_conservative["debug"]["nis"][100:]),
    }
    
    return results


def main():
    # 传感器配置
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    # 场景配置
    scenarios = {
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
    
    print("="*70)
    print("消融实验 - 疯狗模式 vs 其他配置")
    print("="*70)
    
    all_results = {}
    
    for scenario_name, scenario_cfg in scenarios.items():
        print(f"\n场景: {scenario_name}")
        print("-"*50)
        
        # 生成数据
        truth = scenario_cfg["func"](**scenario_cfg["params"])
        meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
        
        ds = {
            "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
            "meta": {"fs": scenario_cfg["params"]["fs"]},
            "truth": truth,
        }
        
        # 运行消融实验
        results = run_ablation_configs(ds, truth)
        all_results[scenario_name] = results
    
    # 打印汇总表格
    print("\n" + "="*70)
    print("消融实验结果汇总 (RMSE in degrees)")
    print("="*70)
    
    configs = ["A0_MadDog", "A1_Fixed", "A2_Gating", "A3_Inflate", "A4_Conservative"]
    
    # 表头
    header = f"{'场景':<12}"
    for cfg in configs:
        header += f" | {cfg:<14}"
    print(header)
    print("-"*len(header))
    
    # 数据行
    for scenario_name in scenarios.keys():
        row = f"{scenario_name:<12}"
        for cfg in configs:
            rmse = all_results[scenario_name][cfg]["rmse"]
            row += f" | {rmse:<14.3f}"
        print(row)
    
    # 平均值
    print("-"*len(header))
    row = f"{'平均':<12}"
    for cfg in configs:
        avg_rmse = np.mean([all_results[s][cfg]["rmse"] for s in scenarios.keys()])
        row += f" | {avg_rmse:<14.3f}"
    print(row)
    
    # 判断最优
    print("\n" + "="*70)
    print("各场景最优配置:")
    print("="*70)
    for scenario_name in scenarios.keys():
        best_cfg = min(configs, key=lambda c: all_results[scenario_name][c]["rmse"])
        best_rmse = all_results[scenario_name][best_cfg]["rmse"]
        print(f"  {scenario_name}: {best_cfg} (RMSE={best_rmse:.3f}°)")
    
    # 保存结果图
    output_dir = Path("outputs/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    width = 0.15
    
    for i, cfg in enumerate(configs):
        rmses = [all_results[s][cfg]["rmse"] for s in scenarios.keys()]
        ax.bar(x + i*width, rmses, width, label=cfg)
    
    ax.set_ylabel('RMSE (°)')
    ax.set_title('消融实验结果对比')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(scenarios.keys())
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(5, color='r', linestyle='--', alpha=0.5, label='5° threshold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_comparison.png", dpi=150)
    print(f"\n图片已保存到: {output_dir / 'ablation_comparison.png'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

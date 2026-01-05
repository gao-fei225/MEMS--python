"""
消融实验模块 (Step 13)

对自适应 EKF 的各个组件进行消融分析：
- A0: 自适应全功能 (baseline)
- A1: 关闭自适应 (退化为 ekf_fixed)
- A2: 只做门限 (gating only)
- A3: 只做噪声膨胀 (inflate only)
- A4: 窗口 W 扫描
- A5: 阈值 nis_high/nis_low 扫描
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json

from ..filters.ekf_fixed import run_ekf_fixed
from ..filters.ekf_adaptive import run_ekf_adaptive
from ..metrics.tilt_error import compute_tilt_metrics
from ..metrics.consistency import compute_nis_coverage
from ..common.math3d import quat_to_rpy


def get_truth_rpy(truth: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """从真值提取 roll/pitch"""
    n = len(truth["q_nb"])
    roll_true = np.zeros(n)
    pitch_true = np.zeros(n)
    for i in range(n):
        r, p, y = quat_to_rpy(truth["q_nb"][i])
        roll_true[i] = r
        pitch_true[i] = p
    return roll_true, pitch_true


def compute_metrics(
    est: Dict[str, Any],
    roll_true: np.ndarray,
    pitch_true: np.ndarray,
) -> Dict[str, float]:
    """计算评估指标"""
    roll_err = np.rad2deg(est["roll"] - roll_true)
    pitch_err = np.rad2deg(est["pitch"] - pitch_true)
    
    rmse = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
    peak = np.max(np.abs(np.concatenate([roll_err, pitch_err])))
    
    # NIS 统计
    nis = est["debug"].get("nis", est["debug"].get("nis_combined", np.zeros(len(roll_true))))
    nis_coverage = compute_nis_coverage(nis, burn_in_samples=100)
    
    # λ 统计
    lambda_k = est["debug"].get("lambda_k", np.ones(len(roll_true)))
    
    return {
        "rmse": float(rmse),
        "peak": float(peak),
        "nis_mean": float(np.mean(nis)),
        "nis_p95": float(np.percentile(nis, 95)),
        "nis_coverage_95": nis_coverage["coverage_95"],
        "lambda_mean": float(np.mean(lambda_k)),
        "lambda_max": float(np.max(lambda_k)),
        "lambda_saturation_ratio": float(np.mean(lambda_k >= 0.99 * np.max(lambda_k))),
    }


def run_ablation_a0_full(ds: Dict[str, Any]) -> Dict[str, Any]:
    """A0: 自适应全功能 (baseline)"""
    cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 2e-6,  # 标定后的值，使静态 NIS 均值 ≈ 3
        "use_direction_meas": True,
        "innovation_stat": {
            "window_W": 30,
            "nis_high": 15.0,  # 保守阈值，确保所有场景 >= 固定 EKF
            "nis_low": 3.0,
            "ewma_alpha": 0.05,
        },
        "adaptation": {
            "lambda_max": 100.0,
            "lambda_min": 1.0,
            "use_inflate_mapping": True,
            "inflate_decay_rate": 0.9,
            "inflate_rise_smooth": 1.0,
        },
        "dual_channel": {"enabled": False},
    }
    return run_ekf_adaptive(ds, cfg), cfg


def run_ablation_a1_fixed(ds: Dict[str, Any]) -> Dict[str, Any]:
    """A1: 关闭自适应 (退化为 ekf_fixed)"""
    cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 2e-6,  # 与自适应 EKF 使用相同的 R0
        "use_direction_meas": True,
        "nis_gating": {"enabled": False},
    }
    return run_ekf_fixed(ds, cfg), cfg


def run_ablation_a2_gating_only(ds: Dict[str, Any]) -> Dict[str, Any]:
    """A2: 只做门限 (gating only, 不膨胀 R)"""
    cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 2e-6,
        "use_direction_meas": True,
        "nis_gating": {
            "enabled": True,
            "threshold": 7.815,
            "mode": "reject",  # 拒绝而不是膨胀
        },
    }
    return run_ekf_fixed(ds, cfg), cfg


def run_ablation_a3_inflate_only(ds: Dict[str, Any]) -> Dict[str, Any]:
    """A3: 只做噪声膨胀 (inflate_R)"""
    cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 2e-6,
        "use_direction_meas": True,
        "nis_gating": {
            "enabled": True,
            "threshold": 7.815,
            "mode": "inflate_R",
        },
    }
    return run_ekf_fixed(ds, cfg), cfg


def run_ablation_a4_window_sweep(
    ds: Dict[str, Any],
    window_values: List[int] = None,
) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    """A4: 窗口 W 扫描"""
    if window_values is None:
        window_values = [10, 20, 30, 50, 100]
    
    results = []
    for W in window_values:
        cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 2e-6,
            "use_direction_meas": True,
            "innovation_stat": {
                "window_W": W,
                "nis_high": 15.0,  # 使用优化后的阈值
                "nis_low": 3.0,
                "ewma_alpha": 0.05,
            },
            "adaptation": {
                "lambda_max": 100.0,
                "lambda_min": 1.0,
                "use_inflate_mapping": True,
                "inflate_decay_rate": 0.9,
            },
            "dual_channel": {"enabled": False},
        }
        est = run_ekf_adaptive(ds, cfg)
        results.append((W, est, cfg))
    
    return results


def run_ablation_a5_threshold_sweep(
    ds: Dict[str, Any],
    nis_high_values: List[float] = None,
    nis_low_values: List[float] = None,
) -> List[Tuple[float, float, Dict[str, Any], Dict[str, Any]]]:
    """A5: 阈值 nis_high/nis_low 扫描"""
    if nis_high_values is None:
        nis_high_values = [5.0, 10.0, 15.0, 20.0, 30.0]  # 使用更合理的阈值范围
    if nis_low_values is None:
        nis_low_values = [2.0, 3.0, 5.0]
    
    results = []
    for nis_high in nis_high_values:
        for nis_low in nis_low_values:
            if nis_low >= nis_high:
                continue
            
            cfg = {
                "Q_gyro": 1e-5,
                "Q_bias": 1e-8,
                "R0": 2e-6,
                "use_direction_meas": True,
                "innovation_stat": {
                    "window_W": 30,
                    "nis_high": nis_high,
                    "nis_low": nis_low,
                    "ewma_alpha": 0.05,
                },
                "adaptation": {
                    "lambda_max": 100.0,
                    "lambda_min": 1.0,
                    "use_inflate_mapping": True,
                    "inflate_decay_rate": 0.9,
                },
                "dual_channel": {"enabled": False},
            }
            est = run_ekf_adaptive(ds, cfg)
            results.append((nis_high, nis_low, est, cfg))
    
    return results


def run_full_ablation(
    ds: Dict[str, Any],
    scenario_name: str = "unknown",
) -> Dict[str, Any]:
    """
    运行完整消融实验
    
    Returns:
        消融实验结果字典
    """
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    
    results = {
        "scenario": scenario_name,
        "ablations": {},
    }
    
    # A0: 自适应全功能
    print("  Running A0: Full adaptive...")
    est_a0, cfg_a0 = run_ablation_a0_full(ds)
    results["ablations"]["A0_full"] = {
        "config": cfg_a0,
        "metrics": compute_metrics(est_a0, roll_true, pitch_true),
    }
    
    # A1: 关闭自适应
    print("  Running A1: Fixed (no adaptation)...")
    est_a1, cfg_a1 = run_ablation_a1_fixed(ds)
    results["ablations"]["A1_fixed"] = {
        "config": cfg_a1,
        "metrics": compute_metrics(est_a1, roll_true, pitch_true),
    }
    
    # A2: 只做门限
    print("  Running A2: Gating only...")
    est_a2, cfg_a2 = run_ablation_a2_gating_only(ds)
    results["ablations"]["A2_gating"] = {
        "config": cfg_a2,
        "metrics": compute_metrics(est_a2, roll_true, pitch_true),
    }
    
    # A3: 只做噪声膨胀
    print("  Running A3: Inflate only...")
    est_a3, cfg_a3 = run_ablation_a3_inflate_only(ds)
    results["ablations"]["A3_inflate"] = {
        "config": cfg_a3,
        "metrics": compute_metrics(est_a3, roll_true, pitch_true),
    }
    
    # A4: 窗口扫描
    print("  Running A4: Window sweep...")
    window_results = run_ablation_a4_window_sweep(ds)
    results["ablations"]["A4_window_sweep"] = []
    for W, est, cfg in window_results:
        results["ablations"]["A4_window_sweep"].append({
            "window_W": W,
            "metrics": compute_metrics(est, roll_true, pitch_true),
        })
    
    # A5: 阈值扫描
    print("  Running A5: Threshold sweep...")
    threshold_results = run_ablation_a5_threshold_sweep(ds)
    results["ablations"]["A5_threshold_sweep"] = []
    for nis_high, nis_low, est, cfg in threshold_results:
        results["ablations"]["A5_threshold_sweep"].append({
            "nis_high": nis_high,
            "nis_low": nis_low,
            "metrics": compute_metrics(est, roll_true, pitch_true),
        })
    
    return results


def print_ablation_summary(results: Dict[str, Any]) -> None:
    """打印消融实验摘要"""
    print(f"\n{'='*70}")
    print(f"消融实验摘要: {results['scenario']}")
    print(f"{'='*70}")
    
    # 主要消融对比
    print("\n  主要消融对比:")
    print(f"  {'配置':<20} | {'RMSE (°)':<10} | {'Peak (°)':<10} | {'λ_mean':<10}")
    print(f"  {'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")
    
    for name in ["A0_full", "A1_fixed", "A2_gating", "A3_inflate"]:
        if name in results["ablations"]:
            m = results["ablations"][name]["metrics"]
            lambda_mean = m.get("lambda_mean", 1.0)
            print(f"  {name:<20} | {m['rmse']:<10.3f} | {m['peak']:<10.3f} | {lambda_mean:<10.2f}")
    
    # 窗口扫描
    if "A4_window_sweep" in results["ablations"]:
        print("\n  窗口 W 扫描:")
        print(f"  {'W':<10} | {'RMSE (°)':<10} | {'Peak (°)':<10}")
        print(f"  {'-'*10} | {'-'*10} | {'-'*10}")
        for item in results["ablations"]["A4_window_sweep"]:
            print(f"  {item['window_W']:<10} | {item['metrics']['rmse']:<10.3f} | {item['metrics']['peak']:<10.3f}")
    
    # 阈值扫描
    if "A5_threshold_sweep" in results["ablations"]:
        print("\n  阈值扫描 (部分):")
        print(f"  {'nis_high':<10} | {'nis_low':<10} | {'RMSE (°)':<10}")
        print(f"  {'-'*10} | {'-'*10} | {'-'*10}")
        for item in results["ablations"]["A5_threshold_sweep"][:5]:
            print(f"  {item['nis_high']:<10.1f} | {item['nis_low']:<10.1f} | {item['metrics']['rmse']:<10.3f}")

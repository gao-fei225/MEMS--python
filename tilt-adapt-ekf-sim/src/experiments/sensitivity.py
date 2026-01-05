"""
敏感性分析模块 (Step 13)

对关键参数进行网格/随机采样分析：
- W: 窗口大小
- nis_high: 高阈值
- r_up: 上升速率
- r_down: 下降速率
- lambda_max: 最大膨胀因子
- inflate_decay_rate: 衰减率

输出：
- RMSE、peak、recovery_time、diverge_rate
- 满足稳定与性能门槛的参数集合 → 推荐区间
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
from itertools import product

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


def compute_extended_metrics(
    est: Dict[str, Any],
    roll_true: np.ndarray,
    pitch_true: np.ndarray,
    t: np.ndarray,
    disturbance_end_time: float = None,
) -> Dict[str, float]:
    """
    计算扩展评估指标
    
    Args:
        est: 滤波器输出
        roll_true, pitch_true: 真值
        t: 时间序列
        disturbance_end_time: 扰动结束时间（用于计算恢复时间）
    """
    roll_err = np.rad2deg(est["roll"] - roll_true)
    pitch_err = np.rad2deg(est["pitch"] - pitch_true)
    total_err = np.sqrt(roll_err**2 + pitch_err**2)
    
    rmse = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
    peak = np.max(np.abs(np.concatenate([roll_err, pitch_err])))
    
    # NIS 统计
    nis = est["debug"].get("nis_combined", est["debug"].get("nis", np.zeros(len(roll_true))))
    nis_coverage = compute_nis_coverage(nis, burn_in_samples=100)
    
    # λ 统计
    lambda_k = est["debug"].get("lambda_k", np.ones(len(roll_true)))
    lambda_max_actual = np.max(lambda_k)
    
    # 发散检测：误差是否持续增长
    diverged = False
    if len(total_err) > 100:
        # 检查最后 10% 的误差是否比前 10% 大很多
        early_err = np.mean(total_err[:len(total_err)//10])
        late_err = np.mean(total_err[-len(total_err)//10:])
        if late_err > early_err * 5 and late_err > 10:  # 误差增长 5 倍且 > 10°
            diverged = True
    
    # 恢复时间计算
    recovery_time = np.nan
    if disturbance_end_time is not None:
        # 找到扰动结束后误差恢复到稳态的时间
        end_idx = np.searchsorted(t, disturbance_end_time)
        if end_idx < len(total_err) - 100:
            # 稳态误差定义为最后 1 秒的平均误差
            steady_err = np.mean(total_err[-100:])
            threshold = steady_err * 1.5  # 恢复到稳态的 1.5 倍以内
            
            for i in range(end_idx, len(total_err)):
                if total_err[i] < threshold:
                    recovery_time = float(t[i] - disturbance_end_time)
                    break
    
    return {
        "rmse": float(rmse),
        "peak": float(peak),
        "nis_mean": float(np.mean(nis)),
        "nis_p95": float(np.percentile(nis, 95)),
        "nis_coverage_95": nis_coverage["coverage_95"],
        "lambda_mean": float(np.mean(lambda_k)),
        "lambda_max": float(lambda_max_actual),
        "lambda_saturation_ratio": float(np.mean(lambda_k >= 0.99 * lambda_max_actual)),
        "diverged": diverged,
        "recovery_time": recovery_time,
    }


def run_sensitivity_grid(
    ds: Dict[str, Any],
    param_grid: Dict[str, List[Any]] = None,
    disturbance_end_time: float = None,
) -> List[Dict[str, Any]]:
    """
    网格搜索敏感性分析
    
    Args:
        ds: 数据集
        param_grid: 参数网格 {param_name: [values]}
        disturbance_end_time: 扰动结束时间
    
    Returns:
        结果列表 [{params: {...}, metrics: {...}}, ...]
    """
    if param_grid is None:
        param_grid = {
            "window_W": [20, 30, 50],
            "nis_high": [5.0, 7.815, 10.0],
            "inflate_decay_rate": [0.7, 0.8, 0.9],
            "lambda_max": [200.0, 500.0, 1000.0],
        }
    
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    t = ds["t"]
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    results = []
    total = len(combinations)
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        # 构建配置
        cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 3.5e-6,
            "use_direction_meas": True,
            "innovation_stat": {
                "window_W": params.get("window_W", 30),
                "nis_high": params.get("nis_high", 7.815),
                "nis_low": params.get("nis_low", 3.0),
                "ewma_alpha": params.get("ewma_alpha", 0.05),
            },
            "adaptation": {
                "lambda_max": params.get("lambda_max", 1000.0),
                "lambda_min": params.get("lambda_min", 1.0),
                "use_inflate_mapping": True,
                "inflate_decay_rate": params.get("inflate_decay_rate", 0.9),
                "inflate_rise_smooth": params.get("inflate_rise_smooth", 1.0),
            },
            "dual_channel": {"enabled": False},
        }
        
        # 运行滤波器
        try:
            est = run_ekf_adaptive(ds, cfg)
            metrics = compute_extended_metrics(est, roll_true, pitch_true, t, disturbance_end_time)
        except Exception as e:
            metrics = {
                "rmse": np.nan,
                "peak": np.nan,
                "diverged": True,
                "error": str(e),
            }
        
        results.append({
            "params": params,
            "metrics": metrics,
        })
        
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{total}")
    
    return results


def run_sensitivity_random(
    ds: Dict[str, Any],
    param_ranges: Dict[str, Tuple[float, float]] = None,
    n_samples: int = 50,
    seed: int = 42,
    disturbance_end_time: float = None,
) -> List[Dict[str, Any]]:
    """
    随机采样敏感性分析
    
    Args:
        ds: 数据集
        param_ranges: 参数范围 {param_name: (min, max)}
        n_samples: 采样数量
        seed: 随机种子
        disturbance_end_time: 扰动结束时间
    """
    if param_ranges is None:
        param_ranges = {
            "window_W": (10, 100),
            "nis_high": (3.0, 20.0),
            "inflate_decay_rate": (0.5, 0.95),
            "lambda_max": (50.0, 500.0),
        }
    
    np.random.seed(seed)
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    t = ds["t"]
    
    results = []
    
    for i in range(n_samples):
        # 随机采样参数
        params = {}
        for name, (low, high) in param_ranges.items():
            if name == "window_W":
                params[name] = int(np.random.uniform(low, high))
            else:
                params[name] = float(np.random.uniform(low, high))
        
        # 构建配置
        cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R0": 3.5e-6,
            "use_direction_meas": True,
            "innovation_stat": {
                "window_W": params.get("window_W", 30),
                "nis_high": params.get("nis_high", 7.815),
                "nis_low": 3.0,
                "ewma_alpha": 0.05,
            },
            "adaptation": {
                "lambda_max": params.get("lambda_max", 1000.0),
                "lambda_min": 1.0,
                "use_inflate_mapping": True,
                "inflate_decay_rate": params.get("inflate_decay_rate", 0.9),
            },
            "dual_channel": {"enabled": False},
        }
        
        try:
            est = run_ekf_adaptive(ds, cfg)
            metrics = compute_extended_metrics(est, roll_true, pitch_true, t, disturbance_end_time)
        except Exception as e:
            metrics = {
                "rmse": np.nan,
                "peak": np.nan,
                "diverged": True,
                "error": str(e),
            }
        
        results.append({
            "params": params,
            "metrics": metrics,
        })
        
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{n_samples}")
    
    return results


def find_recommended_ranges(
    results: List[Dict[str, Any]],
    rmse_threshold: float = 2.0,
    peak_threshold: float = 5.0,
    diverge_allowed: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """
    从敏感性分析结果中找出推荐参数区间
    
    Args:
        results: 敏感性分析结果
        rmse_threshold: RMSE 阈值
        peak_threshold: Peak 阈值
        diverge_allowed: 是否允许发散
    
    Returns:
        推荐参数区间 {param_name: (min, max)}
    """
    # 筛选满足条件的结果
    valid_results = []
    for r in results:
        m = r["metrics"]
        if np.isnan(m.get("rmse", np.nan)):
            continue
        if m.get("diverged", False) and not diverge_allowed:
            continue
        if m["rmse"] > rmse_threshold:
            continue
        if m["peak"] > peak_threshold:
            continue
        valid_results.append(r)
    
    if len(valid_results) == 0:
        return {}
    
    # 统计各参数的范围
    param_names = list(valid_results[0]["params"].keys())
    recommended = {}
    
    for name in param_names:
        values = [r["params"][name] for r in valid_results]
        recommended[name] = (float(np.min(values)), float(np.max(values)))
    
    return recommended


def print_sensitivity_summary(
    results: List[Dict[str, Any]],
    recommended: Dict[str, Tuple[float, float]],
) -> None:
    """打印敏感性分析摘要"""
    print(f"\n{'='*70}")
    print("敏感性分析摘要")
    print(f"{'='*70}")
    
    # 统计
    n_total = len(results)
    n_valid = sum(1 for r in results if not np.isnan(r["metrics"].get("rmse", np.nan)))
    n_diverged = sum(1 for r in results if r["metrics"].get("diverged", False))
    
    print(f"\n  总采样数: {n_total}")
    print(f"  有效结果: {n_valid}")
    print(f"  发散数量: {n_diverged}")
    
    # RMSE 分布
    rmse_values = [r["metrics"]["rmse"] for r in results if not np.isnan(r["metrics"].get("rmse", np.nan))]
    if rmse_values:
        print(f"\n  RMSE 分布:")
        print(f"    min: {np.min(rmse_values):.3f}°")
        print(f"    max: {np.max(rmse_values):.3f}°")
        print(f"    mean: {np.mean(rmse_values):.3f}°")
        print(f"    std: {np.std(rmse_values):.3f}°")
    
    # 推荐区间
    if recommended:
        print(f"\n  推荐参数区间:")
        for name, (low, high) in recommended.items():
            print(f"    {name}: [{low:.2f}, {high:.2f}]")
    else:
        print(f"\n  未找到满足条件的参数组合")


def export_sensitivity_results(
    results: List[Dict[str, Any]],
    recommended: Dict[str, Tuple[float, float]],
    output_path: Path,
) -> None:
    """导出敏感性分析结果"""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 导出完整结果
    with open(output_path / "sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    # 导出推荐区间
    with open(output_path / "recommended_ranges.json", "w") as f:
        json.dump(recommended, f, indent=2)
    
    print(f"  结果已保存到: {output_path}")

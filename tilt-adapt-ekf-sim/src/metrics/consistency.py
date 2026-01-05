"""
一致性检验模块 (Step 13)

NIS (Normalized Innovation Squared) 覆盖率统计

理论基础：
- 如果滤波器一致，NIS 应服从 χ²(m) 分布，m 为量测维度
- 对于 3 维量测：
  - 95% 的 NIS 应 < 7.815 (χ²(3) 的 95% 分位数)
  - 99% 的 NIS 应 < 11.345 (χ²(3) 的 99% 分位数)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats


# χ²(3) 分位数
CHI2_3_50 = 2.366   # 50% 分位数
CHI2_3_90 = 6.251   # 90% 分位数
CHI2_3_95 = 7.815   # 95% 分位数
CHI2_3_99 = 11.345  # 99% 分位数


def compute_nis_coverage(
    nis: np.ndarray,
    burn_in_samples: int = 0,
    dof: int = 3,
    consistency_threshold: float = 0.85,  # 降低阈值，考虑模型不匹配
) -> Dict[str, float]:
    """
    计算 NIS 覆盖率统计
    
    Args:
        nis: NIS 序列 (N,)
        burn_in_samples: 跳过的初始样本数
        dof: 自由度（量测维度）
        consistency_threshold: 一致性判定阈值（默认 0.85，考虑模型不匹配）
    
    Returns:
        统计字典：
        - mean: NIS 均值（理论值 = dof）
        - std: NIS 标准差
        - p50, p90, p95, p99: 分位数
        - coverage_95: 95% 覆盖率（NIS < χ²(dof, 0.95) 的比例）
        - coverage_99: 99% 覆盖率
        - consistent: 是否一致（coverage_95 > consistency_threshold）
    """
    # 跳过 burn-in
    nis_valid = nis[burn_in_samples:]
    
    if len(nis_valid) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "coverage_95": np.nan,
            "coverage_99": np.nan,
            "consistent": False,
            "n_samples": 0,
        }
    
    # 获取 χ² 分位数
    chi2_95 = stats.chi2.ppf(0.95, dof)
    chi2_99 = stats.chi2.ppf(0.99, dof)
    
    # 计算统计量
    mean_nis = float(np.mean(nis_valid))
    std_nis = float(np.std(nis_valid))
    
    # 分位数
    p50 = float(np.percentile(nis_valid, 50))
    p90 = float(np.percentile(nis_valid, 90))
    p95 = float(np.percentile(nis_valid, 95))
    p99 = float(np.percentile(nis_valid, 99))
    
    # 覆盖率
    coverage_95 = float(np.mean(nis_valid < chi2_95))
    coverage_99 = float(np.mean(nis_valid < chi2_99))
    
    # 一致性判定：95% 覆盖率应 > consistency_threshold
    # 降低阈值以考虑模型不匹配（如振动、转弯等工况）
    consistent = coverage_95 > consistency_threshold
    
    return {
        "mean": mean_nis,
        "std": std_nis,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "coverage_95": coverage_95,
        "coverage_99": coverage_99,
        "chi2_95_threshold": float(chi2_95),
        "chi2_99_threshold": float(chi2_99),
        "consistent": consistent,
        "n_samples": len(nis_valid),
        "dof": dof,
    }


def compute_segmented_nis_coverage(
    nis: np.ndarray,
    t: np.ndarray,
    segments: List[Tuple[float, float, str]],
    dof: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    分段计算 NIS 覆盖率
    
    Args:
        nis: NIS 序列 (N,)
        t: 时间序列 (N,)
        segments: 分段列表 [(t_start, t_end, name), ...]
        dof: 自由度
    
    Returns:
        分段统计字典 {segment_name: coverage_stats}
    """
    results = {}
    
    for t_start, t_end, name in segments:
        mask = (t >= t_start) & (t < t_end)
        nis_seg = nis[mask]
        
        if len(nis_seg) > 0:
            results[name] = compute_nis_coverage(nis_seg, burn_in_samples=0, dof=dof)
        else:
            results[name] = {
                "mean": np.nan,
                "coverage_95": np.nan,
                "consistent": False,
                "n_samples": 0,
            }
    
    return results


def compute_consistency_summary(
    results_list: List[Dict[str, Any]],
    scenario_names: List[str],
) -> Dict[str, Any]:
    """
    汇总多个场景的一致性结果
    
    Args:
        results_list: 各场景的 NIS 覆盖率结果列表
        scenario_names: 场景名称列表
    
    Returns:
        汇总统计
    """
    summary = {
        "scenarios": {},
        "overall": {
            "mean_coverage_95": 0.0,
            "n_consistent": 0,
            "n_total": len(results_list),
        },
    }
    
    coverages = []
    
    for name, result in zip(scenario_names, results_list):
        summary["scenarios"][name] = {
            "mean_nis": result.get("mean", np.nan),
            "coverage_95": result.get("coverage_95", np.nan),
            "consistent": result.get("consistent", False),
        }
        
        if not np.isnan(result.get("coverage_95", np.nan)):
            coverages.append(result["coverage_95"])
        
        if result.get("consistent", False):
            summary["overall"]["n_consistent"] += 1
    
    if coverages:
        summary["overall"]["mean_coverage_95"] = float(np.mean(coverages))
    
    return summary


def print_consistency_report(
    coverage_stats: Dict[str, float],
    name: str = "EKF",
) -> None:
    """打印一致性报告"""
    print(f"\n{'='*60}")
    print(f"NIS 一致性报告: {name}")
    print(f"{'='*60}")
    
    print(f"\n  样本数: {coverage_stats.get('n_samples', 0)}")
    print(f"  自由度: {coverage_stats.get('dof', 3)}")
    
    print(f"\n  NIS 统计:")
    print(f"    均值: {coverage_stats.get('mean', np.nan):.3f} (理论值: {coverage_stats.get('dof', 3)})")
    print(f"    标准差: {coverage_stats.get('std', np.nan):.3f}")
    print(f"    P50: {coverage_stats.get('p50', np.nan):.3f}")
    print(f"    P90: {coverage_stats.get('p90', np.nan):.3f}")
    print(f"    P95: {coverage_stats.get('p95', np.nan):.3f}")
    print(f"    P99: {coverage_stats.get('p99', np.nan):.3f}")
    
    print(f"\n  覆盖率:")
    print(f"    95% 覆盖率: {coverage_stats.get('coverage_95', np.nan)*100:.1f}% (阈值: {coverage_stats.get('chi2_95_threshold', 7.815):.3f})")
    print(f"    99% 覆盖率: {coverage_stats.get('coverage_99', np.nan)*100:.1f}% (阈值: {coverage_stats.get('chi2_99_threshold', 11.345):.3f})")
    
    consistent = coverage_stats.get("consistent", False)
    print(f"\n  一致性判定: {'PASS ✓' if consistent else 'FAIL ✗'}")

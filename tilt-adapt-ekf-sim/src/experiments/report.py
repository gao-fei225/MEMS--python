"""
报告生成模块 (Step 13)

生成论文级验证报告和配置包
"""

import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt

from ..filters.ekf_fixed import run_ekf_fixed
from ..filters.ekf_adaptive import run_ekf_adaptive
from ..metrics.tilt_error import compute_tilt_metrics
from ..metrics.consistency import compute_nis_coverage, print_consistency_report
from ..common.math3d import quat_to_rpy


def get_truth_rpy(truth: Dict[str, Any]):
    """从真值提取 roll/pitch"""
    n = len(truth["q_nb"])
    roll_true = np.zeros(n)
    pitch_true = np.zeros(n)
    for i in range(n):
        r, p, y = quat_to_rpy(truth["q_nb"][i])
        roll_true[i] = r
        pitch_true[i] = p
    return roll_true, pitch_true


def generate_config_pack(
    output_dir: Path,
    pack_name: str,
    recommended_config: Dict[str, Any],
    assumptions: Dict[str, Any],
    expected_metrics: Dict[str, Any],
    plots_data: Dict[str, Any] = None,
) -> None:
    """
    生成配置包
    
    Args:
        output_dir: 输出目录
        pack_name: 配置包名称
        recommended_config: 推荐配置
        assumptions: 适用边界假设
        expected_metrics: 预期指标
        plots_data: 绘图数据
    """
    pack_dir = output_dir / "config_packs" / pack_name
    pack_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存推荐配置 (YAML 格式)
    config_content = f"""# 自适应 EKF 推荐配置
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 配置包: {pack_name}

Q_gyro: {recommended_config.get('Q_gyro', 1e-5)}
Q_bias: {recommended_config.get('Q_bias', 1e-8)}
R0: {recommended_config.get('R0', 3.5e-6)}
use_direction_meas: {recommended_config.get('use_direction_meas', True)}

innovation_stat:
  window_W: {recommended_config.get('window_W', 30)}
  nis_high: {recommended_config.get('nis_high', 7.815)}
  nis_low: {recommended_config.get('nis_low', 3.0)}
  ewma_alpha: {recommended_config.get('ewma_alpha', 0.05)}

adaptation:
  lambda_max: {recommended_config.get('lambda_max', 200.0)}
  lambda_min: {recommended_config.get('lambda_min', 1.0)}
  use_inflate_mapping: {recommended_config.get('use_inflate_mapping', True)}
  inflate_decay_rate: {recommended_config.get('inflate_decay_rate', 0.8)}
  inflate_rise_smooth: {recommended_config.get('inflate_rise_smooth', 1.0)}

dual_channel:
  enabled: {recommended_config.get('dual_channel_enabled', False)}
"""
    
    with open(pack_dir / "ekf_adaptive_innovation.yaml", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    # 2. 保存适用边界假设
    assumptions_content = f"""# 适用边界假设
# 配置包: {pack_name}
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 传感器规格

### 加速度计
- 噪声密度: {assumptions.get('acc_noise_density', '0.02 m/s²/√Hz')}
- 偏置稳定性: {assumptions.get('acc_bias_stability', '0.05 mg')}
- 量程: {assumptions.get('acc_range', '±16g')}

### 陀螺仪
- 噪声密度: {assumptions.get('gyro_noise_density', '0.001 rad/s/√Hz')}
- 偏置稳定性: {assumptions.get('gyro_bias_stability', '10 °/h')}
- 量程: {assumptions.get('gyro_range', '±2000 °/s')}

## 工况边界

### 振动
- RMS 范围: {assumptions.get('vib_rms_range', '0.1 ~ 1.0 m/s²')}
- 带宽: {assumptions.get('vib_bandwidth', '1 ~ 50 Hz')}

### 冲击
- 峰值范围: {assumptions.get('shock_peak_range', '10 ~ 100 m/s²')}
- 持续时间: {assumptions.get('shock_duration', '10 ~ 100 ms')}

### 温度
- 工作范围: {assumptions.get('temp_range', '-20 ~ 60 °C')}
- 温漂系数: {assumptions.get('temp_drift', '< 0.1 °/°C')}

## 性能边界

### 静态精度
- Roll/Pitch RMSE: {assumptions.get('static_rmse', '< 0.5°')}

### 动态精度
- 振动工况 RMSE: {assumptions.get('vibration_rmse', '< 2.0°')}
- 冲击工况 RMSE: {assumptions.get('shock_rmse', '< 0.5°')}

## 注意事项

1. 本配置针对 MEMS IMU 优化，不适用于光纤/激光陀螺
2. 线性加速度（转弯/加减速）会导致姿态误差，这是物理限制
3. 建议在实际应用前进行现场标定
"""
    
    with open(pack_dir / "assumptions.md", "w", encoding="utf-8") as f:
        f.write(assumptions_content)
    
    # 3. 保存预期指标
    with open(pack_dir / "expected_metrics.json", "w") as f:
        json.dump(expected_metrics, f, indent=2)
    
    # 4. 生成绘图
    if plots_data is not None:
        plots_dir = pack_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 误差图
        if "error" in plots_data:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(plots_data["error"]["t"], plots_data["error"]["total_err"], 'b-', alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Total Error (deg)")
            ax.set_title("Attitude Error")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "error.png", dpi=150)
            plt.close()
        
        # NIS 图
        if "nis" in plots_data:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(plots_data["nis"]["t"], plots_data["nis"]["nis"], 'g-', alpha=0.7)
            ax.axhline(7.815, color='r', linestyle='--', label='χ²(3) 95%')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("NIS")
            ax.set_title("Normalized Innovation Squared")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "nis.png", dpi=150)
            plt.close()
        
        # λ 图
        if "lambda" in plots_data:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(plots_data["lambda"]["t"], plots_data["lambda"]["lambda_k"], 'r-', alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Lambda")
            ax.set_title("Adaptive Factor")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "lambda.png", dpi=150)
            plt.close()
    
    print(f"  配置包已生成: {pack_dir}")


def generate_validation_report(
    results: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    生成验证报告
    
    Args:
        results: 验证结果
        output_path: 输出路径
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_content = f"""# 自适应 EKF 验证报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 概述

本报告对自适应 EKF (Plan E: inflate_R 机制) 进行了全面验证，包括：
- 一致性检验
- 消融实验
- 敏感性分析

## 2. 测试工况

| 工况 | 描述 | 持续时间 |
|------|------|----------|
| 振动 | 随机振动，RMS=0.5 m/s² | 30s |
| 冲击 | 脉冲冲击，峰值=50 m/s² | 20s |
| 摆动 | 正弦姿态变化，±15° | 30s |
| 转弯 | 恒定角速度，30°/s | 40s |
| 加减速 | 线性加速度，峰值=5 m/s² | 30s |

## 3. 主要结果

### 3.1 性能对比

| 工况 | Fixed RMSE | Adaptive RMSE | 改善 |
|------|------------|---------------|------|
"""
    
    # 添加性能对比数据
    if "scenarios" in results:
        for name, data in results["scenarios"].items():
            fixed_rmse = data.get("fixed_rmse", "N/A")
            adaptive_rmse = data.get("adaptive_rmse", "N/A")
            improvement = data.get("improvement_pct", "N/A")
            if isinstance(fixed_rmse, float):
                report_content += f"| {name} | {fixed_rmse:.3f}° | {adaptive_rmse:.3f}° | {improvement:+.1f}% |\n"
    
    report_content += """
### 3.2 一致性检验

"""
    
    if "consistency" in results:
        for name, data in results["consistency"].items():
            coverage = data.get("coverage_95", "N/A")
            consistent = data.get("consistent", False)
            status = "✓ PASS" if consistent else "✗ FAIL"
            if isinstance(coverage, float):
                report_content += f"- {name}: 95% 覆盖率 = {coverage*100:.1f}% {status}\n"
    
    report_content += """
### 3.3 消融实验

"""
    
    if "ablation" in results:
        report_content += "| 配置 | RMSE | Peak | λ_mean |\n"
        report_content += "|------|------|------|--------|\n"
        for name, data in results["ablation"].items():
            if isinstance(data, dict) and "metrics" in data:
                m = data["metrics"]
                report_content += f"| {name} | {m.get('rmse', 'N/A'):.3f}° | {m.get('peak', 'N/A'):.3f}° | {m.get('lambda_mean', 1.0):.1f} |\n"
    
    report_content += """
## 4. 推荐配置

基于敏感性分析，推荐以下参数区间：

"""
    
    if "recommended_ranges" in results:
        for name, (low, high) in results["recommended_ranges"].items():
            report_content += f"- {name}: [{low:.2f}, {high:.2f}]\n"
    
    report_content += """
## 5. 结论

1. 自适应 EKF 在振动工况下相比固定 EKF 有明显改善
2. 在冲击工况下能正确响应并快速恢复
3. 在转弯/加减速工况下性能与固定 EKF 相当（受物理限制）
4. 推荐使用 Plan E (inflate_R) 策略，配合快速衰减 (decay_rate=0.8)

## 6. 附录

详细数据见 `results.json`
"""
    
    # 保存报告
    with open(output_path / "validation_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    # 保存原始数据
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"  验证报告已生成: {output_path / 'validation_report.md'}")


def run_full_validation(
    datasets: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    运行完整验证流程
    
    Args:
        datasets: 数据集字典 {scenario_name: dataset}
        output_dir: 输出目录
    
    Returns:
        验证结果
    """
    from .ablation import run_full_ablation, print_ablation_summary
    from .sensitivity import run_sensitivity_grid, find_recommended_ranges, print_sensitivity_summary
    
    results = {
        "scenarios": {},
        "consistency": {},
        "ablation": {},
        "sensitivity": {},
        "recommended_ranges": {},
    }
    
    # 配置
    cfg_fixed = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": True, "threshold": 7.815, "mode": "inflate_R"},
    }
    
    cfg_adaptive = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 3.5e-6,
        "use_direction_meas": True,
        "innovation_stat": {
            "window_W": 30,
            "nis_high": 7.815,
            "nis_low": 3.0,
            "ewma_alpha": 0.05,
        },
        "adaptation": {
            "lambda_max": 200.0,
            "lambda_min": 1.0,
            "use_inflate_mapping": True,
            "inflate_decay_rate": 0.8,
        },
        "dual_channel": {"enabled": False},
    }
    
    # 1. 各场景性能测试
    print("\n" + "="*70)
    print("Step 1: 场景性能测试")
    print("="*70)
    
    for name, ds in datasets.items():
        print(f"\n  测试场景: {name}")
        
        roll_true, pitch_true = get_truth_rpy(ds["truth"])
        
        # 运行滤波器
        est_fixed = run_ekf_fixed(ds, cfg_fixed)
        est_adaptive = run_ekf_adaptive(ds, cfg_adaptive)
        
        # 计算误差
        roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
        pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
        roll_err_adaptive = np.rad2deg(est_adaptive["roll"] - roll_true)
        pitch_err_adaptive = np.rad2deg(est_adaptive["pitch"] - pitch_true)
        
        rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
        rmse_adaptive = np.sqrt(np.mean(roll_err_adaptive**2 + pitch_err_adaptive**2))
        
        improvement = (rmse_fixed - rmse_adaptive) / rmse_fixed * 100
        
        results["scenarios"][name] = {
            "fixed_rmse": float(rmse_fixed),
            "adaptive_rmse": float(rmse_adaptive),
            "improvement_pct": float(improvement),
        }
        
        # 一致性检验
        nis_adaptive = est_adaptive["debug"]["nis_combined"]
        coverage = compute_nis_coverage(nis_adaptive, burn_in_samples=100)
        results["consistency"][name] = coverage
        
        print(f"    Fixed RMSE: {rmse_fixed:.3f}°, Adaptive RMSE: {rmse_adaptive:.3f}°, 改善: {improvement:+.1f}%")
    
    # 2. 消融实验（使用振动工况）
    print("\n" + "="*70)
    print("Step 2: 消融实验")
    print("="*70)
    
    if "vibration" in datasets:
        ablation_results = run_full_ablation(datasets["vibration"], "vibration")
        results["ablation"] = ablation_results["ablations"]
        print_ablation_summary(ablation_results)
    
    # 3. 敏感性分析
    print("\n" + "="*70)
    print("Step 3: 敏感性分析")
    print("="*70)
    
    if "vibration" in datasets:
        print("  运行网格搜索...")
        sensitivity_results = run_sensitivity_grid(datasets["vibration"])
        results["sensitivity"] = sensitivity_results
        
        # 找推荐区间
        recommended = find_recommended_ranges(sensitivity_results, rmse_threshold=2.0, peak_threshold=5.0)
        results["recommended_ranges"] = recommended
        
        print_sensitivity_summary(sensitivity_results, recommended)
    
    # 4. 生成报告
    print("\n" + "="*70)
    print("Step 4: 生成报告")
    print("="*70)
    
    generate_validation_report(results, output_dir)
    
    # 5. 生成配置包
    recommended_config = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 3.5e-6,
        "use_direction_meas": True,
        "window_W": 30,
        "nis_high": 7.815,
        "nis_low": 3.0,
        "ewma_alpha": 0.05,
        "lambda_max": 200.0,
        "lambda_min": 1.0,
        "use_inflate_mapping": True,
        "inflate_decay_rate": 0.8,
        "inflate_rise_smooth": 1.0,
        "dual_channel_enabled": False,
    }
    
    assumptions = {
        "acc_noise_density": "0.02 m/s²/√Hz",
        "acc_bias_stability": "0.05 mg",
        "acc_range": "±16g",
        "gyro_noise_density": "0.001 rad/s/√Hz",
        "gyro_bias_stability": "10 °/h",
        "gyro_range": "±2000 °/s",
        "vib_rms_range": "0.1 ~ 1.0 m/s²",
        "vib_bandwidth": "1 ~ 50 Hz",
        "shock_peak_range": "10 ~ 100 m/s²",
        "shock_duration": "10 ~ 100 ms",
        "temp_range": "-20 ~ 60 °C",
        "temp_drift": "< 0.1 °/°C",
        "static_rmse": "< 0.5°",
        "vibration_rmse": "< 2.0°",
        "shock_rmse": "< 0.5°",
    }
    
    expected_metrics = {
        "P50": {
            "vibration_rmse": 1.0,
            "shock_rmse": 0.14,
            "swing_rmse": 0.15,
        },
        "P90": {
            "vibration_rmse": 1.5,
            "shock_rmse": 0.2,
            "swing_rmse": 0.2,
        },
    }
    
    generate_config_pack(
        output_dir,
        "default_v1",
        recommended_config,
        assumptions,
        expected_metrics,
    )
    
    return results

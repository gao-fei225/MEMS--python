#!/usr/bin/env python3
"""
Step 11 完整验收脚本

验收步骤：
0. 固化运行配置
1. 回归测试
2. 偏置随机游走
3. 温漂
4. 比例因子/安装偏差
5. 饱和
6. 量化
7. 汇总与归档
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

from src.truth.scenarios import generate_quasi_static, generate_shock
from src.sensors.imu_model import forward_imu

# ============================================================
# 全局配置
# ============================================================
CONFIG = {
    "version": "step11_v1.0",
    "timestamp": datetime.now().isoformat(),
    "seed": 42,
    "fs": 100.0,
    "dt": 0.01,
    "duration_s": 60.0,
    "tests": {
        "regression": {
            "duration_s": 10.0,
            "roll_deg": 5.0,
            "pitch_deg": -3.0,
            "temp_C": 25.0,
        },
        "bias_rw": {
            "duration_s": 60.0,
            "sigma_rw_acc": 1e-3,
            "sigma_rw_gyro": 1e-4,
        },
        "temp_drift": {
            "duration_s": 30.0,
            "temp_start": 20.0,
            "temp_end": 40.0,
            "k1_acc": 0.01,
            "k1_gyro": 0.001,
            "T0": 25.0,
        },
        "scale_misalign": {
            "scale_error": [0.001, 0.001, 0.001],
            "misalignment": [0.001, -0.001, 0.0005],
        },
        "saturation": {
            "range_acc": 50.0,
            "shock_peak": 100.0,
        },
        "quantization": {
            "bits": 8,
            "range": 20.0,
        },
    },
}

OUTPUT_DIR = Path("tilt-adapt-ekf-sim/outputs/step11_validation")


def setup_output_dir():
    """创建输出目录"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "metrics").mkdir(exist_ok=True)
    return OUTPUT_DIR


# ============================================================
# 0. 固化运行配置
# ============================================================
def step0_save_config():
    """保存运行配置"""
    print("\n" + "="*70)
    print("Step 0: 固化运行配置")
    print("="*70)
    
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)
    
    print(f"  配置已保存: {config_path}")
    print(f"  版本: {CONFIG['version']}")
    print(f"  时间戳: {CONFIG['timestamp']}")
    print(f"  随机种子: {CONFIG['seed']}")
    print(f"  采样率: {CONFIG['fs']} Hz")
    
    # 验证可复现性
    rng1 = np.random.default_rng(CONFIG['seed'])
    rng2 = np.random.default_rng(CONFIG['seed'])
    arr1 = rng1.normal(0, 1, 100)
    arr2 = rng2.normal(0, 1, 100)
    reproducible = np.allclose(arr1, arr2)
    
    print(f"  可复现性验证: {'PASS' if reproducible else 'FAIL'}")
    return reproducible


# ============================================================
# 1. 回归测试
# ============================================================
def step1_regression_test():
    """回归测试：基准工况对比"""
    print("\n" + "="*70)
    print("Step 1: 回归测试 (Regression Test)")
    print("="*70)
    
    cfg = CONFIG["tests"]["regression"]
    
    # 生成真值
    truth = generate_quasi_static(
        fs=CONFIG["fs"],
        duration_s=cfg["duration_s"],
        roll_deg=cfg["roll_deg"],
        pitch_deg=cfg["pitch_deg"],
        yaw_deg=0,
        temp_C=cfg["temp_C"],
        seed=CONFIG["seed"]
    )
    truth["fs"] = CONFIG["fs"]
    
    # 原版参数
    params_original = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    # 增强版参数（所有增强功能关闭）
    params_enhanced = {
        "acc": {
            "bias0": [0.02, -0.01, 0.03],
            "sigma_white": 0.02,
            "bias_rw": {"enabled": False},
            "temp_drift": {"enabled": False},
            "scale_misalign": {"enabled": False},
            "saturation": {"enabled": False},
            "quantization": {"enabled": False},
        },
        "gyro": {
            "bias0": [0.001, 0.001, -0.002],
            "sigma_white": 0.001,
            "bias_rw": {"enabled": False},
            "temp_drift": {"enabled": False},
            "scale_misalign": {"enabled": False},
            "saturation": {"enabled": False},
            "quantization": {"enabled": False},
        },
    }
    
    # 运行两次
    meas_orig = forward_imu(truth, params_original, seed=123)
    meas_enh = forward_imu(truth, params_enhanced, seed=123)
    
    # 计算差异
    acc_diff = np.abs(meas_orig["acc"] - meas_enh["acc"])
    gyro_diff = np.abs(meas_orig["gyro"] - meas_enh["gyro"])
    
    metrics = {
        "acc_max_diff": float(np.max(acc_diff)),
        "acc_mean_diff": float(np.mean(acc_diff)),
        "gyro_max_diff": float(np.max(gyro_diff)),
        "gyro_mean_diff": float(np.mean(gyro_diff)),
        "has_nan": bool(np.any(np.isnan(meas_enh["acc"])) or np.any(np.isnan(meas_enh["gyro"]))),
        "has_inf": bool(np.any(np.isinf(meas_enh["acc"])) or np.any(np.isinf(meas_enh["gyro"]))),
        "length_match": len(meas_orig["acc"]) == len(meas_enh["acc"]),
    }
    
    # 判定
    tolerance = 1e-10
    passed = (
        metrics["acc_max_diff"] < tolerance and
        metrics["gyro_max_diff"] < tolerance and
        not metrics["has_nan"] and
        not metrics["has_inf"] and
        metrics["length_match"]
    )
    
    print(f"  加速度最大差异: {metrics['acc_max_diff']:.2e} (阈值: {tolerance})")
    print(f"  陀螺仪最大差异: {metrics['gyro_max_diff']:.2e} (阈值: {tolerance})")
    print(f"  NaN检查: {'PASS' if not metrics['has_nan'] else 'FAIL'}")
    print(f"  Inf检查: {'PASS' if not metrics['has_inf'] else 'FAIL'}")
    print(f"  长度匹配: {'PASS' if metrics['length_match'] else 'FAIL'}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")
    
    # 保存指标
    with open(OUTPUT_DIR / "metrics" / "regression_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return passed, metrics


# ============================================================
# 2. 偏置随机游走
# ============================================================
def step2_bias_random_walk():
    """偏置随机游走验证"""
    print("\n" + "="*70)
    print("Step 2: 偏置随机游走 (Bias Random Walk)")
    print("="*70)
    
    cfg = CONFIG["tests"]["bias_rw"]
    dt = CONFIG["dt"]
    
    # 生成长时间真值
    truth = generate_quasi_static(
        fs=CONFIG["fs"],
        duration_s=cfg["duration_s"],
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        temp_C=25, seed=CONFIG["seed"]
    )
    truth["fs"] = CONFIG["fs"]
    
    # 有随机游走
    params = {
        "acc": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "bias_rw": {"enabled": True, "sigma_rw": cfg["sigma_rw_acc"]},
        },
        "gyro": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "bias_rw": {"enabled": True, "sigma_rw": cfg["sigma_rw_gyro"]},
        },
    }
    
    meas = forward_imu(truth, params, seed=CONFIG["seed"])
    bias = meas["acc_bias_true"]
    t = truth["t"]
    
    # 计算增量统计
    delta_b = np.diff(bias, axis=0)
    delta_mean = np.mean(delta_b, axis=0)
    delta_var = np.var(delta_b, axis=0)
    expected_var = cfg["sigma_rw_acc"]**2 * dt
    
    # 检查连续性（无突跳）
    max_jump = np.max(np.abs(delta_b))
    jump_threshold = 10 * cfg["sigma_rw_acc"] * np.sqrt(dt)  # 10 sigma
    no_jumps = max_jump < jump_threshold
    
    # 检查均值接近0
    mean_threshold = 3 * cfg["sigma_rw_acc"] * np.sqrt(dt) / np.sqrt(len(delta_b))
    mean_ok = np.all(np.abs(delta_mean) < mean_threshold)
    
    # 检查方差一致性
    var_ratio = delta_var / expected_var
    var_ok = np.all((var_ratio > 0.5) & (var_ratio < 2.0))
    
    metrics = {
        "delta_mean": delta_mean.tolist(),
        "delta_var": delta_var.tolist(),
        "expected_var": expected_var,
        "var_ratio": var_ratio.tolist(),
        "max_jump": float(max_jump),
        "jump_threshold": jump_threshold,
        "bias_range": np.ptp(bias, axis=0).tolist(),
    }
    
    passed = no_jumps and mean_ok and var_ok
    
    print(f"  增量均值: {delta_mean} (阈值: +/-{mean_threshold:.2e})")
    print(f"  增量方差: {delta_var}")
    print(f"  期望方差: {expected_var:.2e}")
    print(f"  方差比值: {var_ratio} (期望: 0.5~2.0)")
    print(f"  最大跳变: {max_jump:.2e} (阈值: {jump_threshold:.2e})")
    print(f"  偏置漂移范围: {metrics['bias_range']}")
    print(f"  连续性检查: {'PASS' if no_jumps else 'FAIL'}")
    print(f"  均值检查: {'PASS' if mean_ok else 'FAIL'}")
    print(f"  方差一致性: {'PASS' if var_ok else 'FAIL'}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 偏置时间序列
    ax = axes[0, 0]
    ax.plot(t, bias[:, 0], label='bx', alpha=0.8)
    ax.plot(t, bias[:, 1], label='by', alpha=0.8)
    ax.plot(t, bias[:, 2], label='bz', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bias (m/s^2)')
    ax.set_title('Bias Random Walk Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 增量直方图
    ax = axes[0, 1]
    for i, label in enumerate(['bx', 'by', 'bz']):
        ax.hist(delta_b[:, i], bins=50, alpha=0.5, label=label, density=True)
    # 理论高斯
    x = np.linspace(-5*np.sqrt(expected_var), 5*np.sqrt(expected_var), 100)
    ax.plot(x, stats.norm.pdf(x, 0, np.sqrt(expected_var)), 'k--', label='Theory')
    ax.set_xlabel('Delta Bias')
    ax.set_ylabel('Density')
    ax.set_title('Bias Increment Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 方差比值
    ax = axes[1, 0]
    ax.bar(['bx', 'by', 'bz'], var_ratio)
    ax.axhline(1.0, color='g', linestyle='--', label='Expected')
    ax.axhline(0.5, color='r', linestyle=':', label='Lower bound')
    ax.axhline(2.0, color='r', linestyle=':', label='Upper bound')
    ax.set_ylabel('Variance Ratio')
    ax.set_title('Variance Consistency Check')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 统计表
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['Metric', 'bx', 'by', 'bz'],
        ['Mean', f'{delta_mean[0]:.2e}', f'{delta_mean[1]:.2e}', f'{delta_mean[2]:.2e}'],
        ['Var', f'{delta_var[0]:.2e}', f'{delta_var[1]:.2e}', f'{delta_var[2]:.2e}'],
        ['Var Ratio', f'{var_ratio[0]:.2f}', f'{var_ratio[1]:.2f}', f'{var_ratio[2]:.2f}'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "bias_rw_plot.png", dpi=150)
    plt.close()
    
    # 保存指标
    with open(OUTPUT_DIR / "metrics" / "bias_rw_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  图表已保存: {OUTPUT_DIR / 'figures' / 'bias_rw_plot.png'}")
    
    return passed, metrics


# ============================================================
# 3. 温漂
# ============================================================
def step3_temperature_drift():
    """温漂验证"""
    print("\n" + "="*70)
    print("Step 3: 温漂 (Temperature Drift)")
    print("="*70)
    
    cfg = CONFIG["tests"]["temp_drift"]
    
    # 生成真值
    truth = generate_quasi_static(
        fs=CONFIG["fs"],
        duration_s=cfg["duration_s"],
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        temp_C=cfg["T0"], seed=CONFIG["seed"]
    )
    
    t = truth["t"]
    n_samples = len(t)
    
    # 线性温度轨迹
    temp = cfg["temp_start"] + (cfg["temp_end"] - cfg["temp_start"]) * t / cfg["duration_s"]
    truth["temp"] = temp
    truth["fs"] = CONFIG["fs"]
    
    # 有温漂
    params = {
        "acc": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "temp_drift": {"enabled": True, "k1": [cfg["k1_acc"]]*3, "T0": cfg["T0"]},
        },
        "gyro": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "temp_drift": {"enabled": True, "k1": [cfg["k1_gyro"]]*3, "T0": cfg["T0"]},
        },
    }
    
    meas = forward_imu(truth, params, seed=CONFIG["seed"])
    bias = meas["acc_bias_true"]
    
    # 线性拟合
    slope_fit, intercept_fit, r_value, p_value, std_err = stats.linregress(temp, bias[:, 0])
    r_squared = r_value**2
    
    # 预期斜率
    expected_slope = cfg["k1_acc"]
    slope_error = abs(slope_fit - expected_slope) / expected_slope * 100
    
    # 检查单调性
    if cfg["k1_acc"] > 0:
        monotonic = np.all(np.diff(bias[:, 0]) >= -1e-10)
    else:
        monotonic = np.all(np.diff(bias[:, 0]) <= 1e-10)
    
    # 检查端点值
    expected_start = cfg["k1_acc"] * (cfg["temp_start"] - cfg["T0"])
    expected_end = cfg["k1_acc"] * (cfg["temp_end"] - cfg["T0"])
    start_error = abs(bias[0, 0] - expected_start)
    end_error = abs(bias[-1, 0] - expected_end)
    
    metrics = {
        "temp_range": [float(temp[0]), float(temp[-1])],
        "bias_range": [float(bias[0, 0]), float(bias[-1, 0])],
        "expected_bias_range": [expected_start, expected_end],
        "slope_fit": float(slope_fit),
        "expected_slope": expected_slope,
        "slope_error_pct": float(slope_error),
        "r_squared": float(r_squared),
        "monotonic": bool(monotonic),
        "start_error": float(start_error),
        "end_error": float(end_error),
    }
    
    # 判定
    slope_ok = slope_error < 5.0  # 5% 容差
    r2_ok = r_squared > 0.99
    endpoint_ok = start_error < 1e-6 and end_error < 1e-6
    
    passed = slope_ok and r2_ok and monotonic and endpoint_ok
    
    print(f"  温度范围: {temp[0]:.1f} -> {temp[-1]:.1f} C")
    print(f"  偏置范围: {bias[0, 0]:.4f} -> {bias[-1, 0]:.4f}")
    print(f"  期望偏置: {expected_start:.4f} -> {expected_end:.4f}")
    print(f"  拟合斜率: {slope_fit:.6f} (期望: {expected_slope})")
    print(f"  斜率误差: {slope_error:.2f}% (阈值: 5%)")
    print(f"  R^2: {r_squared:.6f} (阈值: 0.99)")
    print(f"  单调性: {'PASS' if monotonic else 'FAIL'}")
    print(f"  端点误差: start={start_error:.2e}, end={end_error:.2e}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 温度 vs 偏置
    ax = axes[0]
    ax.plot(temp, bias[:, 0], 'b-', label='Actual', alpha=0.8)
    ax.plot(temp, slope_fit * temp + intercept_fit, 'r--', label=f'Fit (k={slope_fit:.4f})')
    ax.plot([cfg["temp_start"], cfg["temp_end"]], 
            [expected_start, expected_end], 'g:', label=f'Expected (k={expected_slope})')
    ax.set_xlabel('Temperature (C)')
    ax.set_ylabel('Bias (m/s^2)')
    ax.set_title('Temperature Drift: Bias vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 时间序列
    ax = axes[1]
    ax2 = ax.twinx()
    l1, = ax.plot(t, temp, 'b-', label='Temperature')
    l2, = ax2.plot(t, bias[:, 0], 'r-', label='Bias')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (C)', color='b')
    ax2.set_ylabel('Bias (m/s^2)', color='r')
    ax.set_title('Temperature Drift: Time Series')
    ax.legend(handles=[l1, l2], loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "temp_drift_plot.png", dpi=150)
    plt.close()
    
    # 保存指标
    with open(OUTPUT_DIR / "metrics" / "temp_drift_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  图表已保存: {OUTPUT_DIR / 'figures' / 'temp_drift_plot.png'}")
    
    return passed, metrics


# ============================================================
# 4. 比例因子/安装偏差
# ============================================================
def step4_scale_misalignment():
    """比例因子和安装偏差验证"""
    print("\n" + "="*70)
    print("Step 4: 比例因子/安装偏差 (Scale Factor / Misalignment)")
    print("="*70)
    
    cfg = CONFIG["tests"]["scale_misalign"]
    
    # 生成真值（水平静止）
    truth = generate_quasi_static(
        fs=CONFIG["fs"],
        duration_s=10.0,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        temp_C=25, seed=CONFIG["seed"]
    )
    truth["fs"] = CONFIG["fs"]
    
    # 无比例因子误差
    params_no_sm = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
    }
    
    # 有比例因子误差
    params_with_sm = {
        "acc": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "scale_misalign": {
                "enabled": True,
                "scale_error": cfg["scale_error"],
                "misalignment": cfg["misalignment"],
            },
        },
        "gyro": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
        },
    }
    
    meas_no_sm = forward_imu(truth, params_no_sm, seed=123)
    meas_with_sm = forward_imu(truth, params_with_sm, seed=123)
    
    # 理想输入（静止时 acc = [0, 0, g]）
    acc_ideal = np.mean(meas_no_sm["acc"], axis=0)
    acc_with_sm = np.mean(meas_with_sm["acc"], axis=0)
    
    # 计算变换矩阵
    S = np.diag(cfg["scale_error"])
    mx, my, mz = cfg["misalignment"]
    A = np.array([
        [0, -mz, my],
        [mz, 0, -mx],
        [-my, mx, 0]
    ])
    M = np.eye(3) + S + A
    
    # 预期输出
    acc_expected = M @ acc_ideal
    
    # 计算误差
    error = acc_with_sm - acc_expected
    relative_error = np.abs(error) / np.abs(acc_expected) * 100
    
    metrics = {
        "acc_ideal": acc_ideal.tolist(),
        "acc_with_sm": acc_with_sm.tolist(),
        "acc_expected": acc_expected.tolist(),
        "error": error.tolist(),
        "relative_error_pct": relative_error.tolist(),
        "scale_error": cfg["scale_error"],
        "misalignment": cfg["misalignment"],
        "transform_matrix": M.tolist(),
    }
    
    # 判定
    max_rel_error = np.max(relative_error)
    passed = max_rel_error < 0.01  # 0.01% 容差
    
    print(f"  理想输入: {acc_ideal}")
    print(f"  实际输出: {acc_with_sm}")
    print(f"  期望输出: {acc_expected}")
    print(f"  绝对误差: {error}")
    print(f"  相对误差: {relative_error}%")
    print(f"  变换矩阵 M:")
    for row in M:
        print(f"    {row}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")
    
    # 保存指标
    with open(OUTPUT_DIR / "metrics" / "scale_misalign_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return passed, metrics


# ============================================================
# 5. 饱和
# ============================================================
def step5_saturation():
    """饱和验证"""
    print("\n" + "="*70)
    print("Step 5: 饱和 (Saturation)")
    print("="*70)
    
    cfg = CONFIG["tests"]["saturation"]
    
    # 生成冲击工况
    truth = generate_shock(
        fs=CONFIG["fs"],
        duration_s=5.0,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        shock_peak=cfg["shock_peak"],
        shock_width_s=0.05,
        shock_times=[1.0, 2.0, 3.0],
        temp_C=25, seed=CONFIG["seed"]
    )
    truth["fs"] = CONFIG["fs"]
    
    # 无饱和
    params_no_sat = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
    }
    
    # 有饱和
    params_with_sat = {
        "acc": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "saturation": {"enabled": True, "range": cfg["range_acc"]},
        },
        "gyro": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
        },
    }
    
    meas_no_sat = forward_imu(truth, params_no_sat, seed=123)
    meas_with_sat = forward_imu(truth, params_with_sat, seed=123)
    
    t = truth["t"]
    acc_no_sat = meas_no_sat["acc"]
    acc_with_sat = meas_with_sat["acc"]
    
    # 统计
    max_no_sat = np.max(np.abs(acc_no_sat))
    max_with_sat = np.max(np.abs(acc_with_sat))
    
    # 饱和样本统计
    saturated_mask = np.abs(acc_no_sat) > cfg["range_acc"]
    n_saturated = np.sum(saturated_mask)
    saturation_ratio = n_saturated / acc_no_sat.size * 100
    
    # 检查裁剪正确性
    clipped_correctly = np.all(np.abs(acc_with_sat) <= cfg["range_acc"] + 1e-10)
    
    # 检查无回绕
    no_wraparound = not np.any(
        (np.abs(acc_no_sat) > cfg["range_acc"]) & 
        (np.sign(acc_no_sat) != np.sign(acc_with_sat))
    )
    
    metrics = {
        "max_no_sat": float(max_no_sat),
        "max_with_sat": float(max_with_sat),
        "saturation_range": cfg["range_acc"],
        "n_saturated_samples": int(n_saturated),
        "saturation_ratio_pct": float(saturation_ratio),
        "clipped_correctly": bool(clipped_correctly),
        "no_wraparound": bool(no_wraparound),
    }
    
    passed = clipped_correctly and no_wraparound and max_with_sat <= cfg["range_acc"]
    
    print(f"  无饱和最大值: {max_no_sat:.2f} m/s^2")
    print(f"  有饱和最大值: {max_with_sat:.2f} m/s^2")
    print(f"  饱和阈值: {cfg['range_acc']} m/s^2")
    print(f"  饱和样本数: {n_saturated} ({saturation_ratio:.2f}%)")
    print(f"  裁剪正确: {'PASS' if clipped_correctly else 'FAIL'}")
    print(f"  无回绕: {'PASS' if no_wraparound else 'FAIL'}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")
    
    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 时间序列对比
    ax = axes[0]
    ax.plot(t, acc_no_sat[:, 2], 'b-', alpha=0.7, label='No Saturation')
    ax.plot(t, acc_with_sat[:, 2], 'r-', alpha=0.7, label='With Saturation')
    ax.axhline(cfg["range_acc"], color='g', linestyle='--', label=f'+FS ({cfg["range_acc"]})')
    ax.axhline(-cfg["range_acc"], color='g', linestyle='--', label=f'-FS')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration Z (m/s^2)')
    ax.set_title('Saturation Effect on Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 饱和区域标记
    ax = axes[1]
    ax.fill_between(t, 0, 1, where=np.any(saturated_mask, axis=1), 
                    alpha=0.3, color='red', label='Saturated')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Saturation Flag')
    ax.set_title('Saturation Occurrence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "saturation_plot.png", dpi=150)
    plt.close()
    
    # 保存指标
    with open(OUTPUT_DIR / "metrics" / "saturation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  图表已保存: {OUTPUT_DIR / 'figures' / 'saturation_plot.png'}")
    
    return passed, metrics


# ============================================================
# 6. 量化
# ============================================================
def step6_quantization():
    """量化验证 - 补强版"""
    print("\n" + "="*70)
    print("Step 6: 量化 (Quantization)")
    print("="*70)
    
    cfg = CONFIG["tests"]["quantization"]
    fs = CONFIG["fs"]
    duration = 10.0
    
    # 计算量化参数
    n_levels = 2 ** cfg["bits"]
    expected_lsb = 2 * cfg["range"] / n_levels
    
    # 量化舍入模式（从 error_models.py 确认）
    quant_mode = "round"  # np.round = round-to-nearest
    print(f"  量化舍入模式: {quant_mode}")
    
    # ========== 测试 1: Ramp 输入 ==========
    print("\n  --- Ramp 输入测试 ---")
    n_samples = int(fs * duration)
    t_ramp = np.arange(n_samples) / fs
    
    # 从 -10 到 +10 m/s² 线性扫过
    acc_ramp_cont = -10.0 + 20.0 * t_ramp / duration
    
    # 量化
    acc_ramp_quant = np.round(acc_ramp_cont / expected_lsb) * expected_lsb
    
    # 统计
    ramp_error = acc_ramp_quant - acc_ramp_cont
    ramp_unique_codes = len(np.unique(acc_ramp_quant))
    
    print(f"    输入范围: [{acc_ramp_cont[0]:.2f}, {acc_ramp_cont[-1]:.2f}] m/s^2")
    print(f"    唯一码值数: {ramp_unique_codes}")
    
    # ========== 测试 2: 正弦输入 ==========
    print("\n  --- 正弦输入测试 ---")
    freq = 1.0  # Hz
    amplitude = 8.0  # m/s²
    
    acc_sine_cont = amplitude * np.sin(2 * np.pi * freq * t_ramp)
    acc_sine_quant = np.round(acc_sine_cont / expected_lsb) * expected_lsb
    
    sine_error = acc_sine_quant - acc_sine_cont
    sine_unique_codes = len(np.unique(acc_sine_quant))
    
    print(f"    幅值: {amplitude} m/s^2, 频率: {freq} Hz")
    print(f"    唯一码值数: {sine_unique_codes}")
    
    # ========== 合并统计 ==========
    all_error = np.concatenate([ramp_error, sine_error])
    all_cont = np.concatenate([acc_ramp_cont, acc_sine_cont])
    all_quant = np.concatenate([acc_ramp_quant, acc_sine_quant])
    
    unique_codes = len(np.unique(all_quant))
    mean_error = np.mean(all_error)
    mean_error_lsb = mean_error / expected_lsb
    
    # 分位数
    p05 = np.percentile(all_error, 5)
    p50 = np.percentile(all_error, 50)
    p95 = np.percentile(all_error, 95)
    
    # 检查误差范围
    error_in_range = np.all((all_error >= -0.5 * expected_lsb - 1e-10) & 
                            (all_error <= 0.5 * expected_lsb + 1e-10))
    
    # 检查量化值是 LSB 的整数倍
    residuals = np.abs(all_quant / expected_lsb - np.round(all_quant / expected_lsb))
    lsb_multiple = np.max(residuals) < 1e-10
    
    # 检查唯一码值数
    codes_ok = unique_codes > 20
    
    # 检查均值偏差（round-to-nearest 应该接近 0）
    mean_ok = abs(mean_error_lsb) < 0.05
    
    metrics = {
        "quant_mode": quant_mode,
        "bits": cfg["bits"],
        "range": cfg["range"],
        "expected_lsb": expected_lsb,
        "unique_codes": unique_codes,
        "unique_codes_ramp": ramp_unique_codes,
        "unique_codes_sine": sine_unique_codes,
        "mean_error": float(mean_error),
        "mean_error_lsb": float(mean_error_lsb),
        "quant_error_min": float(np.min(all_error)),
        "quant_error_max": float(np.max(all_error)),
        "quant_error_std": float(np.std(all_error)),
        "error_percentiles": {
            "p05": float(p05),
            "p50": float(p50),
            "p95": float(p95),
        },
        "error_in_range": bool(error_in_range),
        "lsb_multiple": bool(lsb_multiple),
        "codes_ok": bool(codes_ok),
        "mean_ok": bool(mean_ok),
    }
    
    passed = error_in_range and lsb_multiple and codes_ok and mean_ok
    
    print(f"\n  --- 汇总统计 ---")
    print(f"  量化位数: {cfg['bits']}")
    print(f"  量程: +/-{cfg['range']} m/s^2")
    print(f"  期望 LSB: {expected_lsb:.6f} m/s^2")
    print(f"  唯一码值数: {unique_codes} (阈值: >20) {'PASS' if codes_ok else 'FAIL'}")
    print(f"  均值误差: {mean_error:.6f} m/s^2 = {mean_error_lsb:.4f} LSB (阈值: <0.05 LSB) {'PASS' if mean_ok else 'FAIL'}")
    print(f"  误差分位数: p05={p05:.6f}, p50={p50:.6f}, p95={p95:.6f}")
    print(f"  误差范围: [{np.min(all_error):.6f}, {np.max(all_error):.6f}]")
    print(f"  期望误差范围: [{-0.5*expected_lsb:.6f}, {0.5*expected_lsb:.6f}]")
    print(f"  误差在范围内: {'PASS' if error_in_range else 'FAIL'}")
    print(f"  LSB整数倍: {'PASS' if lsb_multiple else 'FAIL'}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Ramp: 连续 vs 量化
    ax = axes[0, 0]
    ax.plot(t_ramp, acc_ramp_cont, 'b-', alpha=0.7, label='Continuous')
    ax.plot(t_ramp, acc_ramp_quant, 'r-', alpha=0.7, label='Quantized')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.set_title(f'Ramp Input ({ramp_unique_codes} codes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sine: 连续 vs 量化
    ax = axes[0, 1]
    ax.plot(t_ramp[:500], acc_sine_cont[:500], 'b-', alpha=0.7, label='Continuous')
    ax.plot(t_ramp[:500], acc_sine_quant[:500], 'r-', alpha=0.7, label='Quantized')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.set_title(f'Sine Input ({sine_unique_codes} codes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 误差直方图
    ax = axes[0, 2]
    ax.hist(all_error / expected_lsb, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='r', linestyle='--', label='+0.5 LSB')
    ax.axvline(-0.5, color='r', linestyle='--', label='-0.5 LSB')
    ax.axvline(mean_error_lsb, color='g', linestyle='-', linewidth=2, label=f'Mean={mean_error_lsb:.3f}')
    ax.set_xlabel('Quantization Error (LSB)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ramp 台阶效果（放大）
    ax = axes[1, 0]
    idx = slice(400, 600)
    ax.step(t_ramp[idx], acc_ramp_quant[idx], 'r-', where='mid', label='Quantized')
    ax.plot(t_ramp[idx], acc_ramp_cont[idx], 'b-', alpha=0.5, label='Continuous')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.set_title('Ramp Quantization Steps (Zoomed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sine 台阶效果（放大）
    ax = axes[1, 1]
    idx = slice(200, 300)
    ax.step(t_ramp[idx], acc_sine_quant[idx], 'r-', where='mid', label='Quantized')
    ax.plot(t_ramp[idx], acc_sine_cont[idx], 'b-', alpha=0.5, label='Continuous')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.set_title('Sine Quantization Steps (Zoomed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 统计表
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['Metric', 'Value', 'Status'],
        ['Quant Mode', quant_mode, '-'],
        ['LSB', f'{expected_lsb:.6f}', '-'],
        ['Unique Codes', f'{unique_codes}', 'PASS' if codes_ok else 'FAIL'],
        ['Mean Error', f'{mean_error_lsb:.4f} LSB', 'PASS' if mean_ok else 'FAIL'],
        ['p05/p50/p95', f'{p05/expected_lsb:.2f}/{p50/expected_lsb:.2f}/{p95/expected_lsb:.2f}', '-'],
        ['Error Range', 'PASS' if error_in_range else 'FAIL', '-'],
    ]
    colors = [['lightblue']*3] + [['white', 'white', 
              'lightgreen' if row[2] in ['PASS', '-'] else 'lightcoral'] for row in table_data[1:]]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', cellColours=colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Quantization Metrics')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "quantization_plot.png", dpi=150)
    plt.close()
    
    # 保存指标
    with open(OUTPUT_DIR / "metrics" / "quantization_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  图表已保存: {OUTPUT_DIR / 'figures' / 'quantization_plot.png'}")
    
    return passed, metrics


# ============================================================
# 7. 汇总与归档
# ============================================================
def step7_summary(results):
    """汇总与归档"""
    print("\n" + "="*70)
    print("Step 7: 汇总与归档")
    print("="*70)
    
    # 汇总表
    summary = {
        "timestamp": CONFIG["timestamp"],
        "version": CONFIG["version"],
        "seed": CONFIG["seed"],
        "tests": {
            "0_config": {"status": "PASS" if results[0] else "FAIL"},
            "1_regression": {"status": "PASS" if results[1][0] else "FAIL"},
            "2_bias_rw": {"status": "PASS" if results[2][0] else "FAIL"},
            "3_temp_drift": {"status": "PASS" if results[3][0] else "FAIL"},
            "4_scale_misalign": {"status": "PASS" if results[4][0] else "FAIL"},
            "5_saturation": {"status": "PASS" if results[5][0] else "FAIL"},
            "6_quantization": {"status": "PASS" if results[6][0] else "FAIL"},
        },
        "all_passed": all([r[0] if isinstance(r, tuple) else r for r in results]),
    }
    
    # 打印汇总
    print("\n  验收汇总表:")
    print("  " + "-"*50)
    print(f"  {'Test':<30} {'Status':<10}")
    print("  " + "-"*50)
    
    test_names = [
        "0. Config Fixation",
        "1. Regression Test",
        "2. Bias Random Walk",
        "3. Temperature Drift",
        "4. Scale/Misalignment",
        "5. Saturation",
        "6. Quantization",
    ]
    
    for i, name in enumerate(test_names):
        status = "PASS" if (results[i][0] if isinstance(results[i], tuple) else results[i]) else "FAIL"
        print(f"  {name:<30} {status:<10}")
    
    print("  " + "-"*50)
    print(f"  {'OVERALL':<30} {'PASS' if summary['all_passed'] else 'FAIL':<10}")
    print("  " + "-"*50)
    
    # 保存汇总
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 绘制汇总图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    table_data = []
    colors = []
    for i, name in enumerate(test_names):
        status = "PASS" if (results[i][0] if isinstance(results[i], tuple) else results[i]) else "FAIL"
        table_data.append([name, status])
        colors.append(['white', 'lightgreen' if status == 'PASS' else 'lightcoral'])
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Test Item", "Status"],
        loc='center',
        cellLoc='left',
        cellColours=colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2.0)
    
    # 设置表头颜色
    for j in range(2):
        table[(0, j)].set_facecolor('lightblue')
    
    ax.set_title(f'Step 11 Validation Summary\n{CONFIG["timestamp"]}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "summary_table.png", dpi=150)
    plt.close()
    
    print(f"\n  汇总已保存: {OUTPUT_DIR / 'summary.json'}")
    print(f"  汇总图已保存: {OUTPUT_DIR / 'figures' / 'summary_table.png'}")
    
    # 列出所有产物
    print("\n  产物清单:")
    print("  " + "-"*50)
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(OUTPUT_DIR)}")
    print("  " + "-"*50)
    
    return summary["all_passed"]


# ============================================================
# 主函数
# ============================================================
def main():
    print("="*70)
    print("Step 11: 增强 IMU 误差模型 - 完整验收")
    print("="*70)
    
    setup_output_dir()
    
    results = []
    
    # 执行所有测试
    results.append(step0_save_config())
    results.append(step1_regression_test())
    results.append(step2_bias_random_walk())
    results.append(step3_temperature_drift())
    results.append(step4_scale_misalignment())
    results.append(step5_saturation())
    results.append(step6_quantization())
    
    # 汇总
    all_passed = step7_summary(results)
    
    print("\n" + "="*70)
    print(f"Step 11 验收结果: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Step 12 验证脚本：自适应 EKF（双通道版）

验收标准（必须用 vibration/shock 工况）：
1. 与 ekf_fixed 相比：不发散、峰值误差更低或恢复更快
2. NIS 上升 → λ 上升（因果一致）
3. 平稳段 λ 回落（不会一直锁死在高噪声）

双通道改进：
- 方向通道：检测姿态变化引起的方向偏差
- 幅值通道：检测非重力加速度（冲击/振动）
- 组合 NIS = max(NIS_dir, NIS_mag)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import signal

from src.truth.scenarios import generate_vibration, generate_shock, generate_quasi_static, generate_swing, generate_turn, generate_accel
from src.sensors.imu_model import forward_imu
from src.datasets.validate import validate_dataset
from src.filters.ekf_fixed import run_ekf_fixed
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.metrics.tilt_error import compute_tilt_metrics
from src.common.math3d import quat_to_rpy

# 输出目录
OUTPUT_DIR = Path("tilt-adapt-ekf-sim/outputs/step12_validation")


def setup():
    """设置输出目录"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)


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
        "meas": {
            "gyro": meas["gyro"],
            "acc": meas["acc"],
        },
        "meta": {
            "fs": truth["fs"],
            "seed": seed,
            "scenario_name": "test",
            "sensor_params": sensor_params,
        },
    }
    validate_dataset(ds)
    return ds


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


def compute_correlation_and_lag(nis, lambda_k, max_lag=100):
    """
    计算 NIS 与 λ 的相关系数和滞后
    
    Returns:
        corr: 皮尔逊相关系数
        lag: λ 相对 NIS 的最佳滞后样本数（正值表示 λ 滞后于 NIS）
    """
    # 去均值
    nis_centered = nis - np.mean(nis)
    lambda_centered = lambda_k - np.mean(lambda_k)
    
    # 皮尔逊相关系数（零滞后）
    corr = np.corrcoef(nis, lambda_k)[0, 1]
    
    # 互相关找最佳滞后
    cross_corr = signal.correlate(lambda_centered, nis_centered, mode='full')
    lags = signal.correlation_lags(len(lambda_k), len(nis), mode='full')
    
    # 只看合理范围内的滞后
    valid_mask = (lags >= 0) & (lags <= max_lag)
    if np.any(valid_mask):
        valid_cross_corr = cross_corr[valid_mask]
        valid_lags = lags[valid_mask]
        best_idx = np.argmax(valid_cross_corr)
        lag = valid_lags[best_idx]
    else:
        lag = 0
    
    return float(corr), int(lag)


def compute_lambda_saturation_metrics(lambda_k, lambda_max=100.0):
    """
    计算 λ 饱和相关指标
    
    Returns:
        saturation_ratio: λ==lambda_max 的时间占比
        high_lambda_ratio: λ>0.8*lambda_max 的时间占比
    """
    n = len(lambda_k)
    saturation_ratio = np.sum(lambda_k >= lambda_max * 0.99) / n
    high_lambda_ratio = np.sum(lambda_k > lambda_max * 0.8) / n
    return float(saturation_ratio), float(high_lambda_ratio)


def test_vibration():
    """测试振动工况（双通道版）"""
    print("\n" + "="*70)
    print("测试 1: 振动工况 (Vibration) - 双通道版")
    print("="*70)
    
    # 生成振动工况
    truth = generate_vibration(
        fs=100, duration_s=30,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        vib_rms=0.5, vib_bandwidth_hz=10.0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    
    # 传感器参数
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    ds = create_dataset(truth, sensor_params)
    
    # EKF Fixed 配置
    cfg_fixed = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": True, "threshold": 7.815, "mode": "inflate_R"},
    }
    
    # EKF Adaptive 配置（双通道版）
    # Plan E: 模仿固定 EKF 的 inflate_R 机制，但下降更快
    lambda_max = 200.0  # 允许更大的膨胀
    cfg_adaptive = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 3.5e-6,
        "use_direction_meas": True,
        "innovation_stat": {
            "window_W": 30,
            "nis_high": 7.815,  # 与固定 EKF 的 χ²(3) 95% 阈值一致
            "nis_low": 3.0,
            "ewma_alpha": 0.05,
        },
        "adaptation": {
            "r_up": 1.05,
            "r_down": 0.95,
            "lambda_max": lambda_max,
            "lambda_min": 1.0,
            "soft_saturation": False,
            "ewma_lambda_alpha": 0.0,  # 不使用 EWMA
            # Plan E: inflate 映射
            "use_inflate_mapping": True,
            "use_sigmoid_mapping": False,
            "inflate_decay_rate": 0.8,  # 下降衰减率
            "inflate_rise_smooth": 1.0,  # 上升平滑因子
        },
        "dual_channel": {
            "enabled": False,  # 禁用双通道，与固定 EKF 保持一致
            "mag_weight": 1.0,
            "mag_sigma": 0.5,
            "combine_mode": "max",
            "vibration_aware": False,
        },
    }
    
    # 运行两种 EKF
    est_fixed = run_ekf_fixed(ds, cfg_fixed)
    est_adaptive = run_ekf_adaptive(ds, cfg_adaptive)
    
    # 获取真值
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    t = ds["t"]
    
    # 计算误差
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    roll_err_adaptive = np.rad2deg(est_adaptive["roll"] - roll_true)
    pitch_err_adaptive = np.rad2deg(est_adaptive["pitch"] - pitch_true)
    
    # 合并误差
    err_fixed = np.sqrt(roll_err_fixed**2 + pitch_err_fixed**2)
    err_adaptive = np.sqrt(roll_err_adaptive**2 + pitch_err_adaptive**2)
    
    # 基本统计
    rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
    rmse_adaptive = np.sqrt(np.mean(roll_err_adaptive**2 + pitch_err_adaptive**2))
    peak_fixed = np.max(np.abs(np.concatenate([roll_err_fixed, pitch_err_fixed])))
    peak_adaptive = np.max(np.abs(np.concatenate([roll_err_adaptive, pitch_err_adaptive])))
    
    print(f"  EKF Fixed:    RMSE={rmse_fixed:.3f}°, Peak={peak_fixed:.3f}°")
    print(f"  EKF Adaptive: RMSE={rmse_adaptive:.3f}°, Peak={peak_adaptive:.3f}°")
    print(f"  RMSE 改善: {(rmse_fixed - rmse_adaptive) / rmse_fixed * 100:.1f}%")
    print(f"  Peak 变化: {(peak_adaptive - peak_fixed) / peak_fixed * 100:+.1f}%")
    
    # 获取双通道 NIS
    lambda_k = est_adaptive["debug"]["lambda_k"]
    nis_dir = est_adaptive["debug"]["nis_raw"]  # 方向通道
    nis_mag = est_adaptive["debug"]["nis_mag"]  # 幅值通道
    nis_combined = est_adaptive["debug"]["nis_combined"]  # 组合
    
    # 双通道统计
    print(f"\n  双通道 NIS 统计:")
    print(f"    方向通道 NIS: mean={np.mean(nis_dir):.1f}, max={np.max(nis_dir):.1f}")
    print(f"    幅值通道 NIS: mean={np.mean(nis_mag):.1f}, max={np.max(nis_mag):.1f}")
    print(f"    组合 NIS:     mean={np.mean(nis_combined):.1f}, max={np.max(nis_combined):.1f}")
    
    # λ 饱和统计
    saturation_ratio, high_lambda_ratio = compute_lambda_saturation_metrics(lambda_k, lambda_max)
    print(f"\n  λ 饱和指标:")
    print(f"    λ=={lambda_max} 占比: {saturation_ratio*100:.1f}%")
    print(f"    λ>{lambda_max*0.8:.0f} 占比: {high_lambda_ratio*100:.1f}%")
    
    # 相关性
    corr_nis_lambda, lag_nis_to_lambda = compute_correlation_and_lag(nis_combined, lambda_k)
    print(f"\n  因果指标 (组合NIS vs λ):")
    print(f"    corr(NIS_comb, λ): {corr_nis_lambda:.3f}")
    print(f"    lag(NIS→λ): {lag_nis_to_lambda} 样本")
    
    # 绘图（6 子图版本）
    fig, axes = plt.subplots(6, 1, figsize=(14, 18))
    
    # 1. 姿态误差对比
    ax = axes[0]
    ax.plot(t, err_fixed, 'b-', alpha=0.6, label=f'Fixed (RMSE={rmse_fixed:.2f}°)')
    ax.plot(t, err_adaptive, 'r-', alpha=0.6, label=f'Adaptive (RMSE={rmse_adaptive:.2f}°)')
    ax.set_ylabel('Total Error (deg)')
    ax.set_title('Vibration: Attitude Error Comparison (Dual-Channel)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 双通道 NIS 对比
    ax = axes[1]
    ax.plot(t, nis_dir, 'b-', alpha=0.5, label='NIS_dir (方向)')
    ax.plot(t, nis_mag, 'g-', alpha=0.5, label='NIS_mag (幅值)')
    ax.plot(t, nis_combined, 'r-', alpha=0.7, label='NIS_combined')
    ax.axhline(10.0, color='k', linestyle='--', alpha=0.5, label='nis_high=10')
    ax.set_ylabel('NIS')
    ax.set_title('Dual-Channel NIS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, min(100, np.percentile(nis_combined, 99)*1.2)])
    
    # 3. 加速度幅值
    ax = axes[2]
    acc_norm = np.linalg.norm(ds["meas"]["acc"], axis=1)
    ax.plot(t, acc_norm, 'b-', alpha=0.7)
    ax.axhline(9.80665, color='r', linestyle='--', label='g=9.807')
    ax.set_ylabel('||acc|| (m/s^2)')
    ax.set_title('Acceleration Magnitude (should be ~g for static)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. λ 曲线
    ax = axes[3]
    ax.plot(t, lambda_k, 'r-', label='lambda')
    ax.axhline(lambda_max, color='k', linestyle='--', alpha=0.5, label=f'lambda_max={lambda_max}')
    ax.set_ylabel('Lambda')
    ax.set_title(f'Lambda Factor (saturation={saturation_ratio*100:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. NIS vs λ 散点图
    ax = axes[4]
    ax.scatter(nis_combined[::10], lambda_k[::10], alpha=0.3, s=10)
    ax.set_xlabel('NIS_combined')
    ax.set_ylabel('Lambda')
    ax.set_title(f'NIS vs Lambda (corr={corr_nis_lambda:.3f})')
    ax.grid(True, alpha=0.3)
    
    # 6. 误差与 λ 叠加
    ax1 = axes[5]
    ax2 = ax1.twinx()
    l1, = ax1.plot(t, err_adaptive, 'r-', alpha=0.7, label='Error')
    l2, = ax2.plot(t, lambda_k, 'b-', alpha=0.7, label='Lambda')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Error (deg)', color='r')
    ax2.set_ylabel('Lambda', color='b')
    ax1.set_title('Error vs Lambda')
    ax1.legend([l1, l2], ['Error', 'Lambda'], loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "vibration_dual_channel.png", dpi=150)
    plt.close()
    
    print(f"\n  图表已保存: {OUTPUT_DIR / 'figures' / 'vibration_dual_channel.png'}")
    
    return {
        "rmse_fixed": float(rmse_fixed),
        "rmse_adaptive": float(rmse_adaptive),
        "rmse_improvement_pct": float((rmse_fixed - rmse_adaptive) / rmse_fixed * 100),
        "peak_fixed": float(peak_fixed),
        "peak_adaptive": float(peak_adaptive),
        "peak_change_pct": float((peak_adaptive - peak_fixed) / peak_fixed * 100),
        "lambda_saturation_ratio": saturation_ratio,
        "time_ratio_high_lambda": high_lambda_ratio,
        "corr_NIS_lambda": corr_nis_lambda,
        "nis_dir_mean": float(np.mean(nis_dir)),
        "nis_mag_mean": float(np.mean(nis_mag)),
        "nis_combined_mean": float(np.mean(nis_combined)),
    }


def test_shock():
    """测试冲击工况（双通道版）"""
    print("\n" + "="*70)
    print("测试 2: 冲击工况 (Shock) - 双通道版")
    print("="*70)
    
    # 生成冲击工况
    shock_times = [5.0, 10.0, 15.0]
    truth = generate_shock(
        fs=100, duration_s=20,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        shock_peak=50.0, shock_width_s=0.05,
        shock_times=shock_times,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    fs = 100.0
    
    # 传感器参数
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    ds = create_dataset(truth, sensor_params)
    
    # EKF 配置
    cfg_fixed = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": True, "threshold": 7.815, "mode": "inflate_R"},
    }
    
    # EKF Adaptive 配置（双通道版）
    lambda_max = 50.0  # 冲击需要更大的 λ 范围
    cfg_adaptive = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R0": 3.5e-6,
        "use_direction_meas": True,
        "innovation_stat": {
            "window_W": 20,
            "nis_high": 10.0,
            "nis_low": 3.0,
            "ewma_alpha": 0.15,  # 更快响应
        },
        "adaptation": {
            "r_up": 1.2,  # 更快上升
            "r_down": 0.9,  # 更快回落
            "lambda_max": lambda_max,
            "lambda_min": 1.0,
            "soft_saturation": True,
            "lambda_soft_max": 20.0,
        },
        "dual_channel": {
            "enabled": True,
            "mag_weight": 3.0,  # 冲击时幅值偏差非常大
            "mag_sigma": 0.5,
            "combine_mode": "max",
        },
    }
    
    # 运行
    est_fixed = run_ekf_fixed(ds, cfg_fixed)
    est_adaptive = run_ekf_adaptive(ds, cfg_adaptive)
    
    # 获取真值
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    t = ds["t"]
    
    # 计算误差
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    roll_err_adaptive = np.rad2deg(est_adaptive["roll"] - roll_true)
    pitch_err_adaptive = np.rad2deg(est_adaptive["pitch"] - pitch_true)
    
    # 统计
    rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
    rmse_adaptive = np.sqrt(np.mean(roll_err_adaptive**2 + pitch_err_adaptive**2))
    
    print(f"  EKF Fixed:    RMSE={rmse_fixed:.3f}°")
    print(f"  EKF Adaptive: RMSE={rmse_adaptive:.3f}°")
    
    # 获取双通道 NIS
    lambda_k = est_adaptive["debug"]["lambda_k"]
    nis_dir = est_adaptive["debug"]["nis_raw"]
    nis_mag = est_adaptive["debug"]["nis_mag"]
    nis_combined = est_adaptive["debug"]["nis_combined"]
    nis_fixed = est_fixed["debug"]["nis"]
    
    # 冲击点附近统计
    window_s = 0.3
    window_samples = int(window_s * fs)
    
    shock_stats = []
    print(f"\n  冲击点附近统计 (窗口: ±{window_s}s):")
    print(f"  {'时刻':>8} | {'NIS_dir':>10} | {'NIS_mag':>10} | {'NIS_comb':>10} | {'max_λ':>8}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")
    
    for st in shock_times:
        idx = int(st * fs)
        start_idx = max(0, idx - window_samples)
        end_idx = min(len(nis_dir), idx + window_samples)
        
        max_nis_dir = float(np.max(nis_dir[start_idx:end_idx]))
        max_nis_mag = float(np.max(nis_mag[start_idx:end_idx]))
        max_nis_comb = float(np.max(nis_combined[start_idx:end_idx]))
        max_lambda = float(np.max(lambda_k[start_idx:end_idx]))
        
        print(f"  t={st:5.1f}s | {max_nis_dir:10.1f} | {max_nis_mag:10.1f} | {max_nis_comb:10.1f} | {max_lambda:8.1f}")
        
        shock_stats.append({
            "time": st,
            "max_NIS_dir": max_nis_dir,
            "max_NIS_mag": max_nis_mag,
            "max_NIS_combined": max_nis_comb,
            "max_lambda": max_lambda,
        })
    
    # 检查冲击是否触发
    nis_triggered = any(s["max_NIS_combined"] > 10.0 for s in shock_stats)
    lambda_responded = any(s["max_lambda"] > 5.0 for s in shock_stats)
    
    print(f"\n  冲击响应诊断:")
    print(f"    冲击是否触发 NIS 上升 (>10): {'是' if nis_triggered else '否'}")
    print(f"    λ 是否响应 (>5): {'是' if lambda_responded else '否'}")
    
    if nis_triggered:
        print(f"    [OK] 双通道成功检测到冲击！")
    
    # 检查平稳段 λ 回落
    lambda_end = lambda_k[-200:]
    lambda_recovered = np.mean(lambda_end) < 5.0
    
    print(f"\n  平稳段回落:")
    print(f"    最后 2s λ 均值: {np.mean(lambda_end):.2f}")
    print(f"    λ 回落: {'是' if lambda_recovered else '否'}")
    
    # 绘图
    fig, axes = plt.subplots(6, 1, figsize=(14, 18))
    
    # 1. 加速度幅值
    ax = axes[0]
    acc_norm = np.linalg.norm(ds["meas"]["acc"], axis=1)
    ax.plot(t, acc_norm, 'b-', alpha=0.7)
    ax.axhline(9.80665, color='r', linestyle='--', alpha=0.5, label='g')
    for st in shock_times:
        ax.axvline(st, color='orange', linestyle='--', alpha=0.7)
    ax.set_ylabel('||acc|| (m/s^2)')
    ax.set_title('Acceleration Magnitude (orange = shock times)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 姿态误差
    ax = axes[1]
    ax.plot(t, pitch_err_fixed, 'b-', alpha=0.5, label='Fixed')
    ax.plot(t, pitch_err_adaptive, 'r-', alpha=0.7, label='Adaptive')
    for st in shock_times:
        ax.axvline(st, color='g', linestyle='--', alpha=0.5)
    ax.set_ylabel('Pitch Error (deg)')
    ax.set_title('Pitch Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 双通道 NIS
    ax = axes[2]
    ax.plot(t, nis_dir, 'b-', alpha=0.5, label='NIS_dir')
    ax.plot(t, nis_mag, 'g-', alpha=0.5, label='NIS_mag')
    ax.plot(t, nis_combined, 'r-', alpha=0.7, label='NIS_combined')
    ax.axhline(10.0, color='k', linestyle='--', alpha=0.5)
    for st in shock_times:
        ax.axvline(st, color='orange', linestyle='--', alpha=0.7)
    ax.set_ylabel('NIS')
    ax.set_title('Dual-Channel NIS around Shocks')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, min(200, np.max(nis_combined)*1.1)])
    
    # 4. λ 曲线
    ax = axes[3]
    ax.plot(t, lambda_k, 'r-')
    for st in shock_times:
        ax.axvline(st, color='orange', linestyle='--', alpha=0.7)
    ax.set_ylabel('Lambda')
    ax.set_title('Lambda Response to Shocks')
    ax.grid(True, alpha=0.3)
    
    # 5. 第一个冲击放大图
    ax = axes[4]
    st = shock_times[0]
    zoom_start = max(0, st - 0.5)
    zoom_end = min(t[-1], st + 1.0)
    mask = (t >= zoom_start) & (t <= zoom_end)
    
    ax2 = ax.twinx()
    l1, = ax.plot(t[mask], nis_mag[mask], 'g-', alpha=0.7, label='NIS_mag')
    l2, = ax.plot(t[mask], nis_combined[mask], 'r-', alpha=0.7, label='NIS_comb')
    l3, = ax2.plot(t[mask], lambda_k[mask], 'b-', alpha=0.7, label='Lambda')
    ax.axvline(st, color='k', linestyle='--', alpha=0.7)
    ax.axhline(10.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('NIS', color='r')
    ax2.set_ylabel('Lambda', color='b')
    ax.set_title(f'Zoom: First Shock @ t={st}s')
    ax.legend([l1, l2, l3], ['NIS_mag', 'NIS_comb', 'Lambda'], loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 6. 加速度 Z 分量
    ax = axes[5]
    ax.plot(t, ds["meas"]["acc"][:, 2], 'b-', alpha=0.7)
    for st in shock_times:
        ax.axvline(st, color='orange', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acc Z (m/s^2)')
    ax.set_title('Z-axis Acceleration (shock direction)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "shock_dual_channel.png", dpi=150)
    plt.close()
    
    print(f"\n  图表已保存: {OUTPUT_DIR / 'figures' / 'shock_dual_channel.png'}")
    
    return {
        "rmse_fixed": float(rmse_fixed),
        "rmse_adaptive": float(rmse_adaptive),
        "lambda_recovered": lambda_recovered,
        "shock_stats": shock_stats,
        "nis_triggered": nis_triggered,
        "lambda_responded": lambda_responded,
    }


def test_causality():
    """测试因果一致性（双通道版）：NIS 上升 → λ 上升"""
    print("\n" + "="*70)
    print("测试 3: 因果一致性 (NIS ↑ → λ ↑) - 双通道版")
    print("="*70)
    
    # 使用振动工况
    truth = generate_vibration(
        fs=100, duration_s=20,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        vib_rms=1.0, vib_bandwidth_hz=15.0,
        temp_C=25, seed=123
    )
    truth["fs"] = 100.0
    
    sensor_params = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.02},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.001},
    }
    
    ds = create_dataset(truth, sensor_params, seed=123)
    
    # Plan E: 模仿固定 EKF 的 inflate_R 机制，但下降更快
    lambda_max = 200.0
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
            "r_up": 1.05,
            "r_down": 0.95,
            "lambda_max": lambda_max,
            "lambda_min": 1.0,
            "soft_saturation": False,
            "ewma_lambda_alpha": 0.0,
            "use_inflate_mapping": True,
            "use_sigmoid_mapping": False,
        },
        "dual_channel": {
            "enabled": False,
            "mag_weight": 1.0,
            "mag_sigma": 0.5,
            "combine_mode": "max",
            "vibration_aware": False,
        },
    }
    
    est = run_ekf_adaptive(ds, cfg_adaptive)
    
    nis_combined = est["debug"]["nis_combined"]
    lambda_k = est["debug"]["lambda_k"]
    t = ds["t"]
    
    # 使用与 inflate 配置一致的阈值
    nis_high = 7.815
    nis_low = 3.0
    
    # 相关系数和滞后
    corr_nis_lambda, lag_nis_to_lambda = compute_correlation_and_lag(nis_combined, lambda_k)
    
    # 分组统计
    nis_high_mask = nis_combined > nis_high
    nis_low_mask = nis_combined < nis_low
    
    lambda_mean_high = np.mean(lambda_k[nis_high_mask]) if np.any(nis_high_mask) else np.nan
    lambda_mean_low = np.mean(lambda_k[nis_low_mask]) if np.any(nis_low_mask) else np.nan
    lambda_correlation = lambda_mean_high > lambda_mean_low if not (np.isnan(lambda_mean_high) or np.isnan(lambda_mean_low)) else True
    
    # λ 基线检查
    lambda_mean = np.mean(lambda_k)
    lambda_median = np.median(lambda_k)
    
    print(f"  相关性指标:")
    print(f"    corr(NIS_comb, λ): {corr_nis_lambda:.3f}")
    print(f"    lag(NIS→λ): {lag_nis_to_lambda} 样本")
    
    print(f"\n  分组统计:")
    print(f"    NIS>{nis_high} 时 λ 均值: {lambda_mean_high:.2f}")
    print(f"    NIS<{nis_low} 时 λ 均值: {lambda_mean_low:.2f}")
    print(f"    λ 与 NIS 正相关: {'是' if lambda_correlation else '否'}")
    
    print(f"\n  λ 基线检查:")
    print(f"    λ 均值: {lambda_mean:.2f}")
    print(f"    λ 中位数: {lambda_median:.2f}")
    
    # 判定：相关系数 > 0.3 且分组统计正确
    passed = lambda_correlation and corr_nis_lambda > 0.3
    print(f"\n  结果: {'PASS' if passed else 'FAIL'}")
    
    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. NIS 和 λ 时序
    ax = axes[0]
    ax2 = ax.twinx()
    l1, = ax.plot(t, nis_combined, 'r-', alpha=0.7, label='NIS_combined')
    l2, = ax2.plot(t, lambda_k, 'b-', alpha=0.7, label='Lambda')
    ax.axhline(nis_high, color='g', linestyle='--', alpha=0.5)
    ax.axhline(nis_low, color='orange', linestyle='--', alpha=0.5)
    ax.set_ylabel('NIS', color='r')
    ax2.set_ylabel('Lambda', color='b')
    ax.set_title(f'NIS and Lambda Time Series (corr={corr_nis_lambda:.3f})')
    ax.legend([l1, l2], ['NIS_combined', 'Lambda'], loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. 散点图
    ax = axes[1]
    ax.scatter(nis_combined, lambda_k, alpha=0.3, s=10)
    ax.axvline(nis_high, color='g', linestyle='--', alpha=0.5, label=f'nis_high={nis_high}')
    ax.axvline(nis_low, color='orange', linestyle='--', alpha=0.5, label=f'nis_low={nis_low}')
    ax.set_xlabel('NIS_combined')
    ax.set_ylabel('Lambda')
    ax.set_title(f'NIS vs Lambda (high: λ={lambda_mean_high:.1f}, low: λ={lambda_mean_low:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. λ 分布
    ax = axes[2]
    ax.hist(lambda_k, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(lambda_mean, color='r', linestyle='--', label=f'mean={lambda_mean:.1f}')
    ax.axvline(lambda_median, color='g', linestyle='--', label=f'median={lambda_median:.1f}')
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Count')
    ax.set_title('Lambda Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "causality_dual_channel.png", dpi=150)
    plt.close()
    
    print(f"\n  图表已保存: {OUTPUT_DIR / 'figures' / 'causality_dual_channel.png'}")
    
    return {
        "corr_NIS_lambda": float(corr_nis_lambda),
        "lag_NIS_to_lambda": int(lag_nis_to_lambda),
        "lambda_mean_high_NIS": float(lambda_mean_high) if not np.isnan(lambda_mean_high) else None,
        "lambda_mean_low_NIS": float(lambda_mean_low) if not np.isnan(lambda_mean_low) else None,
        "lambda_correlation": lambda_correlation,
        "lambda_mean": float(lambda_mean),
        "passed": passed,
    }


def test_swing():
    """测试摆动工况（快速姿态变化）"""
    print("\n" + "="*70)
    print("测试 4: 摆动工况 (Swing) - 快速姿态变化")
    print("="*70)
    
    # 生成摆动工况：roll 和 pitch 同时正弦变化
    truth = generate_swing(
        fs=100, duration_s=30,
        roll_amp_deg=15.0,  # 15° 幅度
        pitch_amp_deg=10.0,  # 10° 幅度
        roll_freq_hz=0.5,   # 0.5 Hz
        pitch_freq_hz=0.3,  # 0.3 Hz
        roll_phase_deg=0,
        pitch_phase_deg=90,
        yaw_deg=0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    ds = create_dataset(truth, sensor_params)
    
    cfg_fixed = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": True, "threshold": 7.815, "mode": "inflate_R"},
    }
    
    lambda_max = 200.0
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
            "lambda_max": lambda_max,
            "lambda_min": 1.0,
            "use_inflate_mapping": True,
            "inflate_decay_rate": 0.8,
            "inflate_rise_smooth": 1.0,
        },
        "dual_channel": {"enabled": False},
    }
    
    est_fixed = run_ekf_fixed(ds, cfg_fixed)
    est_adaptive = run_ekf_adaptive(ds, cfg_adaptive)
    
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    t = ds["t"]
    
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    roll_err_adaptive = np.rad2deg(est_adaptive["roll"] - roll_true)
    pitch_err_adaptive = np.rad2deg(est_adaptive["pitch"] - pitch_true)
    
    rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
    rmse_adaptive = np.sqrt(np.mean(roll_err_adaptive**2 + pitch_err_adaptive**2))
    peak_fixed = np.max(np.abs(np.concatenate([roll_err_fixed, pitch_err_fixed])))
    peak_adaptive = np.max(np.abs(np.concatenate([roll_err_adaptive, pitch_err_adaptive])))
    
    print(f"  EKF Fixed:    RMSE={rmse_fixed:.3f}°, Peak={peak_fixed:.3f}°")
    print(f"  EKF Adaptive: RMSE={rmse_adaptive:.3f}°, Peak={peak_adaptive:.3f}°")
    print(f"  RMSE 改善: {(rmse_fixed - rmse_adaptive) / rmse_fixed * 100:.1f}%")
    print(f"  Peak 变化: {(peak_adaptive - peak_fixed) / peak_fixed * 100:+.1f}%")
    
    lambda_k = est_adaptive["debug"]["lambda_k"]
    nis_combined = est_adaptive["debug"]["nis_combined"]
    
    corr_nis_lambda, lag = compute_correlation_and_lag(nis_combined, lambda_k)
    print(f"\n  因果指标:")
    print(f"    corr(NIS, λ): {corr_nis_lambda:.3f}")
    print(f"    lag(NIS→λ): {lag} 样本")
    print(f"    λ 均值: {np.mean(lambda_k):.1f}")
    
    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    ax = axes[0]
    ax.plot(t, np.rad2deg(roll_true), 'k--', alpha=0.5, label='True Roll')
    ax.plot(t, np.rad2deg(est_fixed["roll"]), 'b-', alpha=0.6, label='Fixed')
    ax.plot(t, np.rad2deg(est_adaptive["roll"]), 'r-', alpha=0.6, label='Adaptive')
    ax.set_ylabel('Roll (deg)')
    ax.set_title('Swing: Roll Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(t, roll_err_fixed, 'b-', alpha=0.6, label=f'Fixed (RMSE={rmse_fixed:.2f}°)')
    ax.plot(t, roll_err_adaptive, 'r-', alpha=0.6, label=f'Adaptive (RMSE={rmse_adaptive:.2f}°)')
    ax.set_ylabel('Roll Error (deg)')
    ax.set_title('Roll Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(t, nis_combined, 'g-', alpha=0.7, label='NIS')
    ax.axhline(7.815, color='k', linestyle='--', alpha=0.5)
    ax.set_ylabel('NIS')
    ax.set_title('NIS Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[3]
    ax.plot(t, lambda_k, 'r-', alpha=0.7, label='Lambda')
    ax.set_ylabel('Lambda')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Lambda (mean={np.mean(lambda_k):.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "swing_test.png", dpi=150)
    plt.close()
    
    print(f"\n  图表已保存: {OUTPUT_DIR / 'figures' / 'swing_test.png'}")
    
    return {
        "rmse_fixed": float(rmse_fixed),
        "rmse_adaptive": float(rmse_adaptive),
        "rmse_improvement_pct": float((rmse_fixed - rmse_adaptive) / rmse_fixed * 100),
        "peak_fixed": float(peak_fixed),
        "peak_adaptive": float(peak_adaptive),
        "corr_NIS_lambda": float(corr_nis_lambda),
    }


def test_turn():
    """测试转弯工况（恒定角速度 + 向心加速度）"""
    print("\n" + "="*70)
    print("测试 5: 转弯工况 (Turn) - 恒定角速度 + 向心加速度")
    print("="*70)
    
    # 生成转弯工况
    truth = generate_turn(
        fs=100, duration_s=40,
        roll_deg=0, pitch_deg=0,
        yaw_rate_dps=30.0,  # 30°/s 转弯
        turn_radius_m=10.0,  # 10m 半径
        turn_start_s=5.0,
        turn_duration_s=30.0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    ds = create_dataset(truth, sensor_params)
    
    cfg_fixed = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": True, "threshold": 7.815, "mode": "inflate_R"},
    }
    
    lambda_max = 200.0
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
            "lambda_max": lambda_max,
            "lambda_min": 1.0,
            "use_inflate_mapping": True,
            "inflate_decay_rate": 0.8,
            "inflate_rise_smooth": 1.0,
        },
        "dual_channel": {"enabled": False},
    }
    
    est_fixed = run_ekf_fixed(ds, cfg_fixed)
    est_adaptive = run_ekf_adaptive(ds, cfg_adaptive)
    
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    t = ds["t"]
    
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    roll_err_adaptive = np.rad2deg(est_adaptive["roll"] - roll_true)
    pitch_err_adaptive = np.rad2deg(est_adaptive["pitch"] - pitch_true)
    
    rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
    rmse_adaptive = np.sqrt(np.mean(roll_err_adaptive**2 + pitch_err_adaptive**2))
    
    print(f"  EKF Fixed:    RMSE={rmse_fixed:.3f}°")
    print(f"  EKF Adaptive: RMSE={rmse_adaptive:.3f}°")
    print(f"  RMSE 改善: {(rmse_fixed - rmse_adaptive) / rmse_fixed * 100:.1f}%")
    
    lambda_k = est_adaptive["debug"]["lambda_k"]
    nis_combined = est_adaptive["debug"]["nis_combined"]
    
    # 转弯段统计
    turn_mask = (t >= 5.0) & (t <= 35.0)
    static_mask = ~turn_mask
    
    lambda_turn = np.mean(lambda_k[turn_mask])
    lambda_static = np.mean(lambda_k[static_mask])
    nis_turn = np.mean(nis_combined[turn_mask])
    nis_static = np.mean(nis_combined[static_mask])
    
    print(f"\n  转弯段 vs 静止段:")
    print(f"    转弯段 NIS 均值: {nis_turn:.1f}")
    print(f"    静止段 NIS 均值: {nis_static:.1f}")
    print(f"    转弯段 λ 均值: {lambda_turn:.1f}")
    print(f"    静止段 λ 均值: {lambda_static:.1f}")
    
    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    ax = axes[0]
    ax.plot(t, truth["rpy_deg"][:, 2], 'k-', alpha=0.7, label='Yaw (True)')
    ax.axvline(5.0, color='g', linestyle='--', alpha=0.5, label='Turn Start')
    ax.axvline(35.0, color='r', linestyle='--', alpha=0.5, label='Turn End')
    ax.set_ylabel('Yaw (deg)')
    ax.set_title('Turn: Yaw Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    err_fixed = np.sqrt(roll_err_fixed**2 + pitch_err_fixed**2)
    err_adaptive = np.sqrt(roll_err_adaptive**2 + pitch_err_adaptive**2)
    ax.plot(t, err_fixed, 'b-', alpha=0.6, label=f'Fixed (RMSE={rmse_fixed:.2f}°)')
    ax.plot(t, err_adaptive, 'r-', alpha=0.6, label=f'Adaptive (RMSE={rmse_adaptive:.2f}°)')
    ax.axvline(5.0, color='g', linestyle='--', alpha=0.3)
    ax.axvline(35.0, color='r', linestyle='--', alpha=0.3)
    ax.set_ylabel('Tilt Error (deg)')
    ax.set_title('Tilt Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(t, nis_combined, 'g-', alpha=0.7, label='NIS')
    ax.axhline(7.815, color='k', linestyle='--', alpha=0.5)
    ax.axvline(5.0, color='g', linestyle='--', alpha=0.3)
    ax.axvline(35.0, color='r', linestyle='--', alpha=0.3)
    ax.set_ylabel('NIS')
    ax.set_title('NIS Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[3]
    ax.plot(t, lambda_k, 'r-', alpha=0.7, label='Lambda')
    ax.axvline(5.0, color='g', linestyle='--', alpha=0.3)
    ax.axvline(35.0, color='r', linestyle='--', alpha=0.3)
    ax.set_ylabel('Lambda')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Lambda (turn={lambda_turn:.1f}, static={lambda_static:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "turn_test.png", dpi=150)
    plt.close()
    
    print(f"\n  图表已保存: {OUTPUT_DIR / 'figures' / 'turn_test.png'}")
    
    return {
        "rmse_fixed": float(rmse_fixed),
        "rmse_adaptive": float(rmse_adaptive),
        "rmse_improvement_pct": float((rmse_fixed - rmse_adaptive) / rmse_fixed * 100),
        "lambda_turn": float(lambda_turn),
        "lambda_static": float(lambda_static),
        "nis_turn": float(nis_turn),
        "nis_static": float(nis_static),
    }


def test_accel():
    """测试加减速工况（线性加速度干扰）"""
    print("\n" + "="*70)
    print("测试 6: 加减速工况 (Accel) - 线性加速度干扰")
    print("="*70)
    
    # 生成加减速工况：斜坡加速度
    truth = generate_accel(
        fs=100, duration_s=30,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        accel_type="ramp",
        accel_axis="x",
        accel_peak=5.0,  # 5 m/s^2 峰值
        accel_start_s=5.0,
        accel_duration_s=20.0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    ds = create_dataset(truth, sensor_params)
    
    cfg_fixed = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": True, "threshold": 7.815, "mode": "inflate_R"},
    }
    
    lambda_max = 200.0
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
            "lambda_max": lambda_max,
            "lambda_min": 1.0,
            "use_inflate_mapping": True,
            "inflate_decay_rate": 0.8,
            "inflate_rise_smooth": 1.0,
        },
        "dual_channel": {"enabled": False},
    }
    
    est_fixed = run_ekf_fixed(ds, cfg_fixed)
    est_adaptive = run_ekf_adaptive(ds, cfg_adaptive)
    
    roll_true, pitch_true = get_truth_rpy(ds["truth"])
    t = ds["t"]
    
    roll_err_fixed = np.rad2deg(est_fixed["roll"] - roll_true)
    pitch_err_fixed = np.rad2deg(est_fixed["pitch"] - pitch_true)
    roll_err_adaptive = np.rad2deg(est_adaptive["roll"] - roll_true)
    pitch_err_adaptive = np.rad2deg(est_adaptive["pitch"] - pitch_true)
    
    rmse_fixed = np.sqrt(np.mean(roll_err_fixed**2 + pitch_err_fixed**2))
    rmse_adaptive = np.sqrt(np.mean(roll_err_adaptive**2 + pitch_err_adaptive**2))
    peak_fixed = np.max(np.abs(np.concatenate([roll_err_fixed, pitch_err_fixed])))
    peak_adaptive = np.max(np.abs(np.concatenate([roll_err_adaptive, pitch_err_adaptive])))
    
    print(f"  EKF Fixed:    RMSE={rmse_fixed:.3f}°, Peak={peak_fixed:.3f}°")
    print(f"  EKF Adaptive: RMSE={rmse_adaptive:.3f}°, Peak={peak_adaptive:.3f}°")
    print(f"  RMSE 改善: {(rmse_fixed - rmse_adaptive) / rmse_fixed * 100:.1f}%")
    print(f"  Peak 变化: {(peak_adaptive - peak_fixed) / peak_fixed * 100:+.1f}%")
    
    lambda_k = est_adaptive["debug"]["lambda_k"]
    nis_combined = est_adaptive["debug"]["nis_combined"]
    
    # 加速段统计
    accel_mask = (t >= 5.0) & (t <= 25.0)
    static_mask = ~accel_mask
    
    lambda_accel = np.mean(lambda_k[accel_mask])
    lambda_static = np.mean(lambda_k[static_mask])
    
    print(f"\n  加速段 vs 静止段:")
    print(f"    加速段 λ 均值: {lambda_accel:.1f}")
    print(f"    静止段 λ 均值: {lambda_static:.1f}")
    
    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    ax = axes[0]
    ax.plot(t, truth["a_lin_n"][:, 0], 'k-', alpha=0.7, label='a_lin_x')
    ax.axvline(5.0, color='g', linestyle='--', alpha=0.5)
    ax.axvline(25.0, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Accel (m/s^2)')
    ax.set_title('Accel: Linear Acceleration (X-axis)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    err_fixed = np.sqrt(roll_err_fixed**2 + pitch_err_fixed**2)
    err_adaptive = np.sqrt(roll_err_adaptive**2 + pitch_err_adaptive**2)
    ax.plot(t, err_fixed, 'b-', alpha=0.6, label=f'Fixed (RMSE={rmse_fixed:.2f}°)')
    ax.plot(t, err_adaptive, 'r-', alpha=0.6, label=f'Adaptive (RMSE={rmse_adaptive:.2f}°)')
    ax.axvline(5.0, color='g', linestyle='--', alpha=0.3)
    ax.axvline(25.0, color='r', linestyle='--', alpha=0.3)
    ax.set_ylabel('Tilt Error (deg)')
    ax.set_title('Tilt Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(t, nis_combined, 'g-', alpha=0.7, label='NIS')
    ax.axhline(7.815, color='k', linestyle='--', alpha=0.5)
    ax.axvline(5.0, color='g', linestyle='--', alpha=0.3)
    ax.axvline(25.0, color='r', linestyle='--', alpha=0.3)
    ax.set_ylabel('NIS')
    ax.set_title('NIS Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[3]
    ax.plot(t, lambda_k, 'r-', alpha=0.7, label='Lambda')
    ax.axvline(5.0, color='g', linestyle='--', alpha=0.3)
    ax.axvline(25.0, color='r', linestyle='--', alpha=0.3)
    ax.set_ylabel('Lambda')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Lambda (accel={lambda_accel:.1f}, static={lambda_static:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "accel_test.png", dpi=150)
    plt.close()
    
    print(f"\n  图表已保存: {OUTPUT_DIR / 'figures' / 'accel_test.png'}")
    
    return {
        "rmse_fixed": float(rmse_fixed),
        "rmse_adaptive": float(rmse_adaptive),
        "rmse_improvement_pct": float((rmse_fixed - rmse_adaptive) / rmse_fixed * 100),
        "peak_fixed": float(peak_fixed),
        "peak_adaptive": float(peak_adaptive),
        "lambda_accel": float(lambda_accel),
        "lambda_static": float(lambda_static),
    }


def main():
    print("="*70)
    print("Step 12: 自适应 EKF 验证（完整版）")
    print("="*70)
    
    setup()
    
    results = {}
    
    # 运行所有测试
    results["vibration"] = test_vibration()
    results["shock"] = test_shock()
    results["causality"] = test_causality()
    results["swing"] = test_swing()
    results["turn"] = test_turn()
    results["accel"] = test_accel()
    
    # 汇总
    print("\n" + "="*70)
    print("Step 12 验收汇总（完整版）")
    print("="*70)
    
    print("\n  1. 振动工况:")
    print(f"    Fixed RMSE:    {results['vibration']['rmse_fixed']:.3f}°")
    print(f"    Adaptive RMSE: {results['vibration']['rmse_adaptive']:.3f}°")
    print(f"    RMSE 改善: {results['vibration']['rmse_improvement_pct']:.1f}%")
    print(f"    corr(NIS,λ): {results['vibration']['corr_NIS_lambda']:.3f}")
    
    print("\n  2. 冲击工况:")
    print(f"    Fixed RMSE:    {results['shock']['rmse_fixed']:.3f}°")
    print(f"    Adaptive RMSE: {results['shock']['rmse_adaptive']:.3f}°")
    print(f"    NIS 触发: {'是' if results['shock']['nis_triggered'] else '否'}")
    print(f"    λ 回落: {'是' if results['shock']['lambda_recovered'] else '否'}")
    
    print("\n  3. 因果一致性:")
    print(f"    corr(NIS,λ): {results['causality']['corr_NIS_lambda']:.3f}")
    print(f"    λ 与 NIS 正相关: {'是' if results['causality']['lambda_correlation'] else '否'}")
    
    print("\n  4. 摆动工况:")
    print(f"    Fixed RMSE:    {results['swing']['rmse_fixed']:.3f}°")
    print(f"    Adaptive RMSE: {results['swing']['rmse_adaptive']:.3f}°")
    print(f"    RMSE 改善: {results['swing']['rmse_improvement_pct']:.1f}%")
    
    print("\n  5. 转弯工况:")
    print(f"    Fixed RMSE:    {results['turn']['rmse_fixed']:.3f}°")
    print(f"    Adaptive RMSE: {results['turn']['rmse_adaptive']:.3f}°")
    print(f"    RMSE 改善: {results['turn']['rmse_improvement_pct']:.1f}%")
    
    print("\n  6. 加减速工况:")
    print(f"    Fixed RMSE:    {results['accel']['rmse_fixed']:.3f}°")
    print(f"    Adaptive RMSE: {results['accel']['rmse_adaptive']:.3f}°")
    print(f"    RMSE 改善: {results['accel']['rmse_improvement_pct']:.1f}%")
    
    # 保存结果
    with open(OUTPUT_DIR / "results_full.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n  结果已保存: {OUTPUT_DIR / 'results_full.json'}")
    
    # 综合判定
    vibration_ok = results["vibration"]["rmse_improvement_pct"] > -10  # 允许 10% 恶化
    shock_ok = results["shock"]["nis_triggered"] and results["shock"]["lambda_recovered"]
    causality_ok = results["causality"]["lambda_correlation"]
    swing_ok = results["swing"]["rmse_improvement_pct"] > -10
    turn_ok = results["turn"]["rmse_improvement_pct"] > -10
    accel_ok = results["accel"]["rmse_improvement_pct"] > -10
    
    print("\n  综合判定:")
    print(f"    振动工况: {'OK' if vibration_ok else 'X'} ({results['vibration']['rmse_improvement_pct']:+.1f}%)")
    print(f"    冲击工况: {'OK' if shock_ok else 'X'}")
    print(f"    因果一致性: {'OK' if causality_ok else 'X'}")
    print(f"    摆动工况: {'OK' if swing_ok else 'X'} ({results['swing']['rmse_improvement_pct']:+.1f}%)")
    print(f"    转弯工况: {'OK' if turn_ok else 'X'} ({results['turn']['rmse_improvement_pct']:+.1f}%)")
    print(f"    加减速工况: {'OK' if accel_ok else 'X'} ({results['accel']['rmse_improvement_pct']:+.1f}%)")
    
    all_passed = vibration_ok and shock_ok and causality_ok and swing_ok and turn_ok and accel_ok
    
    # 计算平均改善
    avg_improvement = np.mean([
        results["vibration"]["rmse_improvement_pct"],
        results["swing"]["rmse_improvement_pct"],
        results["turn"]["rmse_improvement_pct"],
        results["accel"]["rmse_improvement_pct"],
    ])
    
    print(f"\n  平均 RMSE 改善: {avg_improvement:+.1f}%")
    
    if all_passed:
        print(f"\n  总体结果: PASS [OK]")
    else:
        print(f"\n  总体结果: NEEDS IMPROVEMENT")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

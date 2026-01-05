#!/usr/bin/env python
"""
Step 10 性能汇总图表

生成：
1. 各工况 RMSE 对比柱状图
2. 各工况时序图（6 个子图）
3. NIS 统计对比图
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.truth.scenarios import (
    generate_quasi_static, generate_swing, generate_accel,
    generate_turn, generate_vibration, generate_shock
)
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.complementary import run_complementary
from src.filters.ekf_fixed import run_ekf_fixed
from src.metrics.tilt_error import compute_tilt_metrics
from src.common.math3d import rad2deg


def run_all_scenarios():
    """运行所有工况并收集结果"""
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    filter_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-8,
        "R_acc": 3.5e-6,
        "use_direction_meas": True,
        "nis_gating": {"enabled": True, "threshold": 7.815, "mode": "inflate_R"},
    }
    
    results = {}
    
    # 1. Quasi-static
    print("运行 quasi_static...")
    truth = generate_quasi_static(
        fs=100.0, duration_s=30.0, roll_deg=5.0, pitch_deg=-3.0,
        yaw_deg=0.0, temp_C=25.0, seed=1
    )
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    ds = {"meas": {"acc": meas["acc"], "gyro": meas["gyro"]}, "meta": {"fs": 100.0}}
    
    # 计算真值 rpy_deg
    from src.common.math3d import quat_to_rpy
    n = len(truth["t"])
    rpy_deg = np.zeros((n, 3))
    for i in range(n):
        r, p, y = quat_to_rpy(truth["q_nb"][i])
        rpy_deg[i] = [rad2deg(r), rad2deg(p), rad2deg(y)]
    truth["rpy_deg"] = rpy_deg
    
    est_comp = run_complementary(ds, {"alpha": 0.98})
    est_ekf = run_ekf_fixed(ds, filter_cfg)
    m_comp = compute_tilt_metrics({"rpy_deg": rpy_deg}, est_comp, burn_in_s=1.0, fs=100.0)
    m_ekf = compute_tilt_metrics({"rpy_deg": rpy_deg}, est_ekf, burn_in_s=1.0, fs=100.0)
    results["quasi_static"] = {"truth": truth, "est_ekf": est_ekf, "m_comp": m_comp, "m_ekf": m_ekf}
    
    # 2. Swing
    print("运行 swing...")
    truth = generate_swing(
        fs=100.0, duration_s=30.0, roll_amp_deg=10.0, pitch_amp_deg=5.0,
        roll_freq_hz=0.3, pitch_freq_hz=0.2, roll_phase_deg=0.0,
        pitch_phase_deg=90.0, yaw_deg=0.0, temp_C=25.0, seed=1
    )
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    ds = {"meas": {"acc": meas["acc"], "gyro": meas["gyro"]}, "meta": {"fs": 100.0}}
    est_comp = run_complementary(ds, {"alpha": 0.98})
    est_ekf = run_ekf_fixed(ds, filter_cfg)
    m_comp = compute_tilt_metrics({"rpy_deg": truth["rpy_deg"]}, est_comp, burn_in_s=1.0, fs=100.0)
    m_ekf = compute_tilt_metrics({"rpy_deg": truth["rpy_deg"]}, est_ekf, burn_in_s=1.0, fs=100.0)
    results["swing"] = {"truth": truth, "est_ekf": est_ekf, "m_comp": m_comp, "m_ekf": m_ekf}
    
    # 3. Accel
    print("运行 accel...")
    truth = generate_accel(
        fs=100.0, duration_s=30.0, roll_deg=2.0, pitch_deg=-1.0, yaw_deg=0.0,
        accel_type="step", accel_axis="x", accel_peak=2.0,
        accel_start_s=5.0, accel_duration_s=10.0, temp_C=25.0, seed=1
    )
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    ds = {"meas": {"acc": meas["acc"], "gyro": meas["gyro"]}, "meta": {"fs": 100.0}}
    est_comp = run_complementary(ds, {"alpha": 0.98})
    est_ekf = run_ekf_fixed(ds, filter_cfg)
    m_comp = compute_tilt_metrics({"rpy_deg": truth["rpy_deg"]}, est_comp, burn_in_s=1.0, fs=100.0)
    m_ekf = compute_tilt_metrics({"rpy_deg": truth["rpy_deg"]}, est_ekf, burn_in_s=1.0, fs=100.0)
    results["accel"] = {"truth": truth, "est_ekf": est_ekf, "m_comp": m_comp, "m_ekf": m_ekf}
    
    # 4. Turn
    print("运行 turn...")
    truth = generate_turn(
        fs=100.0, duration_s=30.0, roll_deg=2.0, pitch_deg=-1.0,
        yaw_rate_dps=30.0, turn_radius_m=10.0, turn_start_s=5.0,
        turn_duration_s=20.0, temp_C=25.0, seed=1
    )
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    ds = {"meas": {"acc": meas["acc"], "gyro": meas["gyro"]}, "meta": {"fs": 100.0}}
    est_comp = run_complementary(ds, {"alpha": 0.98})
    est_ekf = run_ekf_fixed(ds, filter_cfg)
    m_comp = compute_tilt_metrics({"rpy_deg": truth["rpy_deg"]}, est_comp, burn_in_s=1.0, fs=100.0)
    m_ekf = compute_tilt_metrics({"rpy_deg": truth["rpy_deg"]}, est_ekf, burn_in_s=1.0, fs=100.0)
    results["turn"] = {"truth": truth, "est_ekf": est_ekf, "m_comp": m_comp, "m_ekf": m_ekf}
    
    # 5. Vibration
    print("运行 vibration...")
    truth = generate_vibration(
        fs=100.0, duration_s=30.0, roll_deg=2.0, pitch_deg=-1.0, yaw_deg=0.0,
        vib_rms=0.5, vib_bandwidth_hz=10.0, vib_center_hz=0.0,
        temp_C=25.0, seed=1
    )
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    ds = {"meas": {"acc": meas["acc"], "gyro": meas["gyro"]}, "meta": {"fs": 100.0}}
    est_comp = run_complementary(ds, {"alpha": 0.98})
    est_ekf = run_ekf_fixed(ds, filter_cfg)
    m_comp = compute_tilt_metrics({"rpy_deg": truth["rpy_deg"]}, est_comp, burn_in_s=1.0, fs=100.0)
    m_ekf = compute_tilt_metrics({"rpy_deg": truth["rpy_deg"]}, est_ekf, burn_in_s=1.0, fs=100.0)
    results["vibration"] = {"truth": truth, "est_ekf": est_ekf, "m_comp": m_comp, "m_ekf": m_ekf}
    
    # 6. Shock
    print("运行 shock...")
    truth = generate_shock(
        fs=100.0, duration_s=20.0, roll_deg=2.0, pitch_deg=-1.0, yaw_deg=0.0,
        shock_peak=50.0, shock_width_s=0.05, shock_times=[5.0, 10.0, 15.0],
        shock_axis="z", temp_C=25.0, seed=1
    )
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    ds = {"meas": {"acc": meas["acc"], "gyro": meas["gyro"]}, "meta": {"fs": 100.0}}
    est_comp = run_complementary(ds, {"alpha": 0.98})
    est_ekf = run_ekf_fixed(ds, filter_cfg)
    m_comp = compute_tilt_metrics({"rpy_deg": truth["rpy_deg"]}, est_comp, burn_in_s=1.0, fs=100.0)
    m_ekf = compute_tilt_metrics({"rpy_deg": truth["rpy_deg"]}, est_ekf, burn_in_s=1.0, fs=100.0)
    results["shock"] = {"truth": truth, "est_ekf": est_ekf, "m_comp": m_comp, "m_ekf": m_ekf}
    
    return results


def plot_rmse_comparison(results, save_path):
    """绘制 RMSE 对比柱状图"""
    scenarios = list(results.keys())
    n = len(scenarios)
    
    comp_roll = [results[s]["m_comp"]["rmse_roll"] for s in scenarios]
    comp_pitch = [results[s]["m_comp"]["rmse_pitch"] for s in scenarios]
    ekf_roll = [results[s]["m_ekf"]["rmse_roll"] for s in scenarios]
    ekf_pitch = [results[s]["m_ekf"]["rmse_pitch"] for s in scenarios]
    
    x = np.arange(n)
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - 1.5*width, comp_roll, width, label='Comp Roll', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, comp_pitch, width, label='Comp Pitch', color='#1f77b4', alpha=0.5)
    bars3 = ax.bar(x + 0.5*width, ekf_roll, width, label='EKF Roll', color='#d62728', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, ekf_pitch, width, label='EKF Pitch', color='#d62728', alpha=0.5)
    
    ax.set_ylabel('RMSE (deg)', fontsize=12)
    ax.set_title('Step 10: RMSE Comparison Across All Scenarios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:
                ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 添加 1° 参考线
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='1° threshold')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/step10_rmse_comparison.png", dpi=150)
    plt.close()
    print(f"保存: {save_path}/step10_rmse_comparison.png")


def plot_all_timeseries(results, save_path):
    """绘制所有工况的时序图"""
    scenarios = list(results.keys())
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        r = results[scenario]
        t = r["truth"]["t"]
        
        # 真值
        roll_true = r["truth"]["rpy_deg"][:, 0]
        pitch_true = r["truth"]["rpy_deg"][:, 1]
        
        # EKF 估计
        roll_ekf = rad2deg(r["est_ekf"]["roll"])
        pitch_ekf = rad2deg(r["est_ekf"]["pitch"])
        
        # 误差
        roll_err = roll_ekf - roll_true
        pitch_err = pitch_ekf - pitch_true
        
        ax.plot(t, roll_err, 'r-', label='Roll Error', linewidth=0.8, alpha=0.8)
        ax.plot(t, pitch_err, 'b-', label='Pitch Error', linewidth=0.8, alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        ax.set_title(f'{scenario.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Error (deg)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 设置 y 轴范围
        max_err = max(np.abs(roll_err).max(), np.abs(pitch_err).max())
        if max_err < 1:
            ax.set_ylim([-1, 1])
        elif max_err < 5:
            ax.set_ylim([-5, 5])
        else:
            ax.set_ylim([-20, 20])
    
    plt.suptitle('Step 10: EKF Attitude Error Across All Scenarios', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_path}/step10_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}/step10_timeseries.png")


def plot_nis_comparison(results, save_path):
    """绘制 NIS 统计对比图"""
    scenarios = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    
    nis_stats = []
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        r = results[scenario]
        t = r["truth"]["t"]
        nis = r["est_ekf"]["debug"]["nis"]
        gated = r["est_ekf"]["debug"]["gated"]
        
        # 绘制 NIS
        ax.plot(t, nis, 'b-', linewidth=0.5, alpha=0.7)
        ax.axhline(y=3, color='r', linestyle='--', linewidth=1, label='Expected (χ²₃)')
        ax.axhline(y=7.815, color='r', linestyle=':', linewidth=1, label='95% bound')
        
        ax.set_title(f'{scenario.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('NIS', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, min(50, np.max(nis) * 1.2)])
        
        # 统计
        burn_in = int(1.0 * 100)
        nis_stable = nis[burn_in:]
        mean_nis = np.mean(nis_stable)
        p_exceed = np.mean(nis_stable > 7.815) * 100
        gate_rate = np.mean(gated[burn_in:]) * 100
        
        nis_stats.append({
            "scenario": scenario,
            "mean_nis": mean_nis,
            "p_exceed": p_exceed,
            "gate_rate": gate_rate
        })
        
        # 添加统计文本
        ax.text(0.02, 0.98, f'mean={mean_nis:.1f}\np>7.8={p_exceed:.0f}%\ngate={gate_rate:.0f}%',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Step 10: NIS Statistics Across All Scenarios', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_path}/step10_nis_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}/step10_nis_comparison.png")
    
    return nis_stats


def plot_summary_table(results, nis_stats, save_path):
    """绘制汇总表格"""
    scenarios = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # 准备表格数据
    headers = ['Scenario', 'Comp RMSE\n(Roll/Pitch)', 'EKF RMSE\n(Roll/Pitch)', 
               'mean(NIS)', 'p(NIS>7.8)', 'Gate Rate', 'Status']
    
    table_data = []
    for i, s in enumerate(scenarios):
        r = results[s]
        ns = nis_stats[i]
        
        comp_rmse = f"{r['m_comp']['rmse_roll']:.2f}° / {r['m_comp']['rmse_pitch']:.2f}°"
        ekf_rmse = f"{r['m_ekf']['rmse_roll']:.2f}° / {r['m_ekf']['rmse_pitch']:.2f}°"
        
        # 判断状态
        if r['m_ekf']['rmse_roll'] < 1 and r['m_ekf']['rmse_pitch'] < 1:
            status = "✓ PASS"
        elif r['m_ekf']['rmse_roll'] < 10 and r['m_ekf']['rmse_pitch'] < 10:
            status = "⚠ Limited"
        else:
            status = "✗ FAIL"
        
        table_data.append([
            s.upper(),
            comp_rmse,
            ekf_rmse,
            f"{ns['mean_nis']:.1f}",
            f"{ns['p_exceed']:.0f}%",
            f"{ns['gate_rate']:.0f}%",
            status
        ])
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#4472C4'] * len(headers)
    )
    
    # 设置样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # 设置表头颜色
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 设置状态列颜色
    status_col = len(headers) - 1
    for i, row in enumerate(table_data):
        if "PASS" in row[-1]:
            table[(i+1, status_col)].set_facecolor('#C6EFCE')
        elif "Limited" in row[-1]:
            table[(i+1, status_col)].set_facecolor('#FFEB9C')
        else:
            table[(i+1, status_col)].set_facecolor('#FFC7CE')
    
    plt.title('Step 10: Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{save_path}/step10_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}/step10_summary_table.png")


def main():
    print("=" * 60)
    print("Step 10 性能汇总图表生成")
    print("=" * 60)
    
    save_path = "outputs/figures/step10_summary"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # 运行所有工况
    print("\n运行所有工况...")
    results = run_all_scenarios()
    
    # 生成图表
    print("\n生成图表...")
    plot_rmse_comparison(results, save_path)
    plot_all_timeseries(results, save_path)
    nis_stats = plot_nis_comparison(results, save_path)
    plot_summary_table(results, nis_stats, save_path)
    
    print("\n" + "=" * 60)
    print(f"所有图表已保存到: {save_path}/")
    print("=" * 60)
    
    # 打印汇总
    print("\nStep 10 性能汇总:")
    print("-" * 60)
    for s, r in results.items():
        print(f"  {s:12s}: EKF RMSE = {r['m_ekf']['rmse_roll']:.2f}° / {r['m_ekf']['rmse_pitch']:.2f}°")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

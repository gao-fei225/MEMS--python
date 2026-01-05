#!/usr/bin/env python3
"""
Step 11 验证脚本：增强 IMU 误差模型

验证内容：
1. 回归测试：关闭所有增强功能时，结果与原版一致
2. 偏置随机游走：开启后偏置有漂移
3. 温漂：温度变化时偏置跟随变化
4. 比例因子/安装偏差：测量值有系统性偏差
5. 饱和/量化：极端值被限幅，小信号有量化台阶
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.truth.scenarios import generate_quasi_static
from src.sensors.imu_model import forward_imu


def test_regression():
    """回归测试：关闭所有增强功能，结果应与原版一致"""
    print("\n" + "="*60)
    print("测试 1: 回归测试（关闭所有增强功能）")
    print("="*60)
    
    # 生成真值
    truth = generate_quasi_static(
        fs=100, duration_s=10,
        roll_deg=5, pitch_deg=-3, yaw_deg=0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    
    # 原版参数（只有 bias0 和 sigma_white）
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
    
    # 使用相同种子
    meas_orig = forward_imu(truth, params_original, seed=123)
    meas_enh = forward_imu(truth, params_enhanced, seed=123)
    
    # 比较结果
    acc_diff = np.max(np.abs(meas_orig["acc"] - meas_enh["acc"]))
    gyro_diff = np.max(np.abs(meas_orig["gyro"] - meas_enh["gyro"]))
    
    print(f"  加速度最大差异: {acc_diff:.2e} m/s²")
    print(f"  陀螺仪最大差异: {gyro_diff:.2e} rad/s")
    
    passed = acc_diff < 1e-10 and gyro_diff < 1e-10
    print(f"  结果: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_bias_random_walk():
    """测试偏置随机游走"""
    print("\n" + "="*60)
    print("测试 2: 偏置随机游走")
    print("="*60)
    
    # 长时间真值
    truth = generate_quasi_static(
        fs=100, duration_s=60,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    
    # 无随机游走
    params_no_rw = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
    }
    
    # 有随机游走
    params_with_rw = {
        "acc": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "bias_rw": {"enabled": True, "sigma_rw": 1e-3},
        },
        "gyro": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "bias_rw": {"enabled": True, "sigma_rw": 1e-4},
        },
    }
    
    meas_no_rw = forward_imu(truth, params_no_rw, seed=123)
    meas_with_rw = forward_imu(truth, params_with_rw, seed=123)
    
    # 检查偏置漂移
    bias_no_rw = meas_no_rw["acc_bias_true"]
    bias_with_rw = meas_with_rw["acc_bias_true"]
    
    drift_no_rw = np.std(bias_no_rw, axis=0)
    drift_with_rw = np.std(bias_with_rw, axis=0)
    
    print(f"  无RW偏置标准差: {drift_no_rw}")
    print(f"  有RW偏置标准差: {drift_with_rw}")
    print(f"  偏置漂移范围: {np.ptp(bias_with_rw, axis=0)}")
    
    passed = np.all(drift_with_rw > drift_no_rw * 10)
    print(f"  结果: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed, truth["t"], bias_with_rw


def test_temperature_drift():
    """测试温漂"""
    print("\n" + "="*60)
    print("测试 3: 温漂")
    print("="*60)
    
    # 生成带温度变化的真值
    fs, duration = 100, 30
    
    truth = generate_quasi_static(
        fs=fs, duration_s=duration,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        temp_C=25, seed=42
    )
    
    # 使用 truth 的实际长度生成温度序列
    t = truth["t"]
    n_samples = len(t)
    
    # 温度从 20°C 升到 40°C
    temp = 20 + 20 * t / duration
    truth["temp"] = temp
    truth["fs"] = fs
    
    # 无温漂
    params_no_td = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
    }
    
    # 有温漂
    params_with_td = {
        "acc": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "temp_drift": {"enabled": True, "k1": [0.01, 0.01, 0.01], "T0": 25.0},
        },
        "gyro": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "temp_drift": {"enabled": True, "k1": [0.001, 0.001, 0.001], "T0": 25.0},
        },
    }
    
    meas_no_td = forward_imu(truth, params_no_td, seed=123)
    meas_with_td = forward_imu(truth, params_with_td, seed=123)
    
    # 检查偏置与温度的相关性
    bias_with_td = meas_with_td["acc_bias_true"]
    
    # 预期：bias = k1 * (T - T0) = 0.01 * (T - 25)
    # T从20到40，所以bias从-0.05到+0.15
    expected_start = 0.01 * (20 - 25)  # -0.05
    expected_end = 0.01 * (40 - 25)    # +0.15
    
    actual_start = bias_with_td[0, 0]
    actual_end = bias_with_td[-1, 0]
    
    print(f"  温度范围: {temp[0]:.1f}°C -> {temp[-1]:.1f}°C")
    print(f"  预期偏置: {expected_start:.4f} -> {expected_end:.4f}")
    print(f"  实际偏置: {actual_start:.4f} -> {actual_end:.4f}")
    
    passed = (abs(actual_start - expected_start) < 1e-6 and 
              abs(actual_end - expected_end) < 1e-6)
    print(f"  结果: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed, t, temp, bias_with_td


def test_scale_misalignment():
    """测试比例因子和安装偏差"""
    print("\n" + "="*60)
    print("测试 4: 比例因子/安装偏差")
    print("="*60)
    
    truth = generate_quasi_static(
        fs=100, duration_s=10,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    
    # 无比例因子误差
    params_no_sm = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
    }
    
    # 有比例因子误差 (1000 ppm = 0.1%)
    params_with_sm = {
        "acc": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "scale_misalign": {
                "enabled": True,
                "scale_error": [0.001, 0.001, 0.001],  # 0.1%
                "misalignment": [0.0, 0.0, 0.0],
            },
        },
        "gyro": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
        },
    }
    
    meas_no_sm = forward_imu(truth, params_no_sm, seed=123)
    meas_with_sm = forward_imu(truth, params_with_sm, seed=123)
    
    # 静止时 acc_z ≈ -g
    acc_z_no_sm = np.mean(meas_no_sm["acc"][:, 2])
    acc_z_with_sm = np.mean(meas_with_sm["acc"][:, 2])
    
    # 预期：acc_z_with_sm = acc_z_no_sm * (1 + 0.001)
    expected_ratio = 1.001
    actual_ratio = acc_z_with_sm / acc_z_no_sm
    
    print(f"  无比例因子误差 acc_z: {acc_z_no_sm:.6f} m/s²")
    print(f"  有比例因子误差 acc_z: {acc_z_with_sm:.6f} m/s²")
    print(f"  预期比值: {expected_ratio:.6f}")
    print(f"  实际比值: {actual_ratio:.6f}")
    
    passed = abs(actual_ratio - expected_ratio) < 1e-6
    print(f"  结果: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_saturation():
    """测试饱和"""
    print("\n" + "="*60)
    print("测试 5: 饱和")
    print("="*60)
    
    # 生成大加速度场景
    from src.truth.scenarios import generate_shock
    truth = generate_shock(
        fs=100, duration_s=5,
        roll_deg=0, pitch_deg=0, yaw_deg=0,
        temp_C=25, seed=42,
        shock_peak=100.0,  # 100 m/s² > 饱和阈值
        shock_width_s=0.05,
        shock_times=[1.0, 2.0, 3.0]
    )
    truth["fs"] = 100.0
    
    # 无饱和
    params_no_sat = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
    }
    
    # 有饱和 (±50 m/s²)
    params_with_sat = {
        "acc": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
            "saturation": {"enabled": True, "range": 50.0},
        },
        "gyro": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
        },
    }
    
    meas_no_sat = forward_imu(truth, params_no_sat, seed=123)
    meas_with_sat = forward_imu(truth, params_with_sat, seed=123)
    
    max_no_sat = np.max(np.abs(meas_no_sat["acc"]))
    max_with_sat = np.max(np.abs(meas_with_sat["acc"]))
    
    print(f"  无饱和最大加速度: {max_no_sat:.2f} m/s²")
    print(f"  有饱和最大加速度: {max_with_sat:.2f} m/s²")
    print(f"  饱和阈值: 50.0 m/s²")
    
    passed = max_with_sat <= 50.0 and max_no_sat > 50.0
    print(f"  结果: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_quantization():
    """测试量化"""
    print("\n" + "="*60)
    print("测试 6: 量化")
    print("="*60)
    
    truth = generate_quasi_static(
        fs=100, duration_s=10,
        roll_deg=5, pitch_deg=-3, yaw_deg=0,
        temp_C=25, seed=42
    )
    truth["fs"] = 100.0
    
    # 无量化
    params_no_quant = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.001},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
    }
    
    # 有量化 (8位，范围±20 m/s²，步长 = 40/256 ≈ 0.156 m/s²)
    params_with_quant = {
        "acc": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.001,
            "quantization": {"enabled": True, "bits": 8, "range": 20.0},
        },
        "gyro": {
            "bias0": [0.0, 0.0, 0.0],
            "sigma_white": 0.0,
        },
    }
    
    meas_no_quant = forward_imu(truth, params_no_quant, seed=123)
    meas_with_quant = forward_imu(truth, params_with_quant, seed=123)
    
    # 检查量化步长
    acc_diff = np.diff(meas_with_quant["acc"][:, 0])
    unique_diffs = np.unique(np.round(acc_diff, 6))
    
    expected_step = 2 * 20.0 / 256  # ≈ 0.156
    
    print(f"  预期量化步长: {expected_step:.6f} m/s²")
    print(f"  实际唯一差值数: {len(unique_diffs)}")
    print(f"  差值样本: {unique_diffs[:5]}")
    
    # 量化后的值应该是步长的整数倍
    acc_quant = meas_with_quant["acc"][:, 0]
    residuals = np.abs(acc_quant / expected_step - np.round(acc_quant / expected_step))
    max_residual = np.max(residuals)
    
    print(f"  量化残差最大值: {max_residual:.2e}")
    
    passed = max_residual < 1e-10
    print(f"  结果: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def plot_results(results, output_dir):
    """绘制验证结果图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 偏置随机游走
    if results.get("bias_rw"):
        t, bias = results["bias_rw"]
        ax = axes[0, 0]
        ax.plot(t, bias[:, 0], label='bx')
        ax.plot(t, bias[:, 1], label='by')
        ax.plot(t, bias[:, 2], label='bz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Bias (m/s^2)')
        ax.set_title('Bias Random Walk')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 温漂
    if results.get("temp_drift"):
        t, temp, bias = results["temp_drift"]
        ax = axes[0, 1]
        ax2 = ax.twinx()
        ax.plot(t, temp, 'b-', label='Temperature')
        ax2.plot(t, bias[:, 0], 'r-', label='Bias')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (C)', color='b')
        ax2.set_ylabel('Bias (m/s^2)', color='r')
        ax.set_title('Temperature Drift')
        ax.grid(True, alpha=0.3)
    
    # 汇总表
    ax = axes[1, 0]
    ax.axis('off')
    
    test_names = [
        "1. Regression Test",
        "2. Bias Random Walk",
        "3. Temperature Drift",
        "4. Scale/Misalignment",
        "5. Saturation",
        "6. Quantization",
    ]
    
    statuses = [
        "PASS" if results.get("regression", False) else "FAIL",
        "PASS" if results.get("bias_rw_pass", False) else "FAIL",
        "PASS" if results.get("temp_drift_pass", False) else "FAIL",
        "PASS" if results.get("scale_misalign", False) else "FAIL",
        "PASS" if results.get("saturation", False) else "FAIL",
        "PASS" if results.get("quantization", False) else "FAIL",
    ]
    
    # 设置颜色
    colors = [['white', 'lightgreen' if s == 'PASS' else 'lightcoral'] for s in statuses]
    
    table_data = [[name, status] for name, status in zip(test_names, statuses)]
    table = ax.table(
        cellText=table_data,
        colLabels=["Test Item", "Status"],
        loc='center',
        cellLoc='left',
        cellColours=colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    ax.set_title('Step 11 Validation Summary', fontsize=14, fontweight='bold')
    
    # 空白
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "step11_imu_enhanced_validation.png", dpi=150)
    plt.close()
    print(f"\n图表已保存: {output_dir / 'step11_imu_enhanced_validation.png'}")


def main():
    print("="*60)
    print("Step 11: 增强 IMU 误差模型验证")
    print("="*60)
    
    output_dir = Path("tilt-adapt-ekf-sim/outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 运行所有测试
    results["regression"] = test_regression()
    
    passed, t, bias = test_bias_random_walk()
    results["bias_rw_pass"] = passed
    results["bias_rw"] = (t, bias)
    
    passed, t, temp, bias = test_temperature_drift()
    results["temp_drift_pass"] = passed
    results["temp_drift"] = (t, temp, bias)
    
    results["scale_misalign"] = test_scale_misalignment()
    results["saturation"] = test_saturation()
    results["quantization"] = test_quantization()
    
    # 汇总
    print("\n" + "="*60)
    print("Step 11 验证汇总")
    print("="*60)
    
    all_passed = all([
        results["regression"],
        results["bias_rw_pass"],
        results["temp_drift_pass"],
        results["scale_misalign"],
        results["saturation"],
        results["quantization"],
    ])
    
    print(f"\n总体结果: {'✓ ALL PASS' if all_passed else '✗ SOME FAILED'}")
    
    # 绘图
    plot_results(results, output_dir)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

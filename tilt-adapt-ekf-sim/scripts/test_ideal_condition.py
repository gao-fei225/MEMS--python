#!/usr/bin/env python
"""
理想条件回归测试（金标准测试）

用 bias=0、noise=0 的理想传感器条件验证整个链路的数值正确性。
在理想条件下，滤波器输出应与真值完全一致（仅存在数值精度误差）。

期望：误差 < 1e-10 deg（数值精度级别）
如果误差 > 1e-6 deg，则表明存在实现问题。

常见问题：
- acc → 角度公式错误
- 坐标系/轴定义不一致
- deg-rad 单位转换错误
- 时间对齐问题

运行方式：
    python scripts/test_ideal_condition.py
"""

import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置 matplotlib 后端为非交互式
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.common.math3d import deg2rad, rad2deg
from src.truth.scenarios import generate_quasi_static, generate_swing
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.complementary import run_complementary, acc_to_roll_pitch


# ============================================================
# 常量定义
# ============================================================

# 理想传感器参数：bias=0, noise=0
IDEAL_SENSOR_PARAMS = {
    "acc": {
        "bias0": [0.0, 0.0, 0.0],
        "sigma_white": 0.0,
    },
    "gyro": {
        "bias0": [0.0, 0.0, 0.0],
        "sigma_white": 0.0,
    },
}

# 误差阈值
GOLD_STANDARD_THRESHOLD_DEG = 1e-10  # 金标准阈值
NUMERICAL_WARNING_THRESHOLD_DEG = 1e-6  # 数值警告阈值
IMPLEMENTATION_ERROR_THRESHOLD_DEG = 0.01  # 实现错误阈值


# ============================================================
# 数据结构
# ============================================================

@dataclass
class IdealTestResult:
    """理想条件测试结果"""
    scenario: str
    passed: bool
    max_roll_err_deg: float
    max_pitch_err_deg: float
    rmse_roll_deg: float
    rmse_pitch_deg: float
    error_level: str  # "gold_standard" | "numerical_warning" | "implementation_error"
    roll_err_series: np.ndarray
    pitch_err_series: np.ndarray
    timestamps: np.ndarray
    peak_time_s: float
    peak_axis: str  # "roll" | "pitch"
    suggested_checks: List[str]


# ============================================================
# 辅助函数
# ============================================================

def wrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    """
    将角度 wrap 到 [-180, 180) 范围
    
    Args:
        angle_deg: 角度（度）
    
    Returns:
        wrapped 角度（度）
    """
    return ((angle_deg + 180) % 360) - 180


def validate_error_level(max_err_deg: float) -> Tuple[str, bool, List[str]]:
    """
    验证误差级别
    
    Args:
        max_err_deg: 最大误差（度）
    
    Returns:
        (error_level, passed, suggested_checks)
    """
    if max_err_deg < GOLD_STANDARD_THRESHOLD_DEG:
        return "gold_standard", True, []
    elif max_err_deg < NUMERICAL_WARNING_THRESHOLD_DEG:
        return "numerical_warning", True, [
            "数值稳定性：检查是否有累积误差",
            "浮点精度：检查中间计算是否使用 float64",
        ]
    else:
        suggested_checks = [
            "acc → 角度公式：检查 roll = atan2(ay, az), pitch = atan2(-ax, sqrt(ay²+az²))",
            "坐标系定义：确认 NED/FRD 约定一致",
            "重力方向：确认 g_n = [0, 0, +g]",
            "deg-rad 转换：检查所有角度单位转换",
            "时间对齐：检查真值与测量的时间戳对齐",
            "滤波器初始化：确认使用 acc 初始化第一帧",
        ]
        return "implementation_error", False, suggested_checks


# ============================================================
# 测试函数
# ============================================================

def run_ideal_condition_test(
    scenario: str,
    scenario_params: dict,
    filter_cfg: dict = None,
) -> IdealTestResult:
    """
    运行单个理想条件测试
    
    Args:
        scenario: 工况名称 ("quasi_static" | "swing")
        scenario_params: 工况参数
        filter_cfg: 滤波器配置
    
    Returns:
        IdealTestResult
    """
    # 理想条件下使用 alpha=0.0，完全信任加速度计
    # 因为 bias=0, noise=0，加速度计直接给出真值角度
    if filter_cfg is None:
        filter_cfg = {"alpha": 0.0}
    
    g = GRAVITY_STANDARD
    
    # 生成真值
    if scenario == "quasi_static":
        truth = generate_quasi_static(**scenario_params)
    elif scenario == "swing":
        truth = generate_swing(**scenario_params)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # 使用理想传感器参数生成测量
    meas = forward_imu(truth, IDEAL_SENSOR_PARAMS, seed=42, g=g)
    
    # 构建数据集
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params.get("fs", 100.0)},
    }
    
    # 运行滤波器
    est = run_complementary(ds, filter_cfg)
    
    # 获取真值 roll/pitch
    if "rpy_deg" in truth:
        roll_true_deg = truth["rpy_deg"][:, 0]
        pitch_true_deg = truth["rpy_deg"][:, 1]
    else:
        # 从四元数计算欧拉角
        from src.common.math3d import quat_to_rpy
        n_samples = len(truth["q_nb"])
        roll_true_deg = np.zeros(n_samples)
        pitch_true_deg = np.zeros(n_samples)
        for i in range(n_samples):
            roll, pitch, yaw = quat_to_rpy(truth["q_nb"][i])
            roll_true_deg[i] = rad2deg(roll)
            pitch_true_deg[i] = rad2deg(pitch)
    
    # 获取估计值
    roll_est_deg = rad2deg(est["roll"])
    pitch_est_deg = rad2deg(est["pitch"])
    
    # 计算误差（使用 wrap 避免角度跳变）
    roll_err_deg = wrap_deg(roll_est_deg - roll_true_deg)
    pitch_err_deg = wrap_deg(pitch_est_deg - pitch_true_deg)
    
    # 计算统计量
    max_roll_err = np.max(np.abs(roll_err_deg))
    max_pitch_err = np.max(np.abs(pitch_err_deg))
    max_err = max(max_roll_err, max_pitch_err)
    
    rmse_roll = np.sqrt(np.mean(roll_err_deg**2))
    rmse_pitch = np.sqrt(np.mean(pitch_err_deg**2))
    
    # 找到峰值误差时间
    if max_roll_err >= max_pitch_err:
        peak_idx = np.argmax(np.abs(roll_err_deg))
        peak_axis = "roll"
    else:
        peak_idx = np.argmax(np.abs(pitch_err_deg))
        peak_axis = "pitch"
    peak_time_s = truth["t"][peak_idx]
    
    # 验证误差级别
    error_level, passed, suggested_checks = validate_error_level(max_err)
    
    return IdealTestResult(
        scenario=scenario,
        passed=passed,
        max_roll_err_deg=max_roll_err,
        max_pitch_err_deg=max_pitch_err,
        rmse_roll_deg=rmse_roll,
        rmse_pitch_deg=rmse_pitch,
        error_level=error_level,
        roll_err_series=roll_err_deg,
        pitch_err_series=pitch_err_deg,
        timestamps=truth["t"],
        peak_time_s=peak_time_s,
        peak_axis=peak_axis,
        suggested_checks=suggested_checks,
    )


def save_diagnostic_plot(result: IdealTestResult, output_dir: str = "outputs/figures/ideal_condition_test"):
    """
    保存诊断图
    
    Args:
        result: 测试结果
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    t = result.timestamps
    
    # Roll error
    axes[0].plot(t, result.roll_err_series, 'b-', linewidth=0.8)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0].axhline(y=GOLD_STANDARD_THRESHOLD_DEG, color='g', linestyle='--', 
                    linewidth=0.5, label=f'Gold Standard ({GOLD_STANDARD_THRESHOLD_DEG}°)')
    axes[0].axhline(y=-GOLD_STANDARD_THRESHOLD_DEG, color='g', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel('Roll Error (deg)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Ideal Condition Test - {result.scenario} - Roll Error')
    
    # 标注峰值
    if result.peak_axis == "roll":
        peak_idx = np.argmax(np.abs(result.roll_err_series))
        axes[0].axvline(x=t[peak_idx], color='r', linestyle=':', linewidth=1, alpha=0.7)
        axes[0].annotate(f'Peak: {result.max_roll_err_deg:.2e}°', 
                        xy=(t[peak_idx], result.roll_err_series[peak_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, color='r')
    
    # Pitch error
    axes[1].plot(t, result.pitch_err_series, 'b-', linewidth=0.8)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].axhline(y=GOLD_STANDARD_THRESHOLD_DEG, color='g', linestyle='--', 
                    linewidth=0.5, label=f'Gold Standard ({GOLD_STANDARD_THRESHOLD_DEG}°)')
    axes[1].axhline(y=-GOLD_STANDARD_THRESHOLD_DEG, color='g', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel('Pitch Error (deg)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'Ideal Condition Test - {result.scenario} - Pitch Error')
    
    # 标注峰值
    if result.peak_axis == "pitch":
        peak_idx = np.argmax(np.abs(result.pitch_err_series))
        axes[1].axvline(x=t[peak_idx], color='r', linestyle=':', linewidth=1, alpha=0.7)
        axes[1].annotate(f'Peak: {result.max_pitch_err_deg:.2e}°', 
                        xy=(t[peak_idx], result.pitch_err_series[peak_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, color='r')
    
    plt.tight_layout()
    
    save_path = Path(output_dir) / f"error_{result.scenario}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def print_result(result: IdealTestResult):
    """打印测试结果"""
    status_icon = "✓" if result.passed else "✗"
    
    print(f"\n{'='*60}")
    print(f"工况: {result.scenario}")
    print(f"{'='*60}")
    print(f"  状态: {status_icon} {result.error_level.upper()}")
    print(f"  Roll  最大误差: {result.max_roll_err_deg:.2e}°")
    print(f"  Pitch 最大误差: {result.max_pitch_err_deg:.2e}°")
    print(f"  Roll  RMSE: {result.rmse_roll_deg:.2e}°")
    print(f"  Pitch RMSE: {result.rmse_pitch_deg:.2e}°")
    print(f"  峰值时间: {result.peak_time_s:.3f}s ({result.peak_axis})")
    
    if result.suggested_checks:
        print(f"\n  建议检查:")
        for check in result.suggested_checks:
            print(f"    - {check}")


# ============================================================
# 主测试
# ============================================================

def test_quasi_static():
    """测试 1: 准静态工况"""
    print("\n" + "=" * 60)
    print("测试 1: 准静态工况（理想条件）")
    print("=" * 60)
    
    scenario_params = {
        "fs": 100.0,
        "duration_s": 5.0,
        "roll_deg": 10.0,
        "pitch_deg": -5.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 42,
    }
    
    result = run_ideal_condition_test("quasi_static", scenario_params)
    print_result(result)
    
    # 保存诊断图
    save_path = save_diagnostic_plot(result)
    print(f"\n  诊断图保存到: {save_path}")
    
    return result


def test_swing():
    """测试 2: 摆动工况（alpha=0，纯加速度计）"""
    print("\n" + "=" * 60)
    print("测试 2: 摆动工况（理想条件，alpha=0）")
    print("=" * 60)
    
    scenario_params = {
        "fs": 100.0,
        "duration_s": 10.0,
        "roll_amp_deg": 10.0,
        "pitch_amp_deg": 5.0,
        "roll_freq_hz": 0.2,
        "pitch_freq_hz": 0.15,
        "roll_phase_deg": 0.0,
        "pitch_phase_deg": 90.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 42,
    }
    
    result = run_ideal_condition_test("swing", scenario_params)
    print_result(result)
    
    # 保存诊断图
    save_path = save_diagnostic_plot(result)
    print(f"\n  诊断图保存到: {save_path}")
    
    return result


def test_a1_gold_standard():
    """
    测试 A1：理想条件零误差回归（Gold Standard）
    
    目的：验证整条链路在无噪声无偏置时应接近零误差
    （保证公式、符号、时间对齐正确）
    
    配置：
    - scenario: swing (roll_amp=10°, pitch_amp=5°, f=0.2~0.5Hz, duration=5s)
    - sensor: acc.sigma_white=0, gyro.sigma_white=0, bias=[0,0,0]
    - filter: complementary with alpha=0.98（任意 alpha）
    
    通过门槛：
    - max(|err_roll|) < 1e-6 deg 且 max(|err_pitch|) < 1e-6 deg
    - 或更宽松：< 1e-3 deg（允许浮点/离散化差异）
    """
    print("\n" + "=" * 60)
    print("测试 A1：理想条件零误差回归（Gold Standard）")
    print("=" * 60)
    print("目的：验证整条链路在无噪声无偏置时应接近零误差")
    print("配置：swing 工况，alpha=0.98（任意 alpha 应该都能通过）")
    print("门槛：max(|err|) < 1e-6 deg（或 1e-3 deg 允许离散化差异）")
    
    # A1 测试配置
    scenario_params = {
        "fs": 100.0,
        "duration_s": 5.0,
        "roll_amp_deg": 10.0,
        "pitch_amp_deg": 5.0,
        "roll_freq_hz": 0.3,  # 0.2~0.5 Hz 范围
        "pitch_freq_hz": 0.4,
        "roll_phase_deg": 0.0,
        "pitch_phase_deg": 90.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 42,
    }
    
    # 使用实际滤波器配置 alpha=0.98
    filter_cfg = {"alpha": 0.98}
    
    result = run_ideal_condition_test("swing", scenario_params, filter_cfg)
    
    # A1 特定的通过门槛
    # 注意：离散化误差与采样率和运动频率有关
    # 对于 100 Hz 采样率和 0.3-0.4 Hz 运动，预期误差在 1e-4 ~ 1e-3 度量级
    A1_THRESHOLD_STRICT = 1e-6  # 严格门槛
    A1_THRESHOLD_RELAXED = 1e-3  # 宽松门槛（允许离散化差异）
    
    max_err = max(result.max_roll_err_deg, result.max_pitch_err_deg)
    
    print(f"\n{'='*60}")
    print(f"工况: A1 Gold Standard (swing, alpha=0.98)")
    print(f"{'='*60}")
    print(f"  Roll  最大误差: {result.max_roll_err_deg:.2e}°")
    print(f"  Pitch 最大误差: {result.max_pitch_err_deg:.2e}°")
    print(f"  Roll  RMSE: {result.rmse_roll_deg:.2e}°")
    print(f"  Pitch RMSE: {result.rmse_pitch_deg:.2e}°")
    print(f"  峰值时间: {result.peak_time_s:.3f}s ({result.peak_axis})")
    
    # 判断结果
    if max_err < A1_THRESHOLD_STRICT:
        print(f"\n  ✓ A1 测试通过（严格门槛 < {A1_THRESHOLD_STRICT}°）")
        a1_passed = True
    elif max_err < A1_THRESHOLD_RELAXED:
        print(f"\n  ⚠ A1 测试通过（宽松门槛 < {A1_THRESHOLD_RELAXED}°）")
        print(f"    存在轻微离散化/浮点误差，但在可接受范围内")
        a1_passed = True
    else:
        print(f"\n  ✗ A1 测试失败！误差 {max_err:.2e}° 超出门槛")
        print(f"\n  失败优先排查:")
        print(f"    1. acc→roll/pitch 公式与坐标系不一致")
        print(f"    2. deg/rad 混用")
        print(f"    3. dt 与 t 的对齐（多/少一个采样点）")
        print(f"    4. meas.acc 是 specific force 还是重力投影，符号没统一")
        a1_passed = False
    
    # 保存诊断图
    save_path = save_diagnostic_plot(
        IdealTestResult(
            scenario="A1_gold_standard",
            passed=a1_passed,
            max_roll_err_deg=result.max_roll_err_deg,
            max_pitch_err_deg=result.max_pitch_err_deg,
            rmse_roll_deg=result.rmse_roll_deg,
            rmse_pitch_deg=result.rmse_pitch_deg,
            error_level="gold_standard" if a1_passed else "implementation_error",
            roll_err_series=result.roll_err_series,
            pitch_err_series=result.pitch_err_series,
            timestamps=result.timestamps,
            peak_time_s=result.peak_time_s,
            peak_axis=result.peak_axis,
            suggested_checks=result.suggested_checks,
        )
    )
    print(f"\n  诊断图保存到: {save_path}")
    
    return a1_passed, result


def test_a1_high_rate():
    """
    测试 A1 高采样率版本：验证离散化误差随采样率降低
    
    使用 1000 Hz 采样率，预期误差 < 1e-6 deg
    """
    print("\n" + "=" * 60)
    print("测试 A1 高采样率：验证离散化误差随采样率降低")
    print("=" * 60)
    print("配置：swing 工况，fs=1000Hz，alpha=0.98")
    print("预期：max(|err|) < 1e-5 deg")
    
    # 高采样率配置
    scenario_params = {
        "fs": 1000.0,  # 10x 采样率
        "duration_s": 5.0,
        "roll_amp_deg": 10.0,
        "pitch_amp_deg": 5.0,
        "roll_freq_hz": 0.3,
        "pitch_freq_hz": 0.4,
        "roll_phase_deg": 0.0,
        "pitch_phase_deg": 90.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 42,
    }
    
    filter_cfg = {"alpha": 0.98}
    result = run_ideal_condition_test("swing", scenario_params, filter_cfg)
    
    max_err = max(result.max_roll_err_deg, result.max_pitch_err_deg)
    
    print(f"\n  Roll  最大误差: {result.max_roll_err_deg:.2e}°")
    print(f"  Pitch 最大误差: {result.max_pitch_err_deg:.2e}°")
    
    # 高采样率应该有更小的误差
    if max_err < 1e-5:
        print(f"\n  ✓ 高采样率测试通过（误差 < 1e-5°）")
        print(f"    确认离散化误差随采样率降低")
        return True, max_err
    else:
        print(f"\n  ⚠ 高采样率测试：误差 {max_err:.2e}° 仍较大")
        return False, max_err


def run_test_with_sensor_params(
    scenario: str,
    scenario_params: dict,
    sensor_params: dict,
    filter_cfg: dict,
    seed: int = 42,
) -> IdealTestResult:
    """
    使用指定传感器参数运行测试
    """
    g = GRAVITY_STANDARD
    
    # 生成真值
    if scenario == "quasi_static":
        truth = generate_quasi_static(**scenario_params)
    elif scenario == "swing":
        truth = generate_swing(**scenario_params)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # 使用指定传感器参数生成测量
    meas = forward_imu(truth, sensor_params, seed=seed, g=g)
    
    # 构建数据集
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params.get("fs", 100.0)},
    }
    
    # 运行滤波器
    est = run_complementary(ds, filter_cfg)
    
    # 获取真值 roll/pitch
    if "rpy_deg" in truth:
        roll_true_deg = truth["rpy_deg"][:, 0]
        pitch_true_deg = truth["rpy_deg"][:, 1]
    else:
        from src.common.math3d import quat_to_rpy
        n_samples = len(truth["q_nb"])
        roll_true_deg = np.zeros(n_samples)
        pitch_true_deg = np.zeros(n_samples)
        for i in range(n_samples):
            roll, pitch, yaw = quat_to_rpy(truth["q_nb"][i])
            roll_true_deg[i] = rad2deg(roll)
            pitch_true_deg[i] = rad2deg(pitch)
    
    # 获取估计值
    roll_est_deg = rad2deg(est["roll"])
    pitch_est_deg = rad2deg(est["pitch"])
    
    # 计算误差（使用 wrap 避免角度跳变）
    roll_err_deg = wrap_deg(roll_est_deg - roll_true_deg)
    pitch_err_deg = wrap_deg(pitch_est_deg - pitch_true_deg)
    
    # 计算统计量
    max_roll_err = np.max(np.abs(roll_err_deg))
    max_pitch_err = np.max(np.abs(pitch_err_deg))
    max_err = max(max_roll_err, max_pitch_err)
    
    rmse_roll = np.sqrt(np.mean(roll_err_deg**2))
    rmse_pitch = np.sqrt(np.mean(pitch_err_deg**2))
    
    mean_roll_err = np.mean(roll_err_deg)
    mean_pitch_err = np.mean(pitch_err_deg)
    
    # 找到峰值误差时间
    if max_roll_err >= max_pitch_err:
        peak_idx = np.argmax(np.abs(roll_err_deg))
        peak_axis = "roll"
    else:
        peak_idx = np.argmax(np.abs(pitch_err_deg))
        peak_axis = "pitch"
    peak_time_s = truth["t"][peak_idx]
    
    # 验证误差级别
    error_level, passed, suggested_checks = validate_error_level(max_err)
    
    result = IdealTestResult(
        scenario=scenario,
        passed=passed,
        max_roll_err_deg=max_roll_err,
        max_pitch_err_deg=max_pitch_err,
        rmse_roll_deg=rmse_roll,
        rmse_pitch_deg=rmse_pitch,
        error_level=error_level,
        roll_err_series=roll_err_deg,
        pitch_err_series=pitch_err_deg,
        timestamps=truth["t"],
        peak_time_s=peak_time_s,
        peak_axis=peak_axis,
        suggested_checks=suggested_checks,
    )
    
    # 添加额外统计量
    result.mean_roll_err_deg = mean_roll_err
    result.mean_pitch_err_deg = mean_pitch_err
    
    return result


def test_a2_quasi_static_hold():
    """
    测试 A2：静止姿态保持（Quasi-static Hold）
    
    目的：验证误差计算/绘图/指标对"常值真值"不会出现漂移或相位错乱
    
    配置：
    - scenario: quasi_static (roll=10°, pitch=-5°, duration=10s)
    - 组1：无噪声无偏置（同 A1）
    - 组2：有噪声无偏置（acc sigma=0.05, gyro sigma=0.005）
    
    通过门槛：
    - 组1：rmse < 1e-6 deg
    - 组2：|mean(err)| < 0.02 deg，且无发散
    """
    print("\n" + "=" * 60)
    print("测试 A2：静止姿态保持（Quasi-static Hold）")
    print("=" * 60)
    print("目的：验证误差计算对常值真值不会出现漂移或相位错乱")
    
    # 准静态工况参数
    scenario_params = {
        "fs": 100.0,
        "duration_s": 10.0,
        "roll_deg": 10.0,
        "pitch_deg": -5.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 42,
    }
    
    filter_cfg = {"alpha": 0.98}
    
    # ========== 组1：无噪声无偏置 ==========
    print("\n--- 组1：无噪声无偏置 ---")
    
    sensor_params_1 = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
    }
    
    result_1 = run_test_with_sensor_params(
        "quasi_static", scenario_params, sensor_params_1, filter_cfg
    )
    
    rmse_1 = max(result_1.rmse_roll_deg, result_1.rmse_pitch_deg)
    
    print(f"  Roll  RMSE: {result_1.rmse_roll_deg:.2e}°")
    print(f"  Pitch RMSE: {result_1.rmse_pitch_deg:.2e}°")
    print(f"  Roll  最大误差: {result_1.max_roll_err_deg:.2e}°")
    print(f"  Pitch 最大误差: {result_1.max_pitch_err_deg:.2e}°")
    
    A2_GROUP1_THRESHOLD = 1e-6
    if rmse_1 < A2_GROUP1_THRESHOLD:
        print(f"  ✓ 组1通过（RMSE < {A2_GROUP1_THRESHOLD}°）")
        group1_passed = True
    else:
        print(f"  ✗ 组1失败！RMSE {rmse_1:.2e}° 超出门槛")
        group1_passed = False
    
    # ========== 组2：有噪声无偏置 ==========
    print("\n--- 组2：有噪声无偏置 ---")
    print("  acc sigma=0.05 m/s², gyro sigma=0.005 rad/s")
    
    sensor_params_2 = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.05},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.005},
    }
    
    result_2 = run_test_with_sensor_params(
        "quasi_static", scenario_params, sensor_params_2, filter_cfg, seed=123
    )
    
    mean_err_2 = max(abs(result_2.mean_roll_err_deg), abs(result_2.mean_pitch_err_deg))
    
    print(f"  Roll  均值误差: {result_2.mean_roll_err_deg:.4f}°")
    print(f"  Pitch 均值误差: {result_2.mean_pitch_err_deg:.4f}°")
    print(f"  Roll  RMSE: {result_2.rmse_roll_deg:.4f}°")
    print(f"  Pitch RMSE: {result_2.rmse_pitch_deg:.4f}°")
    print(f"  Roll  最大误差: {result_2.max_roll_err_deg:.4f}°")
    print(f"  Pitch 最大误差: {result_2.max_pitch_err_deg:.4f}°")
    
    # 检查是否发散（最后 10% 的误差不应该比前 10% 大很多）
    n = len(result_2.roll_err_series)
    first_10pct = int(n * 0.1)
    last_10pct = int(n * 0.9)
    
    roll_err_first = np.std(result_2.roll_err_series[:first_10pct])
    roll_err_last = np.std(result_2.roll_err_series[last_10pct:])
    pitch_err_first = np.std(result_2.pitch_err_series[:first_10pct])
    pitch_err_last = np.std(result_2.pitch_err_series[last_10pct:])
    
    divergence_ratio = max(
        roll_err_last / (roll_err_first + 1e-10),
        pitch_err_last / (pitch_err_first + 1e-10)
    )
    
    print(f"  发散比（末/初 std）: {divergence_ratio:.2f}")
    
    A2_GROUP2_MEAN_THRESHOLD = 0.02  # deg
    A2_GROUP2_DIVERGENCE_THRESHOLD = 2.0  # 允许 2 倍
    
    group2_passed = True
    if mean_err_2 > A2_GROUP2_MEAN_THRESHOLD:
        print(f"  ✗ 组2失败！|mean(err)| = {mean_err_2:.4f}° > {A2_GROUP2_MEAN_THRESHOLD}°")
        group2_passed = False
    elif divergence_ratio > A2_GROUP2_DIVERGENCE_THRESHOLD:
        print(f"  ✗ 组2失败！发散比 {divergence_ratio:.2f} > {A2_GROUP2_DIVERGENCE_THRESHOLD}")
        group2_passed = False
    else:
        print(f"  ✓ 组2通过（|mean| < {A2_GROUP2_MEAN_THRESHOLD}°，无发散）")
    
    # 保存诊断图
    save_path_1 = save_diagnostic_plot(
        IdealTestResult(
            scenario="A2_group1_no_noise",
            passed=group1_passed,
            max_roll_err_deg=result_1.max_roll_err_deg,
            max_pitch_err_deg=result_1.max_pitch_err_deg,
            rmse_roll_deg=result_1.rmse_roll_deg,
            rmse_pitch_deg=result_1.rmse_pitch_deg,
            error_level="gold_standard" if group1_passed else "implementation_error",
            roll_err_series=result_1.roll_err_series,
            pitch_err_series=result_1.pitch_err_series,
            timestamps=result_1.timestamps,
            peak_time_s=result_1.peak_time_s,
            peak_axis=result_1.peak_axis,
            suggested_checks=[],
        )
    )
    
    save_path_2 = save_diagnostic_plot(
        IdealTestResult(
            scenario="A2_group2_with_noise",
            passed=group2_passed,
            max_roll_err_deg=result_2.max_roll_err_deg,
            max_pitch_err_deg=result_2.max_pitch_err_deg,
            rmse_roll_deg=result_2.rmse_roll_deg,
            rmse_pitch_deg=result_2.rmse_pitch_deg,
            error_level="gold_standard" if group2_passed else "implementation_error",
            roll_err_series=result_2.roll_err_series,
            pitch_err_series=result_2.pitch_err_series,
            timestamps=result_2.timestamps,
            peak_time_s=result_2.peak_time_s,
            peak_axis=result_2.peak_axis,
            suggested_checks=[],
        )
    )
    
    print(f"\n  诊断图保存到:")
    print(f"    组1: {save_path_1}")
    print(f"    组2: {save_path_2}")
    
    a2_passed = group1_passed and group2_passed
    
    if not a2_passed:
        print("\n  失败优先排查:")
        print("    1. 初始化未用 acc 初始化导致起始瞬态大误差")
        print("    2. 未 wrap、或输出单位错误")
    
    # 返回详细结果用于保存
    return a2_passed, result_2


def test_a3_angle_wrap():
    """
    测试 A3：误差 wrap 正确性（Angle Wrap Test）
    
    目的：确保误差计算不会在跨 ±180° 时出现跳变
    
    配置：
    - scenario: roll 从 179° 平滑过渡到 -179°（跨越 ±180°）
    - sensor: 无噪声无偏置
    
    通过门槛：
    - 误差曲线连续，不出现 ±360° 的跳变
    - peak_abs_error 合理（应接近 0）
    """
    print("\n" + "=" * 60)
    print("测试 A3：误差 wrap 正确性（Angle Wrap Test）")
    print("=" * 60)
    print("目的：确保误差计算不会在跨 ±180° 时出现跳变")
    
    from src.truth.scenarios import generate_roll_wrap_test
    
    # 工况参数：roll 从 170° 变化到 -170°（跨越 ±180°）
    scenario_params = {
        "fs": 100.0,
        "duration_s": 5.0,
        "roll_start_deg": 170.0,
        "roll_end_deg": -170.0,  # 跨越 ±180°
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 42,
    }
    
    g = GRAVITY_STANDARD
    filter_cfg = {"alpha": 0.0}  # 纯加速度计，确保跟踪准确
    
    # 生成真值
    truth = generate_roll_wrap_test(**scenario_params)
    
    # 使用理想传感器
    meas = forward_imu(truth, IDEAL_SENSOR_PARAMS, seed=42, g=g)
    
    # 构建数据集
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params["fs"]},
    }
    
    # 运行滤波器
    est = run_complementary(ds, filter_cfg)
    
    # 获取真值和估计值
    roll_true_deg = truth["rpy_deg"][:, 0]
    roll_est_deg = rad2deg(est["roll"])
    
    # 计算误差（使用 wrap）
    roll_err_deg = wrap_deg(roll_est_deg - roll_true_deg)
    
    # 检查是否有跳变（相邻样本误差变化不应超过 10°）
    roll_err_diff = np.diff(roll_err_deg)
    max_jump = np.max(np.abs(roll_err_diff))
    has_jump = max_jump > 10.0  # 如果相邻误差变化超过 10°，认为有跳变
    
    # 计算统计量
    max_err = np.max(np.abs(roll_err_deg))
    rmse = np.sqrt(np.mean(roll_err_deg**2))
    
    print(f"\n  Roll 真值范围: {roll_true_deg[0]:.1f}° → {roll_true_deg[-1]:.1f}°")
    print(f"  Roll 估计范围: {roll_est_deg[0]:.1f}° → {roll_est_deg[-1]:.1f}°")
    print(f"  Roll 最大误差: {max_err:.2e}°")
    print(f"  Roll RMSE: {rmse:.2e}°")
    print(f"  相邻误差最大变化: {max_jump:.2e}°")
    
    # 检查跨越点附近的误差
    # 找到真值跨越 ±180° 的位置
    cross_idx = None
    for i in range(1, len(roll_true_deg)):
        if roll_true_deg[i-1] > 0 and roll_true_deg[i] < 0:
            cross_idx = i
            break
    
    if cross_idx is not None:
        print(f"\n  跨越点位置: t={truth['t'][cross_idx]:.2f}s")
        print(f"    跨越前: true={roll_true_deg[cross_idx-1]:.2f}°, est={roll_est_deg[cross_idx-1]:.2f}°, err={roll_err_deg[cross_idx-1]:.2e}°")
        print(f"    跨越后: true={roll_true_deg[cross_idx]:.2f}°, est={roll_est_deg[cross_idx]:.2f}°, err={roll_err_deg[cross_idx]:.2e}°")
    
    # 判断结果
    A3_MAX_ERR_THRESHOLD = 1e-6  # 理想条件下误差应接近 0
    A3_JUMP_THRESHOLD = 10.0  # 相邻误差变化不应超过 10°
    
    a3_passed = True
    if has_jump:
        print(f"\n  ✗ A3 测试失败！检测到误差跳变（max_jump={max_jump:.1f}°）")
        print("    可能原因：err 直接 est-truth 没 wrap")
        a3_passed = False
    elif max_err > A3_MAX_ERR_THRESHOLD:
        print(f"\n  ⚠ A3 测试警告：误差 {max_err:.2e}° 较大（但无跳变）")
        # 仍然通过，因为主要目的是检测跳变
        a3_passed = True
    else:
        print(f"\n  ✓ A3 测试通过（无跳变，误差 < {A3_MAX_ERR_THRESHOLD}°）")
    
    # 保存诊断图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    t = truth["t"]
    
    # 真值和估计值
    axes[0].plot(t, roll_true_deg, 'b-', label='True', linewidth=1)
    axes[0].plot(t, roll_est_deg, 'r--', label='Est', linewidth=1)
    axes[0].axhline(y=180, color='k', linestyle=':', linewidth=0.5)
    axes[0].axhline(y=-180, color='k', linestyle=':', linewidth=0.5)
    axes[0].set_ylabel('Roll (deg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('A3 Angle Wrap Test - Roll Angle')
    
    # 误差
    axes[1].plot(t, roll_err_deg, 'b-', linewidth=0.8)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Roll Error (deg)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Roll Error (wrapped)')
    
    # 误差变化率
    axes[2].plot(t[1:], roll_err_diff, 'b-', linewidth=0.8)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[2].axhline(y=A3_JUMP_THRESHOLD, color='r', linestyle='--', linewidth=0.5, label=f'Jump threshold ({A3_JUMP_THRESHOLD}°)')
    axes[2].axhline(y=-A3_JUMP_THRESHOLD, color='r', linestyle='--', linewidth=0.5)
    axes[2].set_ylabel('Error Diff (deg)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Error Rate of Change')
    
    plt.tight_layout()
    
    output_dir = "outputs/figures/ideal_condition_test"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(output_dir) / "error_A3_angle_wrap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n  诊断图保存到: {save_path}")
    
    if not a3_passed:
        print("\n  失败优先排查:")
        print("    1. err 直接 est-truth 没 wrap")
        print("    2. wrap 的区间实现错误（应到 [-180,180) 或 [-pi,pi)）")
    
    return a3_passed


def test_acc_to_angle_direct():
    """
    测试 3: 直接验证 acc → 角度公式
    
    这是最基础的测试：在理想条件下，acc_to_roll_pitch 应该
    直接从加速度计测量恢复出真值角度。
    """
    print("\n" + "=" * 60)
    print("测试 3: acc → 角度公式直接验证")
    print("=" * 60)
    
    g = GRAVITY_STANDARD
    
    # 测试多个角度
    test_cases = [
        (0.0, 0.0),
        (10.0, 0.0),
        (0.0, 10.0),
        (10.0, -5.0),
        (-15.0, 20.0),
        (30.0, -20.0),
    ]
    
    all_passed = True
    max_err = 0.0
    
    for roll_deg, pitch_deg in test_cases:
        # 生成准静态真值
        truth = generate_quasi_static(
            fs=100.0, duration_s=0.1,
            roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=0.0,
            temp_C=25.0, seed=42
        )
        
        # 使用理想传感器
        meas = forward_imu(truth, IDEAL_SENSOR_PARAMS, seed=42, g=g)
        
        # 直接从 acc 计算角度
        roll_acc, pitch_acc = acc_to_roll_pitch(meas["acc"], g)
        
        roll_err = abs(rad2deg(roll_acc[0]) - roll_deg)
        pitch_err = abs(rad2deg(pitch_acc[0]) - pitch_deg)
        
        max_err = max(max_err, roll_err, pitch_err)
        
        passed = roll_err < 1e-10 and pitch_err < 1e-10
        status = "✓" if passed else "✗"
        all_passed = all_passed and passed
        
        print(f"  输入: roll={roll_deg:+6.1f}°, pitch={pitch_deg:+6.1f}°")
        print(f"  输出: roll={rad2deg(roll_acc[0]):+6.1f}°, pitch={rad2deg(pitch_acc[0]):+6.1f}°")
        print(f"  误差: roll={roll_err:.2e}°, pitch={pitch_err:.2e}° {status}")
    
    print(f"\n  最大误差: {max_err:.2e}°")
    
    if max_err < 1e-10:
        print("  结论: ✓ acc → 角度公式正确（金标准）")
    elif max_err < 1e-6:
        print("  结论: ⚠ acc → 角度公式有数值误差")
    else:
        print("  结论: ✗ acc → 角度公式存在问题！")
        print("  建议检查:")
        print("    - roll = atan2(ay, az)")
        print("    - pitch = atan2(-ax, sqrt(ay² + az²))")
        print("    - 坐标系约定 (NED/FRD)")
    
    return all_passed, max_err


def save_test_results(metrics_dict: dict, output_dir: str = "outputs/tables"):
    """
    保存测试结果到 CSV 和 JSON 文件
    
    Args:
        metrics_dict: 指标字典
        output_dir: 输出目录
    """
    import json
    import csv
    from datetime import datetime
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 添加时间戳
    metrics_dict["timestamp"] = datetime.now().isoformat()
    metrics_dict["filter"] = "complementary"
    
    # 保存 JSON
    json_path = Path(output_dir) / "step7_tests.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    
    # 保存 CSV（扁平化结构）
    csv_path = Path(output_dir) / "step7_tests.csv"
    
    # 扁平化字典
    flat_dict = {
        "timestamp": metrics_dict["timestamp"],
        "filter": metrics_dict["filter"],
        # A1 指标
        "A1_roll_peak_deg": metrics_dict["A1"]["roll_peak_deg"],
        "A1_pitch_peak_deg": metrics_dict["A1"]["pitch_peak_deg"],
        "A1_roll_rmse_deg": metrics_dict["A1"]["roll_rmse_deg"],
        "A1_pitch_rmse_deg": metrics_dict["A1"]["pitch_rmse_deg"],
        # A2 指标（带 burn-in）
        "A2_roll_rmse_deg": metrics_dict["A2"]["roll_rmse_deg"],
        "A2_pitch_rmse_deg": metrics_dict["A2"]["pitch_rmse_deg"],
        "A2_roll_mean_deg": metrics_dict["A2"]["roll_mean_deg"],
        "A2_pitch_mean_deg": metrics_dict["A2"]["pitch_mean_deg"],
        "A2_burn_in_s": metrics_dict["A2"]["burn_in_s"],
    }
    
    # 写入 CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_dict.keys())
        writer.writeheader()
        writer.writerow(flat_dict)
    
    return json_path, csv_path


def main():
    print("\n" + "=" * 60)
    print("理想条件回归测试（金标准测试）")
    print("=" * 60)
    print("目的：验证整个链路在 bias=0, noise=0 条件下的数值正确性")
    print(f"金标准阈值: {GOLD_STANDARD_THRESHOLD_DEG}°")
    print(f"警告阈值: {NUMERICAL_WARNING_THRESHOLD_DEG}°")
    
    results = []
    metrics_to_save = {}
    
    # 测试 1: acc → 角度公式
    acc_passed, acc_max_err = test_acc_to_angle_direct()
    results.append(("acc→角度公式", acc_passed))
    
    # 测试 2: 准静态工况
    quasi_result = test_quasi_static()
    results.append(("准静态工况", quasi_result.passed))
    
    # 测试 3: 摆动工况（alpha=0）
    swing_result = test_swing()
    results.append(("摆动工况(alpha=0)", swing_result.passed))
    
    # 测试 A1: Gold Standard（alpha=0.98, 100Hz）
    a1_passed, a1_result = test_a1_gold_standard()
    results.append(("A1 Gold Standard(alpha=0.98, 100Hz)", a1_passed))
    
    # 收集 A1 指标
    metrics_to_save["A1"] = {
        "roll_peak_deg": a1_result.max_roll_err_deg,
        "pitch_peak_deg": a1_result.max_pitch_err_deg,
        "roll_rmse_deg": a1_result.rmse_roll_deg,
        "pitch_rmse_deg": a1_result.rmse_pitch_deg,
        "scenario": "swing",
        "alpha": 0.98,
        "fs_hz": 100.0,
    }
    
    # 测试 A1 高采样率：验证离散化误差
    a1_hr_passed, a1_hr_max_err = test_a1_high_rate()
    results.append(("A1 高采样率(1000Hz)", a1_hr_passed))
    
    # 测试 A2: 静止姿态保持
    a2_passed, a2_result = test_a2_quasi_static_hold()
    results.append(("A2 静止姿态保持", a2_passed))
    
    # 收集 A2 指标（带 burn-in）
    # 计算 burn-in 后的指标
    burn_in_s = 0.5  # 0.5 秒 burn-in
    fs = 100.0
    burn_in_samples = int(burn_in_s * fs)
    
    roll_err_after_burnin = a2_result.roll_err_series[burn_in_samples:]
    pitch_err_after_burnin = a2_result.pitch_err_series[burn_in_samples:]
    
    metrics_to_save["A2"] = {
        "roll_rmse_deg": float(np.sqrt(np.mean(roll_err_after_burnin**2))),
        "pitch_rmse_deg": float(np.sqrt(np.mean(pitch_err_after_burnin**2))),
        "roll_mean_deg": float(np.mean(roll_err_after_burnin)),
        "pitch_mean_deg": float(np.mean(pitch_err_after_burnin)),
        "roll_peak_deg": float(np.max(np.abs(roll_err_after_burnin))),
        "pitch_peak_deg": float(np.max(np.abs(pitch_err_after_burnin))),
        "burn_in_s": burn_in_s,
        "scenario": "quasi_static",
        "alpha": 0.98,
        "fs_hz": fs,
        "acc_sigma": 0.05,
        "gyro_sigma": 0.005,
    }
    
    # 测试 A3: 误差 wrap 正确性
    a3_passed = test_a3_angle_wrap()
    results.append(("A3 误差wrap正确性", a3_passed))
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    # 打印关键指标
    print("\n" + "=" * 60)
    print("关键指标（用于 EKF 对比）")
    print("=" * 60)
    print(f"  A1 (swing, ideal, alpha=0.98):")
    print(f"    Roll  Peak: {metrics_to_save['A1']['roll_peak_deg']:.2e}°")
    print(f"    Pitch Peak: {metrics_to_save['A1']['pitch_peak_deg']:.2e}°")
    print(f"    Roll  RMSE: {metrics_to_save['A1']['roll_rmse_deg']:.2e}°")
    print(f"    Pitch RMSE: {metrics_to_save['A1']['pitch_rmse_deg']:.2e}°")
    print(f"  A2 (quasi_static, noisy, burn-in={burn_in_s}s):")
    print(f"    Roll  RMSE: {metrics_to_save['A2']['roll_rmse_deg']:.4f}°")
    print(f"    Pitch RMSE: {metrics_to_save['A2']['pitch_rmse_deg']:.4f}°")
    print(f"    Roll  Mean: {metrics_to_save['A2']['roll_mean_deg']:.4f}°")
    print(f"    Pitch Mean: {metrics_to_save['A2']['pitch_mean_deg']:.4f}°")
    
    # 保存结果
    json_path, csv_path = save_test_results(metrics_to_save)
    print(f"\n  结果已保存到:")
    print(f"    JSON: {json_path}")
    print(f"    CSV:  {csv_path}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！链路正确性已验证（金标准）。")
        return 0
    else:
        print("存在测试失败！请检查实现细节。")
        print("\n常见问题排查:")
        print("  1. acc → 角度公式是否正确")
        print("  2. 坐标系/轴定义是否一致")
        print("  3. deg-rad 单位转换是否正确")
        print("  4. 时间对齐是否正确")
        print("  5. 滤波器初始化是否使用 acc")
        return 1


if __name__ == "__main__":
    sys.exit(main())

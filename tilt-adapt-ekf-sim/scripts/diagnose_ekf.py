#!/usr/bin/env python
"""
EKF 诊断脚本

按优先级排查三个根因：
1. 量测模型 h(x) 与 meas.acc 定义是否一致
2. R 是否与实际噪声匹配（NIS 应接近 3）
3. Q_bias 是否合适

目标：
- innovation 在理想条件下应接近 0 均值
- NIS 均值应接近自由度（3）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.truth.scenarios import generate_quasi_static
from src.truth.frames import GRAVITY_STANDARD, gravity_n
from src.sensors.imu_model import forward_imu
from src.filters.ekf_fixed import run_ekf_fixed, EKFFixed
from src.filters.complementary import run_complementary, acc_to_roll_pitch
from src.common.math3d import quat_to_R_bn, quat_to_rpy, rpy_to_quat, rad2deg


def diagnose_measurement_model():
    """
    诊断 1: 量测模型一致性检查
    
    在准静态、无噪无偏置条件下：
    - 计算 r = meas.acc - h(x) 的均值和符号
    - 如果 mean(r) 不接近 0，量测模型就错了
    """
    print("=" * 60)
    print("诊断 1: 量测模型一致性检查")
    print("=" * 60)
    
    # 准静态工况，无噪声无偏置
    scenario_params = {
        "fs": 100.0,
        "duration_s": 10.0,
        "roll_deg": 5.0,
        "pitch_deg": -3.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 1,
    }
    
    # 理想传感器（无偏置无噪声）
    sensor_params = {
        "acc": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
        "gyro": {"bias0": [0.0, 0.0, 0.0], "sigma_white": 0.0},
    }
    
    print("\n生成理想数据（无噪声无偏置）...")
    truth = generate_quasi_static(**scenario_params)
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    
    # 检查 IMU 模型输出
    print("\n[IMU 模型输出]")
    print(f"  acc[0] = {meas['acc'][0]}")
    print(f"  ||acc[0]|| = {np.linalg.norm(meas['acc'][0]):.6f} (应接近 g={GRAVITY_STANDARD})")
    
    # 检查 EKF 量测预测
    print("\n[EKF 量测预测]")
    q_true = truth["q_nb"][0]
    R_bn = quat_to_R_bn(q_true)
    g_n = gravity_n(GRAVITY_STANDARD)
    
    # EKF 的预测: acc_pred = R_bn @ g_n
    acc_pred_ekf = R_bn @ g_n
    print(f"  EKF h(x) = R_bn @ g_n = {acc_pred_ekf}")
    print(f"  ||h(x)|| = {np.linalg.norm(acc_pred_ekf):.6f}")
    
    # 计算 innovation
    innovation = meas['acc'][0] - acc_pred_ekf
    print(f"\n[Innovation = meas - h(x)]")
    print(f"  innovation = {innovation}")
    print(f"  ||innovation|| = {np.linalg.norm(innovation):.6f}")
    
    # 判断
    if np.linalg.norm(innovation) < 0.01:
        print("\n✓ 量测模型一致！innovation 接近 0")
        return True
    else:
        print(f"\n✗ 量测模型不一致！innovation 应接近 0，实际为 {np.linalg.norm(innovation):.4f}")
        print("  可能原因：")
        print("  - IMU 模型和 EKF 对 specific force 的定义不同")
        print("  - 重力方向符号不一致")
        return False


def diagnose_nis_calibration():
    """
    诊断 2: NIS 校准检查
    
    目标：稳态段 mean_NIS ≈ 3（自由度=3）
    如果 NIS 偏低，说明 R 设得过大
    """
    print("\n" + "=" * 60)
    print("诊断 2: NIS 校准检查")
    print("=" * 60)
    
    # 准静态工况，有噪声有偏置
    scenario_params = {
        "fs": 100.0,
        "duration_s": 30.0,
        "roll_deg": 5.0,
        "pitch_deg": -3.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 1,
    }
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    print("\n生成数据...")
    truth = generate_quasi_static(**scenario_params)
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params["fs"]},
    }
    
    # 当前 EKF 参数
    filter_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-10,
        "R_acc": 0.05**2,  # 当前值
    }
    
    print(f"\n当前 EKF 参数:")
    print(f"  Q_gyro = {filter_cfg['Q_gyro']}")
    print(f"  Q_bias = {filter_cfg['Q_bias']}")
    print(f"  R_acc = {filter_cfg['R_acc']} (σ = {np.sqrt(filter_cfg['R_acc']):.4f} m/s²)")
    
    print("\n运行 EKF...")
    est = run_ekf_fixed(ds, filter_cfg)
    
    # 分析 NIS
    nis = est["debug"]["nis"]
    burn_in = int(3.0 * scenario_params["fs"])  # 3s burn-in
    nis_stable = nis[burn_in:]
    
    mean_nis = np.mean(nis_stable)
    std_nis = np.std(nis_stable)
    
    print(f"\n[NIS 统计（burn-in 后）]")
    print(f"  mean_NIS = {mean_nis:.2f} (目标: 3.0)")
    print(f"  std_NIS = {std_nis:.2f}")
    print(f"  NIS 偏离比例 = {mean_nis / 3.0:.2f}x")
    
    # 建议的 R 校准
    if mean_nis < 2.5:
        r_scale = mean_nis / 3.0
        r_new = filter_cfg['R_acc'] * r_scale
        print(f"\n[建议] NIS 偏低，R 设得过大")
        print(f"  建议 R_acc = {r_new:.6f} (缩小 {1/r_scale:.2f}x)")
    elif mean_nis > 4.0:
        r_scale = mean_nis / 3.0
        r_new = filter_cfg['R_acc'] * r_scale
        print(f"\n[建议] NIS 偏高，R 设得过小")
        print(f"  建议 R_acc = {r_new:.6f} (放大 {r_scale:.2f}x)")
    else:
        print(f"\n✓ NIS 接近目标值，R 校准良好")
    
    return mean_nis, filter_cfg['R_acc']


def diagnose_bias_estimation():
    """
    诊断 3: 偏置估计检查
    
    检查 EKF 估计的 gyro bias 是否接近真值
    """
    print("\n" + "=" * 60)
    print("诊断 3: 偏置估计检查")
    print("=" * 60)
    
    scenario_params = {
        "fs": 100.0,
        "duration_s": 30.0,
        "roll_deg": 5.0,
        "pitch_deg": -3.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 1,
    }
    
    # 已知的真实偏置
    true_gyro_bias = np.array([0.001, 0.001, -0.002])  # rad/s
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": true_gyro_bias.tolist(), "sigma_white": 0.001},
    }
    
    print("\n生成数据...")
    truth = generate_quasi_static(**scenario_params)
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params["fs"]},
    }
    
    filter_cfg = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-10,
        "R_acc": 0.05**2,
    }
    
    print("\n运行 EKF...")
    est = run_ekf_fixed(ds, filter_cfg)
    
    # 最终偏置估计
    final_bias = est["bias_gyro"][-1]
    bias_error = final_bias - true_gyro_bias
    
    print(f"\n[偏置估计结果]")
    print(f"  真值:   [{true_gyro_bias[0]*1000:.3f}, {true_gyro_bias[1]*1000:.3f}, {true_gyro_bias[2]*1000:.3f}] mrad/s")
    print(f"  估计:   [{final_bias[0]*1000:.3f}, {final_bias[1]*1000:.3f}, {final_bias[2]*1000:.3f}] mrad/s")
    print(f"  误差:   [{bias_error[0]*1000:.3f}, {bias_error[1]*1000:.3f}, {bias_error[2]*1000:.3f}] mrad/s")
    print(f"  误差范数: {np.linalg.norm(bias_error)*1000:.3f} mrad/s")
    
    return final_bias, true_gyro_bias


def auto_calibrate_r():
    """
    自动校准 R 使 NIS ≈ 3
    """
    print("\n" + "=" * 60)
    print("自动校准 R")
    print("=" * 60)
    
    scenario_params = {
        "fs": 100.0,
        "duration_s": 30.0,
        "roll_deg": 5.0,
        "pitch_deg": -3.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 1,
    }
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    truth = generate_quasi_static(**scenario_params)
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params["fs"]},
    }
    
    truth_for_metrics = {"rpy_deg": np.column_stack([
        np.full(len(truth["t"]), scenario_params["roll_deg"]),
        np.full(len(truth["t"]), scenario_params["pitch_deg"]),
        np.full(len(truth["t"]), scenario_params["yaw_deg"])
    ])}
    
    # 迭代校准
    R_acc = 0.05**2  # 初始值
    
    print("\n迭代校准过程:")
    for iteration in range(5):
        filter_cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-10,
            "R_acc": R_acc,
        }
        
        est = run_ekf_fixed(ds, filter_cfg)
        nis = est["debug"]["nis"]
        burn_in = int(3.0 * scenario_params["fs"])
        mean_nis = np.mean(nis[burn_in:])
        
        # 计算 RMSE
        roll_err = rad2deg(est["roll"]) - truth_for_metrics["rpy_deg"][:, 0]
        pitch_err = rad2deg(est["pitch"]) - truth_for_metrics["rpy_deg"][:, 1]
        rmse_roll = np.sqrt(np.mean(roll_err[burn_in:]**2))
        rmse_pitch = np.sqrt(np.mean(pitch_err[burn_in:]**2))
        
        print(f"  迭代 {iteration+1}: R_acc={R_acc:.6f}, mean_NIS={mean_nis:.2f}, RMSE=[{rmse_roll:.4f}°, {rmse_pitch:.4f}°]")
        
        if abs(mean_nis - 3.0) < 0.2:
            print(f"\n✓ 校准完成！")
            break
        
        # 更新 R
        R_acc = R_acc * (mean_nis / 3.0)
    
    print(f"\n推荐参数:")
    print(f"  R_acc = {R_acc:.6f}")
    
    return R_acc


def compare_with_complementary():
    """
    对比 EKF 和互补滤波性能
    """
    print("\n" + "=" * 60)
    print("EKF vs 互补滤波性能对比")
    print("=" * 60)
    
    scenario_params = {
        "fs": 100.0,
        "duration_s": 30.0,
        "roll_deg": 5.0,
        "pitch_deg": -3.0,
        "yaw_deg": 0.0,
        "temp_C": 25.0,
        "seed": 1,
    }
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    truth = generate_quasi_static(**scenario_params)
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params["fs"]},
    }
    
    truth_rpy = np.column_stack([
        np.full(len(truth["t"]), scenario_params["roll_deg"]),
        np.full(len(truth["t"]), scenario_params["pitch_deg"]),
        np.full(len(truth["t"]), scenario_params["yaw_deg"])
    ])
    
    burn_in = int(3.0 * scenario_params["fs"])
    
    # 互补滤波
    est_comp = run_complementary(ds, {"alpha": 0.98})
    roll_err_comp = rad2deg(est_comp["roll"]) - truth_rpy[:, 0]
    pitch_err_comp = rad2deg(est_comp["pitch"]) - truth_rpy[:, 1]
    rmse_comp_roll = np.sqrt(np.mean(roll_err_comp[burn_in:]**2))
    rmse_comp_pitch = np.sqrt(np.mean(pitch_err_comp[burn_in:]**2))
    
    print(f"\n互补滤波 (alpha=0.98):")
    print(f"  RMSE Roll:  {rmse_comp_roll:.4f}°")
    print(f"  RMSE Pitch: {rmse_comp_pitch:.4f}°")
    
    # EKF - 原始参数
    filter_cfg_old = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-10,
        "R_acc": 0.05**2,
    }
    est_ekf_old = run_ekf_fixed(ds, filter_cfg_old)
    roll_err_ekf_old = rad2deg(est_ekf_old["roll"]) - truth_rpy[:, 0]
    pitch_err_ekf_old = rad2deg(est_ekf_old["pitch"]) - truth_rpy[:, 1]
    rmse_ekf_old_roll = np.sqrt(np.mean(roll_err_ekf_old[burn_in:]**2))
    rmse_ekf_old_pitch = np.sqrt(np.mean(pitch_err_ekf_old[burn_in:]**2))
    nis_old = np.mean(est_ekf_old["debug"]["nis"][burn_in:])
    
    print(f"\nEKF (原始参数 R_acc={filter_cfg_old['R_acc']}):")
    print(f"  RMSE Roll:  {rmse_ekf_old_roll:.4f}°")
    print(f"  RMSE Pitch: {rmse_ekf_old_pitch:.4f}°")
    print(f"  mean_NIS:   {nis_old:.2f}")
    
    # EKF - 校准后参数
    R_calibrated = auto_calibrate_r()
    filter_cfg_new = {
        "Q_gyro": 1e-5,
        "Q_bias": 1e-10,
        "R_acc": R_calibrated,
    }
    est_ekf_new = run_ekf_fixed(ds, filter_cfg_new)
    roll_err_ekf_new = rad2deg(est_ekf_new["roll"]) - truth_rpy[:, 0]
    pitch_err_ekf_new = rad2deg(est_ekf_new["pitch"]) - truth_rpy[:, 1]
    rmse_ekf_new_roll = np.sqrt(np.mean(roll_err_ekf_new[burn_in:]**2))
    rmse_ekf_new_pitch = np.sqrt(np.mean(pitch_err_ekf_new[burn_in:]**2))
    nis_new = np.mean(est_ekf_new["debug"]["nis"][burn_in:])
    
    print(f"\nEKF (校准后 R_acc={R_calibrated:.6f}):")
    print(f"  RMSE Roll:  {rmse_ekf_new_roll:.4f}°")
    print(f"  RMSE Pitch: {rmse_ekf_new_pitch:.4f}°")
    print(f"  mean_NIS:   {nis_new:.2f}")
    
    # 绘图
    t = truth["t"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(t, roll_err_comp, 'b-', label='Comp', linewidth=0.8)
    axes[0].plot(t, roll_err_ekf_old, 'r-', label='EKF (old)', linewidth=0.8)
    axes[0].plot(t, roll_err_ekf_new, 'g-', label='EKF (calibrated)', linewidth=0.8)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0].set_ylabel('Roll Error (deg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('EKF Diagnosis - Error Comparison')
    
    axes[1].plot(t, pitch_err_comp, 'b-', label='Comp', linewidth=0.8)
    axes[1].plot(t, pitch_err_ekf_old, 'r-', label='EKF (old)', linewidth=0.8)
    axes[1].plot(t, pitch_err_ekf_new, 'g-', label='EKF (calibrated)', linewidth=0.8)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Pitch Error (deg)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, est_ekf_old["debug"]["nis"], 'r-', label='EKF (old)', linewidth=0.5)
    axes[2].plot(t, est_ekf_new["debug"]["nis"], 'g-', label='EKF (calibrated)', linewidth=0.5)
    axes[2].axhline(y=3, color='k', linestyle='--', label='Expected', linewidth=1)
    axes[2].set_ylabel('NIS')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 15])
    
    plt.tight_layout()
    Path("outputs/figures/ekf_diagnosis").mkdir(parents=True, exist_ok=True)
    plt.savefig("outputs/figures/ekf_diagnosis/comparison.png", dpi=150)
    plt.close()
    
    print(f"\n图表保存到: outputs/figures/ekf_diagnosis/comparison.png")
    
    return R_calibrated


def main():
    print("=" * 60)
    print("EKF 诊断与校准")
    print("=" * 60)
    
    # 诊断 1: 量测模型一致性
    model_ok = diagnose_measurement_model()
    
    # 诊断 2: NIS 校准
    mean_nis, R_old = diagnose_nis_calibration()
    
    # 诊断 3: 偏置估计
    diagnose_bias_estimation()
    
    # 自动校准并对比
    R_calibrated = compare_with_complementary()
    
    # 总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    
    print(f"\n1. 量测模型一致性: {'✓ 通过' if model_ok else '✗ 需要修复'}")
    print(f"2. NIS 校准: 原始 mean_NIS={mean_nis:.2f}, 目标=3.0")
    print(f"3. 推荐 R_acc = {R_calibrated:.6f}")
    
    print("\n建议更新 configs/filters/ekf_fixed.yaml:")
    print(f"  R_acc: {R_calibrated:.6f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

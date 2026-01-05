#!/usr/bin/env python
"""
EKF 参数调优

目标：
1. NIS ≈ 3（一致性）
2. RMSE ≤ 互补滤波（性能）
3. 偏置估计收敛到真值

调优策略：
1. 先用 NIS 校准 R
2. 再调 Q_bias 使偏置估计更快收敛
3. 最后微调 Q_gyro
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.truth.scenarios import generate_quasi_static, generate_swing
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.ekf_fixed import run_ekf_fixed
from src.filters.complementary import run_complementary
from src.common.math3d import rad2deg


def run_experiment(scenario_type, scenario_params, sensor_params, filter_cfg):
    """运行单次实验"""
    if scenario_type == "quasi_static":
        truth = generate_quasi_static(**scenario_params)
    else:
        truth = generate_swing(**scenario_params)
    
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params["fs"]},
    }
    
    # 真值
    if "rpy_deg" in truth:
        truth_rpy = truth["rpy_deg"]
    else:
        truth_rpy = np.column_stack([
            np.full(len(truth["t"]), scenario_params.get("roll_deg", 0)),
            np.full(len(truth["t"]), scenario_params.get("pitch_deg", 0)),
            np.full(len(truth["t"]), scenario_params.get("yaw_deg", 0))
        ])
    
    # 运行滤波器
    est_comp = run_complementary(ds, {"alpha": 0.98})
    est_ekf = run_ekf_fixed(ds, filter_cfg)
    
    # 计算指标
    burn_in = int(1.0 * scenario_params["fs"])
    
    roll_err_comp = rad2deg(est_comp["roll"]) - truth_rpy[:, 0]
    pitch_err_comp = rad2deg(est_comp["pitch"]) - truth_rpy[:, 1]
    roll_err_ekf = rad2deg(est_ekf["roll"]) - truth_rpy[:, 0]
    pitch_err_ekf = rad2deg(est_ekf["pitch"]) - truth_rpy[:, 1]
    
    rmse_comp = np.sqrt(np.mean(roll_err_comp[burn_in:]**2 + pitch_err_comp[burn_in:]**2) / 2)
    rmse_ekf = np.sqrt(np.mean(roll_err_ekf[burn_in:]**2 + pitch_err_ekf[burn_in:]**2) / 2)
    
    nis = est_ekf["debug"]["nis"]
    mean_nis = np.mean(nis[burn_in:])
    
    return {
        "rmse_comp": rmse_comp,
        "rmse_ekf": rmse_ekf,
        "mean_nis": mean_nis,
        "bias_est": est_ekf["bias_gyro"][-1],
        "truth": truth,
        "est_comp": est_comp,
        "est_ekf": est_ekf,
    }


def grid_search():
    """网格搜索最优参数"""
    print("=" * 60)
    print("EKF 参数网格搜索")
    print("=" * 60)
    
    # 准静态工况
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
    
    true_gyro_bias = np.array([0.001, 0.001, -0.002])
    
    # 参数网格
    R_acc_values = [0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.0025]
    Q_bias_values = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
    Q_gyro_values = [1e-6, 1e-5, 1e-4]
    
    best_result = None
    best_score = float('inf')
    
    results = []
    
    print("\n搜索中...")
    for Q_gyro in Q_gyro_values:
        for Q_bias in Q_bias_values:
            for R_acc in R_acc_values:
                filter_cfg = {
                    "Q_gyro": Q_gyro,
                    "Q_bias": Q_bias,
                    "R_acc": R_acc,
                }
                
                result = run_experiment("quasi_static", scenario_params, sensor_params, filter_cfg)
                
                # 评分：NIS 接近 3 + RMSE 小 + 偏置误差小
                nis_penalty = abs(result["mean_nis"] - 3.0)
                rmse_penalty = result["rmse_ekf"]
                bias_error = np.linalg.norm(result["bias_est"] - true_gyro_bias) * 1000  # mrad/s
                
                # 综合评分（越小越好）
                score = nis_penalty * 0.5 + rmse_penalty * 10 + bias_error * 0.1
                
                results.append({
                    "Q_gyro": Q_gyro,
                    "Q_bias": Q_bias,
                    "R_acc": R_acc,
                    "mean_nis": result["mean_nis"],
                    "rmse_ekf": result["rmse_ekf"],
                    "rmse_comp": result["rmse_comp"],
                    "bias_error": bias_error,
                    "score": score,
                })
                
                if score < best_score:
                    best_score = score
                    best_result = results[-1]
    
    # 排序并显示 top 10
    results.sort(key=lambda x: x["score"])
    
    print("\nTop 10 参数组合:")
    print("-" * 100)
    print(f"{'Q_gyro':>10} {'Q_bias':>10} {'R_acc':>10} {'NIS':>8} {'RMSE_EKF':>10} {'RMSE_Comp':>10} {'Bias_Err':>10} {'Score':>8}")
    print("-" * 100)
    
    for r in results[:10]:
        print(f"{r['Q_gyro']:>10.1e} {r['Q_bias']:>10.1e} {r['R_acc']:>10.4f} {r['mean_nis']:>8.2f} {r['rmse_ekf']:>10.4f}° {r['rmse_comp']:>10.4f}° {r['bias_error']:>10.3f} {r['score']:>8.3f}")
    
    print("\n最优参数:")
    print(f"  Q_gyro = {best_result['Q_gyro']:.1e}")
    print(f"  Q_bias = {best_result['Q_bias']:.1e}")
    print(f"  R_acc = {best_result['R_acc']:.6f}")
    print(f"  mean_NIS = {best_result['mean_nis']:.2f}")
    print(f"  RMSE_EKF = {best_result['rmse_ekf']:.4f}°")
    print(f"  RMSE_Comp = {best_result['rmse_comp']:.4f}°")
    
    return best_result


def validate_best_params(best_params):
    """验证最优参数在多个工况下的表现"""
    print("\n" + "=" * 60)
    print("验证最优参数")
    print("=" * 60)
    
    filter_cfg = {
        "Q_gyro": best_params["Q_gyro"],
        "Q_bias": best_params["Q_bias"],
        "R_acc": best_params["R_acc"],
    }
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    # 工况 1: 准静态
    print("\n[工况 1: 准静态]")
    qs_params = {
        "fs": 100.0, "duration_s": 30.0,
        "roll_deg": 5.0, "pitch_deg": -3.0, "yaw_deg": 0.0,
        "temp_C": 25.0, "seed": 1,
    }
    result_qs = run_experiment("quasi_static", qs_params, sensor_params, filter_cfg)
    print(f"  RMSE Comp: {result_qs['rmse_comp']:.4f}°")
    print(f"  RMSE EKF:  {result_qs['rmse_ekf']:.4f}°")
    print(f"  mean_NIS:  {result_qs['mean_nis']:.2f}")
    print(f"  EKF/Comp:  {result_qs['rmse_ekf']/result_qs['rmse_comp']:.2f}x")
    
    # 工况 2: 摆动
    print("\n[工况 2: 摆动]")
    sw_params = {
        "fs": 100.0, "duration_s": 30.0,
        "roll_amp_deg": 10.0, "pitch_amp_deg": 5.0,
        "roll_freq_hz": 0.3, "pitch_freq_hz": 0.2,
        "roll_phase_deg": 0.0, "pitch_phase_deg": 90.0,
        "yaw_deg": 0.0, "temp_C": 25.0, "seed": 1,
    }
    result_sw = run_experiment("swing", sw_params, sensor_params, filter_cfg)
    print(f"  RMSE Comp: {result_sw['rmse_comp']:.4f}°")
    print(f"  RMSE EKF:  {result_sw['rmse_ekf']:.4f}°")
    print(f"  mean_NIS:  {result_sw['mean_nis']:.2f}")
    print(f"  EKF/Comp:  {result_sw['rmse_ekf']/result_sw['rmse_comp']:.2f}x")
    
    # 绘图
    plot_validation(result_qs, result_sw, filter_cfg)
    
    return result_qs, result_sw


def plot_validation(result_qs, result_sw, filter_cfg):
    """绘制验证结果"""
    Path("outputs/figures/ekf_tuning").mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 准静态
    t_qs = result_qs["truth"]["t"]
    truth_rpy_qs = np.column_stack([np.full(len(t_qs), 5.0), np.full(len(t_qs), -3.0)])
    
    roll_err_comp_qs = rad2deg(result_qs["est_comp"]["roll"]) - truth_rpy_qs[:, 0]
    pitch_err_comp_qs = rad2deg(result_qs["est_comp"]["pitch"]) - truth_rpy_qs[:, 1]
    roll_err_ekf_qs = rad2deg(result_qs["est_ekf"]["roll"]) - truth_rpy_qs[:, 0]
    pitch_err_ekf_qs = rad2deg(result_qs["est_ekf"]["pitch"]) - truth_rpy_qs[:, 1]
    
    axes[0, 0].plot(t_qs, roll_err_comp_qs, 'b-', label='Comp', linewidth=0.8)
    axes[0, 0].plot(t_qs, roll_err_ekf_qs, 'r-', label='EKF', linewidth=0.8)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0, 0].set_ylabel('Roll Error (deg)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Quasi-Static')
    
    axes[0, 1].plot(t_qs, pitch_err_comp_qs, 'b-', label='Comp', linewidth=0.8)
    axes[0, 1].plot(t_qs, pitch_err_ekf_qs, 'r-', label='EKF', linewidth=0.8)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0, 1].set_ylabel('Pitch Error (deg)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(t_qs, result_qs["est_ekf"]["debug"]["nis"], 'b-', linewidth=0.5)
    axes[0, 2].axhline(y=3, color='r', linestyle='--', label='Expected', linewidth=1)
    axes[0, 2].set_ylabel('NIS')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 15])
    
    # 摆动
    t_sw = result_sw["truth"]["t"]
    truth_rpy_sw = result_sw["truth"]["rpy_deg"]
    
    roll_err_comp_sw = rad2deg(result_sw["est_comp"]["roll"]) - truth_rpy_sw[:, 0]
    pitch_err_comp_sw = rad2deg(result_sw["est_comp"]["pitch"]) - truth_rpy_sw[:, 1]
    roll_err_ekf_sw = rad2deg(result_sw["est_ekf"]["roll"]) - truth_rpy_sw[:, 0]
    pitch_err_ekf_sw = rad2deg(result_sw["est_ekf"]["pitch"]) - truth_rpy_sw[:, 1]
    
    axes[1, 0].plot(t_sw, roll_err_comp_sw, 'b-', label='Comp', linewidth=0.8)
    axes[1, 0].plot(t_sw, roll_err_ekf_sw, 'r-', label='EKF', linewidth=0.8)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 0].set_ylabel('Roll Error (deg)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Swing')
    
    axes[1, 1].plot(t_sw, pitch_err_comp_sw, 'b-', label='Comp', linewidth=0.8)
    axes[1, 1].plot(t_sw, pitch_err_ekf_sw, 'r-', label='EKF', linewidth=0.8)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 1].set_ylabel('Pitch Error (deg)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(t_sw, result_sw["est_ekf"]["debug"]["nis"], 'b-', linewidth=0.5)
    axes[1, 2].axhline(y=3, color='r', linestyle='--', label='Expected', linewidth=1)
    axes[1, 2].set_ylabel('NIS')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 15])
    
    plt.suptitle(f"EKF Tuned: Q_gyro={filter_cfg['Q_gyro']:.1e}, Q_bias={filter_cfg['Q_bias']:.1e}, R_acc={filter_cfg['R_acc']:.4f}")
    plt.tight_layout()
    plt.savefig("outputs/figures/ekf_tuning/validation.png", dpi=150)
    plt.close()
    
    print(f"\n图表保存到: outputs/figures/ekf_tuning/validation.png")


def main():
    # 网格搜索
    best_params = grid_search()
    
    # 验证
    result_qs, result_sw = validate_best_params(best_params)
    
    # 验收标准检查
    print("\n" + "=" * 60)
    print("验收标准检查")
    print("=" * 60)
    
    all_passed = True
    
    # 1. 不发散
    print("\n[1] 不发散检查 (RMSE < 1°):")
    qs_ok = result_qs['rmse_ekf'] < 1.0
    sw_ok = result_sw['rmse_ekf'] < 1.0
    print(f"  准静态: {'✓' if qs_ok else '✗'} RMSE={result_qs['rmse_ekf']:.4f}°")
    print(f"  摆动:   {'✓' if sw_ok else '✗'} RMSE={result_sw['rmse_ekf']:.4f}°")
    all_passed = all_passed and qs_ok and sw_ok
    
    # 2. NIS 一致性
    print("\n[2] NIS 一致性 (2.0 < mean_NIS < 4.0):")
    qs_nis_ok = 2.0 < result_qs['mean_nis'] < 4.0
    sw_nis_ok = 2.0 < result_sw['mean_nis'] < 4.0
    print(f"  准静态: {'✓' if qs_nis_ok else '✗'} NIS={result_qs['mean_nis']:.2f}")
    print(f"  摆动:   {'✓' if sw_nis_ok else '✗'} NIS={result_sw['mean_nis']:.2f}")
    all_passed = all_passed and qs_nis_ok and sw_nis_ok
    
    # 3. 性能对比
    print("\n[3] 性能对比 (EKF ≤ 2x Comp):")
    qs_perf_ok = result_qs['rmse_ekf'] <= 2 * result_qs['rmse_comp']
    sw_perf_ok = result_sw['rmse_ekf'] <= 2 * result_sw['rmse_comp']
    print(f"  准静态: {'✓' if qs_perf_ok else '✗'} {result_qs['rmse_ekf']/result_qs['rmse_comp']:.2f}x")
    print(f"  摆动:   {'✓' if sw_perf_ok else '✗'} {result_sw['rmse_ekf']/result_sw['rmse_comp']:.2f}x")
    all_passed = all_passed and qs_perf_ok and sw_perf_ok
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有验收标准通过！")
        print("\n推荐更新 configs/filters/ekf_fixed.yaml:")
        print(f"  Q_gyro: {best_params['Q_gyro']:.1e}")
        print(f"  Q_bias: {best_params['Q_bias']:.1e}")
        print(f"  R_acc: {best_params['R_acc']:.6f}")
    else:
        print("✗ 部分验收标准未通过")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

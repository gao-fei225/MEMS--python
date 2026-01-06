#!/usr/bin/env python
"""
回归测试：验证疯狗模式在各场景下的表现

测试场景：
1. accel - 加减速（主要目标）
2. vibration - 振动
3. quasi_static - 准静态
4. turn - 转弯
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from src.truth.scenarios import generate_accel, generate_vibration, generate_quasi_static, generate_turn
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.common.math3d import rad2deg


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_scenario(name, truth_func, scenario_params, filter_cfg, sensor_params):
    """运行单个场景测试"""
    print(f"\n{'='*60}")
    print(f"场景: {name}")
    print(f"{'='*60}")
    
    # 生成数据
    truth = truth_func(**scenario_params)
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params["fs"]},
    }
    
    # 运行 EKF
    result = run_ekf_adaptive(ds, filter_cfg)
    
    # 计算误差 - 处理不同的真值格式
    if "rpy_deg" in truth:
        roll_true = truth["rpy_deg"][:, 0]
        pitch_true = truth["rpy_deg"][:, 1]
    else:
        # 从四元数计算 RPY
        from src.common.math3d import quat_to_rpy
        rpy_true = np.array([quat_to_rpy(q) for q in truth["q_nb"]])
        roll_true = rad2deg(rpy_true[:, 0])
        pitch_true = rad2deg(rpy_true[:, 1])
    
    roll_est = rad2deg(result["roll"])
    pitch_est = rad2deg(result["pitch"])
    
    roll_err = roll_est - roll_true
    pitch_err = pitch_est - pitch_true
    total_err = np.sqrt(roll_err**2 + pitch_err**2)
    
    # 统计（跳过前1秒）
    burn_in = int(scenario_params["fs"])
    rmse = np.sqrt(np.mean(total_err[burn_in:]**2))
    max_err = np.max(np.abs(total_err[burn_in:]))
    lambda_mean = np.mean(result['debug']['lambda_k'][burn_in:])
    lambda_max = np.max(result['debug']['lambda_k'])
    nis_mean = np.mean(result['debug']['nis'][burn_in:])
    
    print(f"  RMSE: {rmse:.3f}°")
    print(f"  最大误差: {max_err:.3f}°")
    print(f"  λ 均值: {lambda_mean:.1f}")
    print(f"  λ 最大值: {lambda_max:.1f}")
    print(f"  NIS 均值: {nis_mean:.2f}")
    
    return {
        "name": name,
        "rmse": rmse,
        "max_err": max_err,
        "lambda_mean": lambda_mean,
        "lambda_max": lambda_max,
        "nis_mean": nis_mean,
        "truth": truth,
        "result": result,
        "ds": ds,
    }


def main():
    # 加载配置
    filter_cfg = load_yaml("configs/filters/ekf_adaptive_innovation.yaml")
    
    # 传感器配置
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    print("="*60)
    print("回归测试 - 疯狗模式验证")
    print("="*60)
    print(f"关键参数:")
    print(f"  mag_lambda_gain: {filter_cfg['adaptation']['mag_lambda_gain']}")
    print(f"  lambda_max: {filter_cfg['adaptation']['lambda_max']}")
    print(f"  mag_threshold: {filter_cfg['adaptation']['mag_threshold']}")
    print(f"  mag_sigma: {filter_cfg['dual_channel']['mag_sigma']}")
    
    results = []
    
    # 1. Accel 场景
    accel_params = {
        "fs": 100.0, "duration_s": 30.0,
        "roll_deg": 2.0, "pitch_deg": -1.0, "yaw_deg": 0.0,
        "accel_type": "step", "accel_axis": "x", "accel_peak": 2.0,
        "accel_start_s": 5.0, "accel_duration_s": 10.0,
        "temp_C": 25.0, "seed": 1,
    }
    results.append(run_scenario("Accel (step)", generate_accel, accel_params, filter_cfg, sensor_params))
    
    # 2. Vibration 场景
    vibration_params = {
        "fs": 100.0, "duration_s": 30.0,
        "roll_deg": 2.0, "pitch_deg": -1.0, "yaw_deg": 0.0,
        "vib_rms": 0.5, "vib_bandwidth_hz": 20.0, "vib_center_hz": 0.0,
        "temp_C": 25.0, "seed": 1,
    }
    results.append(run_scenario("Vibration", generate_vibration, vibration_params, filter_cfg, sensor_params))
    
    # 3. Quasi-static 场景
    static_params = {
        "fs": 100.0, "duration_s": 30.0,
        "roll_deg": 2.0, "pitch_deg": -1.0, "yaw_deg": 0.0,
        "temp_C": 25.0, "seed": 1,
    }
    results.append(run_scenario("Quasi-static", generate_quasi_static, static_params, filter_cfg, sensor_params))
    
    # 4. Turn 场景
    turn_params = {
        "fs": 100.0, "duration_s": 30.0,
        "roll_deg": 2.0, "pitch_deg": -1.0,
        "yaw_rate_dps": 30.0, "turn_radius_m": 10.0,
        "turn_start_s": 5.0, "turn_duration_s": 10.0,
        "temp_C": 25.0, "seed": 1,
    }
    results.append(run_scenario("Turn", generate_turn, turn_params, filter_cfg, sensor_params))
    
    # 汇总
    print("\n" + "="*60)
    print("汇总结果")
    print("="*60)
    print(f"{'场景':<15} {'RMSE(°)':<10} {'Max(°)':<10} {'λ均值':<12} {'NIS均值':<10} {'状态'}")
    print("-"*60)
    
    all_pass = True
    for r in results:
        # 判断标准
        if r["name"] == "Quasi-static":
            # 静态场景：λ 应该接近 1，RMSE < 1°
            status = "✓ PASS" if r["rmse"] < 1.0 and r["lambda_mean"] < 10 else "✗ FAIL"
        else:
            # 动态场景：RMSE < 5°
            status = "✓ PASS" if r["rmse"] < 5.0 else "✗ FAIL"
        
        if "FAIL" in status:
            all_pass = False
        
        print(f"{r['name']:<15} {r['rmse']:<10.3f} {r['max_err']:<10.3f} {r['lambda_mean']:<12.1f} {r['nis_mean']:<10.2f} {status}")
    
    print("="*60)
    if all_pass:
        print("✓ 所有场景测试通过！")
    else:
        print("✗ 部分场景测试失败")
    print("="*60)
    
    # 保存图片
    output_dir = Path("outputs/regression_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 4*len(results)))
    
    for i, r in enumerate(results):
        t = r["truth"]["t"]
        
        # 误差 - 处理不同的真值格式
        if "rpy_deg" in r["truth"]:
            roll_true = r["truth"]["rpy_deg"][:, 0]
            pitch_true = r["truth"]["rpy_deg"][:, 1]
        else:
            from src.common.math3d import quat_to_rpy
            rpy_true = np.array([quat_to_rpy(q) for q in r["truth"]["q_nb"]])
            roll_true = rad2deg(rpy_true[:, 0])
            pitch_true = rad2deg(rpy_true[:, 1])
        
        roll_err = rad2deg(r["result"]["roll"]) - roll_true
        pitch_err = rad2deg(r["result"]["pitch"]) - pitch_true
        total_err = np.sqrt(roll_err**2 + pitch_err**2)
        
        axes[i, 0].plot(t, total_err, 'b-', linewidth=0.5)
        axes[i, 0].axhline(5, color='r', linestyle='--', alpha=0.5)
        axes[i, 0].set_ylabel(f"{r['name']}\nError (°)")
        axes[i, 0].set_ylim([0, max(10, np.max(total_err)*1.1)])
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title("Attitude Error")
        
        # Lambda
        axes[i, 1].semilogy(t, r["result"]["debug"]["lambda_k"], 'g-', linewidth=0.5)
        axes[i, 1].axhline(1, color='gray', linestyle='--', alpha=0.5)
        axes[i, 1].set_ylabel("λ")
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title("Lambda (log scale)")
        
        # 加速度幅值
        acc_norm = np.linalg.norm(r["ds"]["meas"]["acc"], axis=1)
        axes[i, 2].plot(t, acc_norm, 'r-', linewidth=0.5)
        axes[i, 2].axhline(GRAVITY_STANDARD, color='gray', linestyle='--', alpha=0.5)
        axes[i, 2].set_ylabel("||acc|| (m/s²)")
        axes[i, 2].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 2].set_title("Accelerometer Magnitude")
    
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[-1, 2].set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "regression_summary.png", dpi=150)
    print(f"\n图片已保存到: {output_dir / 'regression_summary.png'}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
找到最优阈值，确保所有场景自适应 >= 固定
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.truth.scenarios import generate_vibration, generate_shock, generate_swing, generate_turn, generate_accel
from src.sensors.imu_model import forward_imu
from src.datasets.validate import validate_dataset
from src.filters.ekf_fixed import run_ekf_fixed
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.common.math3d import quat_to_rpy


def create_dataset(truth, sensor_params, seed=42):
    meas = forward_imu(truth, sensor_params, seed=seed)
    ds = {
        "t": truth["t"],
        "truth": {"q_nb": truth["q_nb"], "omega_b": truth["omega_b"], "a_lin_n": truth["a_lin_n"], "temp": truth["temp"]},
        "meas": {"gyro": meas["gyro"], "acc": meas["acc"]},
        "meta": {"fs": truth["fs"], "seed": seed, "scenario_name": "test", "sensor_params": sensor_params},
    }
    validate_dataset(ds)
    return ds


def get_truth_rpy(truth):
    n = len(truth["q_nb"])
    roll_true = np.zeros(n)
    pitch_true = np.zeros(n)
    for i in range(n):
        r, p, y = quat_to_rpy(truth["q_nb"][i])
        roll_true[i] = r
        pitch_true[i] = p
    return roll_true, pitch_true


def test_config(datasets, R0, nis_high):
    """测试配置在所有场景的表现"""
    results = {}
    
    for name, ds in datasets.items():
        roll_true, pitch_true = get_truth_rpy(ds["truth"])
        
        # 固定 EKF
        fixed_cfg = {"Q_gyro": 1e-5, "Q_bias": 1e-8, "R_acc": R0, "use_direction_meas": True, "nis_gating": {"enabled": False}}
        est_fixed = run_ekf_fixed(ds, fixed_cfg)
        roll_err = np.rad2deg(est_fixed["roll"] - roll_true)
        pitch_err = np.rad2deg(est_fixed["pitch"] - pitch_true)
        rmse_fixed = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
        
        # 自适应 EKF
        adapt_cfg = {
            "Q_gyro": 1e-5, "Q_bias": 1e-8, "R0": R0, "use_direction_meas": True,
            "innovation_stat": {"window_W": 30, "nis_high": nis_high, "nis_low": 3.0, "ewma_alpha": 0.05},
            "adaptation": {"lambda_max": 100.0, "lambda_min": 1.0, "use_inflate_mapping": True, "inflate_decay_rate": 0.9},
            "dual_channel": {"enabled": False},
        }
        est_adapt = run_ekf_adaptive(ds, adapt_cfg)
        roll_err = np.rad2deg(est_adapt["roll"] - roll_true)
        pitch_err = np.rad2deg(est_adapt["pitch"] - pitch_true)
        rmse_adapt = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
        
        nis_mean = np.mean(est_adapt["debug"]["nis"][100:])
        nis_raw_mean = np.mean(est_adapt["debug"]["nis_raw"][100:])
        improvement = (rmse_fixed - rmse_adapt) / rmse_fixed * 100
        
        results[name] = {
            "rmse_fixed": rmse_fixed,
            "rmse_adapt": rmse_adapt,
            "nis_mean": nis_mean,
            "nis_raw_mean": nis_raw_mean,
            "improvement": improvement,
        }
    
    return results


def main():
    print("=" * 70)
    print("寻找最优阈值")
    print("=" * 70)
    
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    # 生成所有数据集
    print("\n生成数据集...")
    datasets = {}
    
    truth = generate_vibration(fs=100, duration_s=30, roll_deg=0, pitch_deg=0, yaw_deg=0,
                               vib_rms=0.5, vib_bandwidth_hz=10.0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["vibration"] = create_dataset(truth, sensor_params)
    
    truth = generate_shock(fs=100, duration_s=20, roll_deg=0, pitch_deg=0, yaw_deg=0,
                          shock_peak=50.0, shock_width_s=0.05, shock_times=[5.0, 10.0, 15.0], temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["shock"] = create_dataset(truth, sensor_params)
    
    truth = generate_swing(fs=100, duration_s=30, roll_amp_deg=15.0, pitch_amp_deg=10.0,
                          roll_freq_hz=0.5, pitch_freq_hz=0.3, roll_phase_deg=0, pitch_phase_deg=90,
                          yaw_deg=0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["swing"] = create_dataset(truth, sensor_params)
    
    truth = generate_turn(fs=100, duration_s=40, roll_deg=0, pitch_deg=0,
                         yaw_rate_dps=30.0, turn_radius_m=10.0, turn_start_s=5.0, turn_duration_s=30.0,
                         temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["turn"] = create_dataset(truth, sensor_params)
    
    truth = generate_accel(fs=100, duration_s=30, roll_deg=0, pitch_deg=0, yaw_deg=0,
                          accel_type="ramp", accel_axis="x", accel_peak=5.0, accel_start_s=5.0,
                          accel_duration_s=20.0, temp_C=25, seed=42)
    truth["fs"] = 100.0
    datasets["accel"] = create_dataset(truth, sensor_params)
    
    # 测试不同 R0 和 nis_high 组合
    R0_values = [1e-6, 2e-6, 3.5e-6]
    nis_high_values = [5.0, 7.815, 10.0, 15.0, 20.0, 30.0, 50.0]
    
    print("\n搜索最优配置...")
    print(f"\n{'R0':<10} | {'nis_high':<10} | {'vib改善':<10} | {'accel改善':<10} | {'全通过':<8}")
    print(f"{'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")
    
    best_config = None
    best_vib_improvement = -float('inf')
    
    for R0 in R0_values:
        for nis_high in nis_high_values:
            results = test_config(datasets, R0, nis_high)
            
            vib_imp = results["vibration"]["improvement"]
            accel_imp = results["accel"]["improvement"]
            all_pass = all(r["improvement"] >= -0.05 for r in results.values())  # 允许 0.05% 误差
            
            status = "✓" if all_pass else "✗"
            print(f"{R0:<10.2e} | {nis_high:<10.1f} | {vib_imp:+9.1f}% | {accel_imp:+9.1f}% | {status:<8}")
            
            if all_pass and vib_imp > best_vib_improvement:
                best_vib_improvement = vib_imp
                best_config = (R0, nis_high, results)
    
    if best_config:
        R0, nis_high, results = best_config
        print(f"\n最优配置: R0={R0:.2e}, nis_high={nis_high}")
        print(f"\n详细结果:")
        print(f"{'场景':<12} | {'固定 RMSE':<12} | {'自适应 RMSE':<14} | {'NIS mean':<10} | {'改善':<10}")
        print(f"{'-'*12} | {'-'*12} | {'-'*14} | {'-'*10} | {'-'*10}")
        for name, r in results.items():
            print(f"{name:<12} | {r['rmse_fixed']:<12.3f} | {r['rmse_adapt']:<14.3f} | {r['nis_mean']:<10.2f} | {r['improvement']:+.1f}%")
    else:
        print("\n未找到满足所有场景的配置，尝试更高的阈值...")
        
        # 尝试更高的阈值
        for nis_high in [100.0, 200.0, 500.0]:
            R0 = 2e-6
            results = test_config(datasets, R0, nis_high)
            
            vib_imp = results["vibration"]["improvement"]
            accel_imp = results["accel"]["improvement"]
            all_pass = all(r["improvement"] >= -0.05 for r in results.values())
            
            status = "✓" if all_pass else "✗"
            print(f"{R0:<10.2e} | {nis_high:<10.1f} | {vib_imp:+9.1f}% | {accel_imp:+9.1f}% | {status:<8}")
            
            if all_pass:
                print(f"\n找到配置: R0={R0:.2e}, nis_high={nis_high}")
                print(f"\n详细结果:")
                for name, r in results.items():
                    print(f"  {name}: 固定={r['rmse_fixed']:.3f}°, 自适应={r['rmse_adapt']:.3f}°, 改善={r['improvement']:+.1f}%")
                break
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

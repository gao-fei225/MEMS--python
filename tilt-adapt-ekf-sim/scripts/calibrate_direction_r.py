#!/usr/bin/env python
"""
校准方向量测的 R 参数

目标：使 mean_NIS ≈ 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.truth.scenarios import generate_quasi_static
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.ekf_fixed import run_ekf_fixed


def calibrate_r():
    """校准 R 使 NIS ≈ 3"""
    print("=" * 60)
    print("方向量测 R 校准")
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
    
    truth = generate_quasi_static(**scenario_params)
    meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
    
    ds = {
        "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
        "meta": {"fs": scenario_params["fs"]},
    }
    
    burn_in = int(2.0 * scenario_params["fs"])
    
    # 搜索 R
    R_values = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    
    print("\n搜索最优 R:")
    print("-" * 60)
    print(f"{'R_acc':>12} {'mean_NIS':>10} {'p(>7.815)':>10}")
    print("-" * 60)
    
    best_r = None
    best_diff = float('inf')
    
    for R_acc in R_values:
        filter_cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R_acc": R_acc,
            "use_direction_meas": True,
            "nis_gating": {"enabled": False},
        }
        
        est = run_ekf_fixed(ds, filter_cfg)
        nis = est["debug"]["nis"][burn_in:]
        mean_nis = np.mean(nis)
        p_exceed = np.mean(nis > 7.815) * 100
        
        print(f"{R_acc:>12.1e} {mean_nis:>10.2f} {p_exceed:>10.1f}%")
        
        diff = abs(mean_nis - 3.0)
        if diff < best_diff:
            best_diff = diff
            best_r = R_acc
    
    print("-" * 60)
    print(f"\n最优 R_acc = {best_r:.1e}")
    
    # 精细搜索
    print("\n精细搜索:")
    R_fine = np.linspace(best_r * 0.3, best_r * 3, 10)
    
    for R_acc in R_fine:
        filter_cfg = {
            "Q_gyro": 1e-5,
            "Q_bias": 1e-8,
            "R_acc": R_acc,
            "use_direction_meas": True,
            "nis_gating": {"enabled": False},
        }
        
        est = run_ekf_fixed(ds, filter_cfg)
        nis = est["debug"]["nis"][burn_in:]
        mean_nis = np.mean(nis)
        
        diff = abs(mean_nis - 3.0)
        if diff < best_diff:
            best_diff = diff
            best_r = R_acc
        
        print(f"  R_acc={R_acc:.2e}, mean_NIS={mean_nis:.2f}")
    
    print(f"\n最终推荐 R_acc = {best_r:.2e}")
    
    return best_r


if __name__ == "__main__":
    calibrate_r()

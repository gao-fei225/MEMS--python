#!/usr/bin/env python
"""
差分进化 (Differential Evolution) 自动调参脚本

用于寻找 Adaptive EKF 的全局最优参数组合
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.optimize import differential_evolution
import time

from src.truth.scenarios import generate_accel, generate_vibration, generate_quasi_static, generate_turn
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.filters.ekf_adaptive import run_ekf_adaptive
from src.common.math3d import rad2deg, quat_to_rpy


# ========== 数据生成（只执行一次） ==========
print("Generating simulation data...")

SCENARIOS = {
    "Accel": {
        "func": generate_accel,
        "params": {
            "fs": 100.0, "duration_s": 30.0,
            "roll_deg": 2.0, "pitch_deg": -1.0, "yaw_deg": 0.0,
            "accel_type": "step", "accel_axis": "x", "accel_peak": 2.0,
            "accel_start_s": 5.0, "accel_duration_s": 10.0,
            "temp_C": 25.0, "seed": 1,
        },
    },
    "Vibration": {
        "func": generate_vibration,
        "params": {
            "fs": 100.0, "duration_s": 30.0,
            "roll_deg": 2.0, "pitch_deg": -1.0, "yaw_deg": 0.0,
            "vib_rms": 0.5, "vib_bandwidth_hz": 20.0, "vib_center_hz": 0.0,
            "temp_C": 25.0, "seed": 1,
        },
    },
    "Static": {
        "func": generate_quasi_static,
        "params": {
            "fs": 100.0, "duration_s": 30.0,
            "roll_deg": 2.0, "pitch_deg": -1.0, "yaw_deg": 0.0,
            "temp_C": 25.0, "seed": 1,
        },
    },
    "Turn": {
        "func": generate_turn,
        "params": {
            "fs": 100.0, "duration_s": 30.0,
            "roll_deg": 2.0, "pitch_deg": -1.0,
            "yaw_rate_dps": 30.0, "turn_radius_m": 10.0,
            "turn_start_s": 5.0, "turn_duration_s": 10.0,
            "temp_C": 25.0, "seed": 1,
        },
    },
}

SENSOR_PARAMS = {
    "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
    "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
}

# 预生成所有数据
DATA_CACHE = {}
for name, cfg in SCENARIOS.items():
    truth = cfg["func"](**cfg["params"])
    meas = forward_imu(truth, SENSOR_PARAMS, seed=1, g=GRAVITY_STANDARD)
    ds = {"meas": {"acc": meas["acc"], "gyro": meas["gyro"]}, "meta": {"fs": cfg["params"]["fs"]}}
    
    # 提取真值
    if "rpy_deg" in truth:
        roll_true, pitch_true = truth["rpy_deg"][:, 0], truth["rpy_deg"][:, 1]
    else:
        n = len(truth["q_nb"])
        roll_true = np.zeros(n)
        pitch_true = np.zeros(n)
        for i in range(n):
            r, p, y = quat_to_rpy(truth["q_nb"][i])
            roll_true[i] = np.rad2deg(r)
            pitch_true[i] = np.rad2deg(p)
    
    DATA_CACHE[name] = {"ds": ds, "roll_true": roll_true, "pitch_true": pitch_true}

print(f"Data ready: {list(DATA_CACHE.keys())}")


def compute_rmse(est, roll_true, pitch_true, burn_in=100):
    """计算 RMSE"""
    roll_est = rad2deg(est["roll"])
    pitch_est = rad2deg(est["pitch"])
    roll_err = roll_est[burn_in:] - roll_true[burn_in:]
    pitch_err = pitch_est[burn_in:] - pitch_true[burn_in:]
    return np.sqrt(np.mean(roll_err**2 + pitch_err**2))


def objective(params):
    """
    目标函数：输入参数，返回平均 RMSE
    
    params = [R0_log, th_nis, th_vib, lambda_vib, mag_threshold, mag_gain_log]
    """
    # 解包参数
    R0_val = 10 ** params[0]
    th_nis_val = params[1]
    th_vib_val = params[2]
    lambda_vib_val = params[3]
    mag_threshold_val = params[4]
    mag_gain_log = params[5]
    mag_gain_val = 10 ** mag_gain_log
    
    # 构建配置
    cfg = {
        "Q_gyro": 1e-5, "Q_bias": 1e-8, "R0": R0_val,
        "use_direction_meas": True,
        "innovation_stat": {"window_W": 50, "nis_high": th_nis_val, "nis_low": 2.0, "ewma_alpha": 0.15},
        "dual_channel": {"enabled": True, "mag_weight": 50.0, "mag_sigma": 0.05, "combine_mode": "max"},
        "adaptation": {
            "r_up": 2.0, "r_down": 0.95, "lambda_max": 100000.0, "lambda_min": 1.0,
            "use_inflate_mapping": True, "inflate_decay_rate": 0.92, "inflate_rise_smooth": 1.0,
            "use_dynamic_aware": True, 
            "mag_threshold": mag_threshold_val, 
            "mag_lambda_gain": mag_gain_val,
            "gyro_threshold": 0.05, "dynamic_alpha": 0.3, "acc_vec_alpha": 0.1, "vib_threshold": 0.05,
            "acc_window_size": 20, "vib_var_threshold": th_vib_val, "maneuver_mean_threshold": 0.4,
            "lambda_vibration": lambda_vib_val,
        },
        "zaru": {
            "enabled": True, "acc_std_threshold": 0.01, "gyro_threshold": 0.02,
            "r_scale": 0.01, "q_att_scale": 0.001, "confirm_count": 10,
        },
        "lpf": {"enabled": False},  # 优化时关闭 LPF 加速
    }
    
    # 跑四个场景
    rmses = []
    weights = {"Accel": 1.0, "Vibration": 1.0, "Static": 0.5, "Turn": 1.0}  # 可调权重
    
    for name, data in DATA_CACHE.items():
        try:
            est = run_ekf_adaptive(data["ds"], cfg)
            rmse = compute_rmse(est, data["roll_true"], data["pitch_true"])
            rmses.append(rmse * weights[name])
        except Exception as e:
            # 参数导致数值问题，返回大惩罚
            return 100.0
    
    avg_rmse = np.sum(rmses) / np.sum(list(weights.values()))
    return avg_rmse


def optimize():
    """运行差分进化优化"""
    # 参数边界
    # [R0_log, th_nis, th_vib, lambda_vib, mag_threshold, mag_gain_log]
    bounds = [
        (-5, -3),      # R0: 1e-5 ~ 1e-3
        (4.0, 12.0),   # th_nis: 4 ~ 12
        (0.02, 0.15),  # th_vib: 0.02 ~ 0.15
        (50.0, 200.0), # lambda_vib: 50 ~ 200
        (0.05, 0.3),   # mag_threshold: 0.05 ~ 0.3
        (2.0, 5.0),    # mag_gain: 100 ~ 100000 (log)
    ]
    
    print("\n" + "="*60)
    print("Starting Differential Evolution Optimization")
    print("="*60)
    print(f"Parameters: R0, th_nis, th_vib, lambda_vib, mag_threshold, mag_gain")
    print(f"Bounds: {bounds}")
    print()
    
    start_time = time.time()
    
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=15,      # 减少迭代次数加速测试
        popsize=6,       # 减少种群大小
        tol=0.005,       # 收敛容差
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True,       # 显示进度
        workers=1,       # 单线程（避免多进程问题）
        updating='deferred',
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Best Average RMSE: {result.fun:.4f} degrees")
    print("-"*40)
    print("Optimal Parameters:")
    print(f"  R0:            {10**result.x[0]:.6f} (log: {result.x[0]:.2f})")
    print(f"  th_nis:        {result.x[1]:.2f}")
    print(f"  th_vib:        {result.x[2]:.4f}")
    print(f"  lambda_vib:    {result.x[3]:.1f}")
    print(f"  mag_threshold: {result.x[4]:.3f}")
    print(f"  mag_gain:      {10**result.x[5]:.1f} (log: {result.x[5]:.2f})")
    print("="*60)
    
    # 用最优参数跑一次验证
    print("\nValidating with optimal parameters...")
    final_rmse = objective(result.x)
    print(f"Validation RMSE: {final_rmse:.4f} degrees")
    
    return result


if __name__ == "__main__":
    optimize()

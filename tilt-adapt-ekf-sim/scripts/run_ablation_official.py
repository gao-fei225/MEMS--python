#!/usr/bin/env python
"""
运行官方消融实验模块
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.truth.scenarios import generate_accel, generate_vibration, generate_quasi_static, generate_turn
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.experiments.ablation import run_full_ablation, print_ablation_summary


def main():
    sensor_params = {
        "acc": {"bias0": [0.02, -0.01, 0.03], "sigma_white": 0.02},
        "gyro": {"bias0": [0.001, 0.001, -0.002], "sigma_white": 0.001},
    }
    
    scenarios = {
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
    
    print("="*70)
    print("官方消融实验")
    print("="*70)
    
    for scenario_name, scenario_cfg in scenarios.items():
        print(f"\n场景: {scenario_name}")
        
        truth = scenario_cfg["func"](**scenario_cfg["params"])
        meas = forward_imu(truth, sensor_params, seed=1, g=GRAVITY_STANDARD)
        
        ds = {
            "meas": {"acc": meas["acc"], "gyro": meas["gyro"]},
            "meta": {"fs": scenario_cfg["params"]["fs"]},
            "truth": truth,
        }
        
        results = run_full_ablation(ds, scenario_name)
        print_ablation_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

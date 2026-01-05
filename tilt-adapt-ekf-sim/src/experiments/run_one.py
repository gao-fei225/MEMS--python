"""
单次实验编排模块 (Step 8)

固定流程：
1. 读取 scenario config → truth = generate_*()
2. 读取 sensor config → meas = forward_imu(truth, sensor_params)
3. 组装 dataset: ds = {t, truth, meas, meta}
4. validate_dataset(ds)
5. 保存 dataset: save_npz(data/generated/...)
6. 跑 filter: 互补滤波或后续 EKF
7. 算 metrics → 保存 tables
8. 出图 → 保存 figures

使用方式：
    python -m src.experiments.run_one \\
        --scenario configs/scenarios/quasi_static.yaml \\
        --sensor configs/sensors/imu_nominal.yaml \\
        --filter configs/filters/complementary.yaml \\
        --global configs/global.yaml
"""

import sys
from pathlib import Path
import argparse
import yaml
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 设置 matplotlib 后端
import matplotlib
matplotlib.use('Agg')

from src.truth.scenarios import (
    generate_quasi_static, generate_swing, generate_accel,
    generate_turn, generate_vibration, generate_shock
)
from src.truth.frames import GRAVITY_STANDARD
from src.sensors.imu_model import forward_imu
from src.datasets.validate import validate_dataset
from src.datasets.serialize import save_npz, load_npz
from src.filters.complementary import run_complementary
from src.filters.ekf_fixed import run_ekf_fixed
from src.metrics.tilt_error import compute_tilt_metrics, print_tilt_metrics
from src.viz.plot_timeseries import plot_attitude_comparison, plot_attitude_error


# ============================================================
# 配置加载
# ============================================================

def load_yaml(path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_global_config(path: str) -> Dict[str, Any]:
    """加载全局配置"""
    cfg = load_yaml(path)
    return {
        "gravity": cfg.get("gravity", GRAVITY_STANDARD),
        "data_dir": cfg.get("data_dir", "data/generated"),
        "output_dir": cfg.get("output_dir", "outputs"),
        "burn_in_s": cfg.get("defaults", {}).get("burn_in_s", 0.5),
    }


def load_scenario_config(path: str) -> Dict[str, Any]:
    """加载工况配置"""
    return load_yaml(path)


def load_sensor_config(path: str) -> Dict[str, Any]:
    """加载传感器配置"""
    return load_yaml(path)


def load_filter_config(path: str) -> Dict[str, Any]:
    """加载滤波器配置"""
    cfg = load_yaml(path)
    # 兼容两种格式
    if "parameters" in cfg:
        return cfg["parameters"]
    return cfg


# ============================================================
# 真值生成
# ============================================================

def generate_truth(scenario_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据工况配置生成真值
    
    Args:
        scenario_cfg: 工况配置
    
    Returns:
        truth dict
    """
    scenario_name = scenario_cfg.get("name", "quasi_static")
    
    if scenario_name == "quasi_static":
        return generate_quasi_static(
            fs=scenario_cfg.get("fs", 100.0),
            duration_s=scenario_cfg.get("duration_s", 30.0),
            roll_deg=scenario_cfg.get("roll_deg", 0.0),
            pitch_deg=scenario_cfg.get("pitch_deg", 0.0),
            yaw_deg=scenario_cfg.get("yaw_deg", 0.0),
            temp_C=scenario_cfg.get("temp_C", 25.0),
            seed=scenario_cfg.get("seed", 42),
        )
    elif scenario_name == "swing":
        return generate_swing(
            fs=scenario_cfg.get("fs", 100.0),
            duration_s=scenario_cfg.get("duration_s", 30.0),
            roll_amp_deg=scenario_cfg.get("roll_amp_deg", 10.0),
            pitch_amp_deg=scenario_cfg.get("pitch_amp_deg", 5.0),
            roll_freq_hz=scenario_cfg.get("roll_freq_hz", 0.2),
            pitch_freq_hz=scenario_cfg.get("pitch_freq_hz", 0.15),
            roll_phase_deg=scenario_cfg.get("roll_phase_deg", 0.0),
            pitch_phase_deg=scenario_cfg.get("pitch_phase_deg", 90.0),
            yaw_deg=scenario_cfg.get("yaw_deg", 0.0),
            temp_C=scenario_cfg.get("temp_C", 25.0),
            seed=scenario_cfg.get("seed", 42),
        )
    elif scenario_name == "accel":
        return generate_accel(
            fs=scenario_cfg.get("fs", 100.0),
            duration_s=scenario_cfg.get("duration_s", 30.0),
            roll_deg=scenario_cfg.get("roll_deg", 0.0),
            pitch_deg=scenario_cfg.get("pitch_deg", 0.0),
            yaw_deg=scenario_cfg.get("yaw_deg", 0.0),
            accel_type=scenario_cfg.get("accel_type", "step"),
            accel_axis=scenario_cfg.get("accel_axis", "x"),
            accel_peak=scenario_cfg.get("accel_peak", 2.0),
            accel_freq_hz=scenario_cfg.get("accel_freq_hz", 0.1),
            accel_start_s=scenario_cfg.get("accel_start_s", 5.0),
            accel_duration_s=scenario_cfg.get("accel_duration_s", 10.0),
            temp_C=scenario_cfg.get("temp_C", 25.0),
            seed=scenario_cfg.get("seed", 42),
        )
    elif scenario_name == "turn":
        return generate_turn(
            fs=scenario_cfg.get("fs", 100.0),
            duration_s=scenario_cfg.get("duration_s", 30.0),
            roll_deg=scenario_cfg.get("roll_deg", 0.0),
            pitch_deg=scenario_cfg.get("pitch_deg", 0.0),
            yaw_rate_dps=scenario_cfg.get("yaw_rate_dps", 30.0),
            turn_radius_m=scenario_cfg.get("turn_radius_m", 10.0),
            turn_start_s=scenario_cfg.get("turn_start_s", 5.0),
            turn_duration_s=scenario_cfg.get("turn_duration_s", 20.0),
            temp_C=scenario_cfg.get("temp_C", 25.0),
            seed=scenario_cfg.get("seed", 42),
        )
    elif scenario_name == "vibration":
        return generate_vibration(
            fs=scenario_cfg.get("fs", 100.0),
            duration_s=scenario_cfg.get("duration_s", 30.0),
            roll_deg=scenario_cfg.get("roll_deg", 0.0),
            pitch_deg=scenario_cfg.get("pitch_deg", 0.0),
            yaw_deg=scenario_cfg.get("yaw_deg", 0.0),
            vib_rms=scenario_cfg.get("vib_rms", 0.5),
            vib_bandwidth_hz=scenario_cfg.get("vib_bandwidth_hz", 10.0),
            vib_center_hz=scenario_cfg.get("vib_center_hz", 0.0),
            temp_C=scenario_cfg.get("temp_C", 25.0),
            seed=scenario_cfg.get("seed", 42),
        )
    elif scenario_name == "shock":
        return generate_shock(
            fs=scenario_cfg.get("fs", 100.0),
            duration_s=scenario_cfg.get("duration_s", 20.0),
            roll_deg=scenario_cfg.get("roll_deg", 0.0),
            pitch_deg=scenario_cfg.get("pitch_deg", 0.0),
            yaw_deg=scenario_cfg.get("yaw_deg", 0.0),
            shock_peak=scenario_cfg.get("shock_peak", 50.0),
            shock_width_s=scenario_cfg.get("shock_width_s", 0.05),
            shock_times=scenario_cfg.get("shock_times", [5.0, 10.0, 15.0]),
            shock_axis=scenario_cfg.get("shock_axis", "z"),
            temp_C=scenario_cfg.get("temp_C", 25.0),
            seed=scenario_cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"未知工况类型: {scenario_name}")


# ============================================================
# 数据集组装
# ============================================================

def assemble_dataset(
    truth: Dict[str, Any],
    meas: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    sensor_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    组装数据集
    
    Args:
        truth: 真值数据
        meas: 测量数据
        scenario_cfg: 工况配置
        sensor_cfg: 传感器配置
    
    Returns:
        dataset dict
    """
    n_samples = len(truth["t"])
    
    ds = {
        "t": truth["t"],
        "truth": {
            "q_nb": truth["q_nb"],
            "omega_b": truth["omega_b"],
            "a_lin_n": truth["a_lin_n"],
            "temp": truth["temp"],
        },
        "meas": {
            "gyro": meas["gyro"],
            "acc": meas["acc"],
        },
        "meta": {
            "fs": float(truth.get("fs", scenario_cfg.get("fs", 100.0))),
            "seed": int(scenario_cfg.get("seed", 42)),
            "scenario_name": str(scenario_cfg.get("name", "unknown")),
            "sensor_params": sensor_cfg,
        },
    }
    
    return ds


# ============================================================
# 滤波器运行
# ============================================================

def run_filter(
    ds: Dict[str, Any],
    filter_cfg: Dict[str, Any],
    filter_type: str = "complementary",
    static_calibration: bool = False,
) -> Dict[str, Any]:
    """
    运行滤波器
    
    Args:
        ds: 数据集
        filter_cfg: 滤波器配置
        filter_type: 滤波器类型 ("complementary" | "ekf_fixed" | "ekf_adaptive")
        static_calibration: 是否启用静止段校准（仅互补滤波）
    
    Returns:
        估计结果 dict
    """
    if filter_type == "complementary":
        if static_calibration:
            from src.filters.complementary import run_complementary_with_static_calibration
            return run_complementary_with_static_calibration(ds, filter_cfg)
        else:
            return run_complementary(ds, filter_cfg)
    elif filter_type == "ekf_fixed":
        return run_ekf_fixed(ds, filter_cfg)
    elif filter_type == "ekf_adaptive":
        from src.filters.ekf_adaptive import run_ekf_adaptive
        return run_ekf_adaptive(ds, filter_cfg)
    else:
        raise ValueError(f"未知滤波器类型: {filter_type}")


# ============================================================
# 结果保存
# ============================================================

def save_metrics(
    metrics: Dict[str, float],
    output_path: str,
    experiment_info: Dict[str, Any]
) -> None:
    """
    保存指标到 JSON 文件
    
    Args:
        metrics: 指标字典
        output_path: 输出路径
        experiment_info: 实验信息
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "metrics": metrics,
        "experiment": experiment_info,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


# ============================================================
# 主流程
# ============================================================

def run_one(
    scenario_path: str,
    sensor_path: str,
    filter_path: str,
    global_path: str = "configs/global.yaml",
    experiment_name: Optional[str] = None,
    save_dataset: bool = True,
    save_figures: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    运行单次实验
    
    固定流程：
    1. 读取配置
    2. 生成真值
    3. 生成测量
    4. 组装数据集
    5. 验证数据集
    6. 保存数据集
    7. 运行滤波器
    8. 计算指标
    9. 保存结果
    10. 生成图表
    
    Args:
        scenario_path: 工况配置路径
        sensor_path: 传感器配置路径
        filter_path: 滤波器配置路径
        global_path: 全局配置路径
        experiment_name: 实验名称（可选，默认自动生成）
        save_dataset: 是否保存数据集
        save_figures: 是否保存图表
        verbose: 是否打印详细信息
    
    Returns:
        结果字典
    """
    # ========== Step 1: 读取配置 ==========
    if verbose:
        print("=" * 60)
        print("Step 8: 单次实验编排 (run_one)")
        print("=" * 60)
        print(f"\n[1/8] 读取配置...")
    
    global_cfg = load_global_config(global_path)
    scenario_cfg = load_scenario_config(scenario_path)
    sensor_cfg = load_sensor_config(sensor_path)
    filter_cfg = load_filter_config(filter_path)
    
    # 生成实验名称
    if experiment_name is None:
        scenario_name = scenario_cfg.get("name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{scenario_name}_{timestamp}"
    
    if verbose:
        print(f"  工况: {scenario_cfg.get('name', 'unknown')}")
        print(f"  传感器: {Path(sensor_path).stem}")
        print(f"  滤波器: {Path(filter_path).stem}")
        print(f"  实验名称: {experiment_name}")
    
    # ========== Step 2: 生成真值 ==========
    if verbose:
        print(f"\n[2/8] 生成真值...")
    
    truth = generate_truth(scenario_cfg)
    
    if verbose:
        print(f"  样本数: {len(truth['t'])}")
        print(f"  时长: {truth['t'][-1]:.1f}s")
        print(f"  采样率: {truth.get('fs', scenario_cfg.get('fs', 100.0))}Hz")
    
    # ========== Step 3: 生成测量 ==========
    if verbose:
        print(f"\n[3/8] 生成测量...")
    
    g = global_cfg.get("gravity", GRAVITY_STANDARD)
    seed = scenario_cfg.get("seed", 42)
    meas = forward_imu(truth, sensor_cfg, seed=seed, g=g)
    
    if verbose:
        print(f"  acc bias: {sensor_cfg.get('acc', {}).get('bias0', [0,0,0])}")
        print(f"  gyro bias: {sensor_cfg.get('gyro', {}).get('bias0', [0,0,0])}")
    
    # ========== Step 4: 组装数据集 ==========
    if verbose:
        print(f"\n[4/8] 组装数据集...")
    
    ds = assemble_dataset(truth, meas, scenario_cfg, sensor_cfg)
    
    # ========== Step 5: 验证数据集 ==========
    if verbose:
        print(f"\n[5/8] 验证数据集...")
    
    validate_dataset(ds)
    
    if verbose:
        print("  ✓ 数据集验证通过")
    
    # ========== Step 6: 保存数据集 ==========
    data_dir = global_cfg.get("data_dir", "data/generated")
    dataset_path = Path(data_dir) / f"{experiment_name}.npz"
    
    if save_dataset:
        if verbose:
            print(f"\n[6/8] 保存数据集...")
        
        save_npz(str(dataset_path), ds)
        
        if verbose:
            print(f"  保存到: {dataset_path}")
    else:
        if verbose:
            print(f"\n[6/8] 跳过数据集保存")
    
    # ========== Step 7: 运行滤波器 ==========
    if verbose:
        print(f"\n[7/8] 运行滤波器...")
    
    # 从配置文件检测滤波器类型
    filter_type = filter_cfg.get("type", filter_cfg.get("filter", {}).get("type", "complementary"))
    if "ekf" in Path(filter_path).stem.lower():
        filter_type = "ekf_fixed"
    
    est = run_filter(ds, filter_cfg, filter_type)
    
    if verbose:
        print(f"  滤波器: {filter_type}")
        if filter_type == "complementary":
            alpha = filter_cfg.get("alpha", 0.98)
            print(f"  alpha: {alpha}")
        elif filter_type == "ekf_fixed":
            Q_gyro = filter_cfg.get("Q_gyro", 1e-6)
            R_acc = filter_cfg.get("R_acc", 0.01)
            print(f"  Q_gyro: {Q_gyro:.2e}")
            print(f"  R_acc: {R_acc:.2e}")
    
    # ========== Step 8: 计算指标 ==========
    if verbose:
        print(f"\n[8/8] 计算指标...")
    
    burn_in_s = global_cfg.get("burn_in_s", 0.5)
    fs = ds["meta"]["fs"]
    
    # 准备真值格式（compute_tilt_metrics 需要 rpy_deg 或 roll/pitch）
    # 如果 truth 中有 rpy_deg，直接使用；否则从四元数计算
    if "rpy_deg" not in truth:
        from src.common.math3d import quat_to_rpy, rad2deg
        n_samples = len(truth["t"])
        rpy_deg = np.zeros((n_samples, 3), dtype=np.float64)
        for i in range(n_samples):
            roll, pitch, yaw = quat_to_rpy(truth["q_nb"][i])
            rpy_deg[i] = [rad2deg(roll), rad2deg(pitch), rad2deg(yaw)]
        truth_for_metrics = {"rpy_deg": rpy_deg}
    else:
        truth_for_metrics = {"rpy_deg": truth["rpy_deg"]}
    
    metrics = compute_tilt_metrics(
        truth=truth_for_metrics,
        est=est,
        burn_in_s=burn_in_s,
        fs=fs,
        wrap_error=True,
    )
    
    if verbose:
        print_tilt_metrics(metrics, name=experiment_name, fs=fs)
    
    # ========== 保存指标 ==========
    output_dir = global_cfg.get("output_dir", "outputs")
    tables_dir = Path(output_dir) / "tables"
    metrics_path = tables_dir / f"{experiment_name}_metrics.json"
    
    experiment_info = {
        "name": experiment_name,
        "scenario": scenario_cfg.get("name", "unknown"),
        "sensor": Path(sensor_path).stem,
        "filter": filter_type,
        "filter_params": filter_cfg,
        "burn_in_s": burn_in_s,
        "fs": fs,
        "duration_s": float(truth["t"][-1]),
        "n_samples": len(truth["t"]),
    }
    
    save_metrics(metrics, str(metrics_path), experiment_info)
    
    if verbose:
        print(f"\n  指标保存到: {metrics_path}")
    
    # ========== 生成图表 ==========
    if save_figures:
        figures_dir = Path(output_dir) / "figures" / experiment_name
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 姿态对比图
        fig1_path = figures_dir / "attitude_comparison.png"
        plot_attitude_comparison(
            t=truth["t"],
            truth=truth_for_metrics,
            est=est,
            save_path=str(fig1_path),
            title=f"{experiment_name} - Attitude",
            show=False,
        )
        
        # 误差图
        fig2_path = figures_dir / "attitude_error.png"
        plot_attitude_error(
            t=truth["t"],
            truth=truth_for_metrics,
            est=est,
            save_path=str(fig2_path),
            title=f"{experiment_name} - Error",
            show=False,
        )
        
        if verbose:
            print(f"  图表保存到: {figures_dir}/")
    
    # ========== 返回结果 ==========
    if verbose:
        print("\n" + "=" * 60)
        print("实验完成！")
        print("=" * 60)
    
    return {
        "experiment_name": experiment_name,
        "dataset_path": str(dataset_path) if save_dataset else None,
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "truth": truth,
        "meas": meas,
        "est": est,
        "ds": ds,
    }


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="单次实验编排 (Step 8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="configs/scenarios/quasi_static.yaml",
        help="工况配置文件路径",
    )
    parser.add_argument(
        "--sensor", "-n",
        type=str,
        default="configs/sensors/imu_nominal.yaml",
        help="传感器配置文件路径",
    )
    parser.add_argument(
        "--filter", "-f",
        type=str,
        default="configs/filters/complementary.yaml",
        help="滤波器配置文件路径",
    )
    parser.add_argument(
        "--global", "-g",
        dest="global_cfg",
        type=str,
        default="configs/global.yaml",
        help="全局配置文件路径",
    )
    parser.add_argument(
        "--name", "-N",
        type=str,
        default=None,
        help="实验名称（可选）",
    )
    parser.add_argument(
        "--no-save-dataset",
        action="store_true",
        help="不保存数据集",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="不生成图表",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式",
    )
    
    args = parser.parse_args()
    
    result = run_one(
        scenario_path=args.scenario,
        sensor_path=args.sensor,
        filter_path=args.filter,
        global_path=args.global_cfg,
        experiment_name=args.name,
        save_dataset=not args.no_save_dataset,
        save_figures=not args.no_figures,
        verbose=not args.quiet,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

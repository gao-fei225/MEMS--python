"""
数据集 Schema 定义

MVP 数据结构：
- t: (N,) 时间戳
- truth: 真值数据
  - q_nb: (N, 4) 姿态四元数
  - omega_b: (N, 3) 角速度 rad/s
  - a_lin_n: (N, 3) 非重力加速度 m/s^2
  - temp: (N,) 温度 Celsius
- meas: 测量数据
  - gyro: (N, 3) 陀螺仪测量 rad/s
  - acc: (N, 3) 加速度计测量 m/s^2
- meta: 元数据
  - fs: float 采样率 Hz
  - seed: int 随机种子
  - scenario_name: str 工况名称
  - sensor_params: dict 传感器参数
"""

from typing import Dict, Any, List
import numpy as np

# Schema 版本
SCHEMA_VERSION = "1.0.0"

# 必需字段定义
REQUIRED_FIELDS = {
    "t": {"dtype": np.floating, "ndim": 1},
    "truth.q_nb": {"dtype": np.floating, "ndim": 2, "shape_1": 4},
    "truth.omega_b": {"dtype": np.floating, "ndim": 2, "shape_1": 3},
    "truth.a_lin_n": {"dtype": np.floating, "ndim": 2, "shape_1": 3},
    "truth.temp": {"dtype": np.floating, "ndim": 1},
    "meas.gyro": {"dtype": np.floating, "ndim": 2, "shape_1": 3},
    "meas.acc": {"dtype": np.floating, "ndim": 2, "shape_1": 3},
    "meta.fs": {"type": (int, float)},
    "meta.seed": {"type": int},
    "meta.scenario_name": {"type": str},
    "meta.sensor_params": {"type": dict},
}


def create_empty_dataset(n_samples: int, fs: float = 100.0, seed: int = 42,
                         scenario_name: str = "unknown") -> Dict[str, Any]:
    """
    创建空数据集结构
    
    Args:
        n_samples: 样本数量
        fs: 采样率 Hz
        seed: 随机种子
        scenario_name: 工况名称
    
    Returns:
        空数据集字典
    """
    return {
        "t": np.zeros(n_samples, dtype=np.float64),
        "truth": {
            "q_nb": np.zeros((n_samples, 4), dtype=np.float64),
            "omega_b": np.zeros((n_samples, 3), dtype=np.float64),
            "a_lin_n": np.zeros((n_samples, 3), dtype=np.float64),
            "temp": np.zeros(n_samples, dtype=np.float64),
        },
        "meas": {
            "gyro": np.zeros((n_samples, 3), dtype=np.float64),
            "acc": np.zeros((n_samples, 3), dtype=np.float64),
        },
        "meta": {
            "fs": float(fs),
            "seed": int(seed),
            "scenario_name": str(scenario_name),
            "sensor_params": {},
            "schema_version": SCHEMA_VERSION,
        },
    }


def get_n_samples(ds: Dict[str, Any]) -> int:
    """获取数据集样本数量"""
    return len(ds["t"])


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    将嵌套字典展平为单层字典
    
    Args:
        d: 嵌套字典
        parent_key: 父键前缀
        sep: 分隔符
    
    Returns:
        展平后的字典
    """
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    将展平的字典还原为嵌套字典
    
    Args:
        d: 展平的字典
        sep: 分隔符
    
    Returns:
        嵌套字典
    """
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result

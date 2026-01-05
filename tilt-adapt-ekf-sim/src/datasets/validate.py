"""
数据集验证模块

验证数据集结构的完整性和正确性
"""

from typing import Dict, Any, Optional
import numpy as np

from .schema import REQUIRED_FIELDS, flatten_dict


class DatasetValidationError(ValueError):
    """数据集验证错误"""
    pass


def validate_dataset(ds: Dict[str, Any]) -> None:
    """
    验证数据集结构
    
    检查：
    1. 所有必需字段存在
    2. 数据类型正确
    3. 数组维度正确
    4. 所有数组第一维长度一致 (N)
    
    Args:
        ds: 数据集字典
    
    Raises:
        DatasetValidationError: 验证失败时抛出
    """
    # 检查顶层结构
    required_top_keys = ["t", "truth", "meas", "meta"]
    for key in required_top_keys:
        if key not in ds:
            raise DatasetValidationError(f"缺少必需顶层字段: {key}")
    
    # 获取样本数量 N
    t = ds["t"]
    if not isinstance(t, np.ndarray):
        raise DatasetValidationError(f"字段 't' 必须是 numpy 数组，实际类型: {type(t)}")
    
    n_samples = len(t)
    
    # 展平字典以便检查数组字段
    flat = flatten_dict(ds)
    
    # 检查数组字段
    array_fields = {
        "t": {"dtype": np.floating, "ndim": 1},
        "truth.q_nb": {"dtype": np.floating, "ndim": 2, "shape_1": 4},
        "truth.omega_b": {"dtype": np.floating, "ndim": 2, "shape_1": 3},
        "truth.a_lin_n": {"dtype": np.floating, "ndim": 2, "shape_1": 3},
        "truth.temp": {"dtype": np.floating, "ndim": 1},
        "meas.gyro": {"dtype": np.floating, "ndim": 2, "shape_1": 3},
        "meas.acc": {"dtype": np.floating, "ndim": 2, "shape_1": 3},
    }
    
    for field_name, spec in array_fields.items():
        if field_name not in flat:
            raise DatasetValidationError(f"缺少必需字段: {field_name}")
        
        value = flat[field_name]
        
        if not isinstance(value, np.ndarray):
            raise DatasetValidationError(
                f"字段 '{field_name}' 必须是 numpy 数组，实际类型: {type(value)}"
            )
        
        # 检查数据类型
        if not np.issubdtype(value.dtype, spec["dtype"]):
            raise DatasetValidationError(
                f"字段 '{field_name}' 数据类型错误，期望: {spec['dtype']}, 实际: {value.dtype}"
            )
        
        # 检查维度
        if value.ndim != spec["ndim"]:
            raise DatasetValidationError(
                f"字段 '{field_name}' 维度错误，期望: {spec['ndim']}, 实际: {value.ndim}"
            )
        
        # 检查第一维长度 (N)
        if value.shape[0] != n_samples:
            raise DatasetValidationError(
                f"字段 '{field_name}' 第一维长度不一致，期望: {n_samples}, 实际: {value.shape[0]}"
            )
        
        # 检查第二维长度（如果指定）
        if "shape_1" in spec and value.ndim > 1:
            if value.shape[1] != spec["shape_1"]:
                raise DatasetValidationError(
                    f"字段 '{field_name}' 第二维长度错误，期望: {spec['shape_1']}, 实际: {value.shape[1]}"
                )
    
    # 检查 meta 字段
    meta = ds["meta"]
    meta_required = {
        "fs": (int, float),
        "seed": int,
        "scenario_name": str,
        "sensor_params": dict,
    }
    
    for field_name, expected_type in meta_required.items():
        if field_name not in meta:
            raise DatasetValidationError(f"缺少必需元数据字段: meta.{field_name}")
        
        value = meta[field_name]
        if not isinstance(expected_type, tuple):
            expected_type = (expected_type,)
        
        if not isinstance(value, expected_type):
            raise DatasetValidationError(
                f"字段 'meta.{field_name}' 类型错误，期望: {expected_type}, 实际: {type(value)}"
            )
    
    # 额外检查：四元数应该是单位四元数
    q_nb = flat["truth.q_nb"]
    q_norms = np.linalg.norm(q_nb, axis=1)
    if not np.allclose(q_norms, 1.0, atol=1e-6):
        max_err = np.max(np.abs(q_norms - 1.0))
        raise DatasetValidationError(
            f"四元数不是单位四元数，最大模长误差: {max_err:.6f}"
        )


def ensure_quaternion_continuity(q_sequence: np.ndarray) -> np.ndarray:
    """
    确保四元数序列的符号连续性
    
    由于 q 和 -q 表示同一旋转，相邻帧可能出现符号翻转。
    此函数通过检查相邻四元数内积，在内积为负时翻转符号。
    
    Args:
        q_sequence: (N, 4) 四元数序列
    
    Returns:
        符号一致化后的四元数序列
    """
    q_out = q_sequence.copy()
    for i in range(1, len(q_out)):
        if np.dot(q_out[i-1], q_out[i]) < 0:
            q_out[i] = -q_out[i]
    return q_out


def validate_dataset_soft(ds: Dict[str, Any]) -> Optional[str]:
    """
    软验证数据集，返回错误信息而不是抛出异常
    
    Args:
        ds: 数据集字典
    
    Returns:
        错误信息字符串，如果验证通过则返回 None
    """
    try:
        validate_dataset(ds)
        return None
    except DatasetValidationError as e:
        return str(e)


def print_dataset_info(ds: Dict[str, Any]) -> None:
    """
    打印数据集信息
    
    Args:
        ds: 数据集字典
    """
    flat = flatten_dict(ds)
    
    print("=" * 50)
    print("数据集信息")
    print("=" * 50)
    
    # 打印元数据
    print("\n元数据:")
    for key in ["meta.fs", "meta.seed", "meta.scenario_name", "meta.schema_version"]:
        if key in flat:
            print(f"  {key}: {flat[key]}")
    
    # 打印数组信息
    print("\n数组字段:")
    for key, value in flat.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # 打印样本数量
    if "t" in flat:
        n = len(flat["t"])
        duration = flat["t"][-1] - flat["t"][0] if n > 1 else 0
        print(f"\n样本数量: {n}")
        print(f"时长: {duration:.2f} s")
    
    print("=" * 50)

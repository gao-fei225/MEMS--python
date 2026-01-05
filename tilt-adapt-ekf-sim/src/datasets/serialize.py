"""
数据集序列化模块

支持 NPZ 格式的保存和加载
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
import json

from .schema import flatten_dict, unflatten_dict, SCHEMA_VERSION


def save_npz(path: str, ds: Dict[str, Any]) -> None:
    """
    保存数据集为 NPZ 格式
    
    嵌套字典会被展平为 'truth.q_nb' 形式的键名
    非数组类型（meta 中的标量）会被序列化为 JSON 字符串
    所有数组强制转换为 float64 以确保精度和端序一致性
    
    Args:
        path: 保存路径
        ds: 数据集字典
    """
    # 确保目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # 展平字典
    flat = flatten_dict(ds)
    
    # 分离数组和非数组数据
    arrays = {}
    meta_json = {}
    
    for key, value in flat.items():
        if isinstance(value, np.ndarray):
            # 强制转换为 float64 以确保精度和端序一致性
            arrays[key] = np.asarray(value, dtype=np.float64)
        else:
            # 非数组数据存入 meta_json
            meta_json[key] = value
    
    # 将非数组数据序列化为 JSON 字符串，存为特殊键
    arrays["__meta_json__"] = np.array(json.dumps(meta_json))
    arrays["__schema_version__"] = np.array(SCHEMA_VERSION)
    
    # 保存
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> Dict[str, Any]:
    """
    从 NPZ 文件加载数据集
    
    Args:
        path: 文件路径
    
    Returns:
        数据集字典（嵌套结构）
    """
    # 加载 NPZ 文件
    with np.load(path, allow_pickle=True) as data:
        flat = {}
        meta_json = {}
        
        for key in data.files:
            if key == "__meta_json__":
                # 解析 JSON 元数据
                json_str = str(data[key])
                meta_json = json.loads(json_str)
            elif key == "__schema_version__":
                # 跳过版本信息（已包含在 meta 中）
                pass
            else:
                flat[key] = data[key]
        
        # 合并数组和元数据
        flat.update(meta_json)
    
    # 还原嵌套结构
    return unflatten_dict(flat)


def dataset_to_dict(ds: Dict[str, Any]) -> Dict[str, Any]:
    """
    将数据集转换为可 JSON 序列化的字典（用于调试）
    
    Args:
        ds: 数据集字典
    
    Returns:
        可序列化的字典（数组转为列表）
    """
    flat = flatten_dict(ds)
    result = {}
    
    for key, value in flat.items():
        if isinstance(value, np.ndarray):
            result[key] = {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sample": value[:3].tolist() if len(value) > 0 else [],
            }
        else:
            result[key] = value
    
    return result

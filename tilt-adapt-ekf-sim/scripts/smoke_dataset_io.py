#!/usr/bin/env python
"""
数据集 I/O 自检脚本

验证内容：
1. 创建假数据集
2. 保存为 NPZ
3. 加载并验证
4. 打印字段信息

运行方式：
    python scripts/smoke_dataset_io.py
"""

import sys
import tempfile
from pathlib import Path
import numpy as np

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.schema import create_empty_dataset, flatten_dict, SCHEMA_VERSION
from src.datasets.validate import validate_dataset, print_dataset_info, DatasetValidationError
from src.datasets.serialize import save_npz, load_npz


def create_test_dataset(n_samples: int = 100, fs: float = 100.0, seed: int = 42) -> dict:
    """创建测试数据集"""
    rng = np.random.default_rng(seed)
    
    # 创建空数据集
    ds = create_empty_dataset(n_samples, fs, seed, "test_scenario")
    
    # 填充时间戳
    ds["t"] = np.arange(n_samples) / fs
    
    # 填充真值数据
    # 四元数：随机生成并归一化
    q = rng.standard_normal((n_samples, 4))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    ds["truth"]["q_nb"] = q
    
    # 角速度：小幅随机
    ds["truth"]["omega_b"] = rng.standard_normal((n_samples, 3)) * 0.1
    
    # 非重力加速度：小幅随机
    ds["truth"]["a_lin_n"] = rng.standard_normal((n_samples, 3)) * 0.5
    
    # 温度：25°C 附近
    ds["truth"]["temp"] = 25.0 + rng.standard_normal(n_samples) * 2.0
    
    # 填充测量数据
    ds["meas"]["gyro"] = ds["truth"]["omega_b"] + rng.standard_normal((n_samples, 3)) * 0.01
    ds["meas"]["acc"] = rng.standard_normal((n_samples, 3)) * 0.1
    ds["meas"]["acc"][:, 2] -= 9.8  # 添加重力
    
    # 填充元数据
    ds["meta"]["sensor_params"] = {
        "gyro_noise": 0.01,
        "acc_noise": 0.1,
    }
    
    return ds


def test_create_and_validate():
    """测试 1: 创建并验证数据集"""
    print("=" * 60)
    print("测试 1: 创建并验证数据集")
    print("=" * 60)
    
    ds = create_test_dataset(n_samples=100)
    
    try:
        validate_dataset(ds)
        print("✓ 数据集验证通过")
        return True
    except DatasetValidationError as e:
        print(f"✗ 数据集验证失败: {e}")
        return False


def test_save_load_roundtrip():
    """测试 2: 保存-加载 Round-Trip"""
    print("\n" + "=" * 60)
    print("测试 2: 保存-加载 Round-Trip")
    print("=" * 60)
    
    ds_original = create_test_dataset(n_samples=100)
    
    # 使用临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_dataset.npz"
        
        # 保存
        save_npz(str(filepath), ds_original)
        print(f"✓ 保存到: {filepath}")
        print(f"  文件大小: {filepath.stat().st_size} bytes")
        
        # 加载
        ds_loaded = load_npz(str(filepath))
        print("✓ 加载成功")
        
        # 验证加载的数据集
        try:
            validate_dataset(ds_loaded)
            print("✓ 加载后验证通过")
        except DatasetValidationError as e:
            print(f"✗ 加载后验证失败: {e}")
            return False
        
        # 比较原始和加载的数据
        flat_orig = flatten_dict(ds_original)
        flat_load = flatten_dict(ds_loaded)
        
        all_match = True
        for key in flat_orig:
            if key not in flat_load:
                print(f"✗ 缺少字段: {key}")
                all_match = False
                continue
            
            orig_val = flat_orig[key]
            load_val = flat_load[key]
            
            if isinstance(orig_val, np.ndarray):
                if not np.allclose(orig_val, load_val):
                    max_diff = np.max(np.abs(orig_val - load_val))
                    print(f"✗ 字段 '{key}' 数值不匹配，最大差异: {max_diff}")
                    all_match = False
            else:
                if orig_val != load_val:
                    print(f"✗ 字段 '{key}' 值不匹配: {orig_val} vs {load_val}")
                    all_match = False
        
        if all_match:
            print("✓ 所有字段匹配")
            return True
        else:
            return False


def test_print_info():
    """测试 3: 打印数据集信息"""
    print("\n" + "=" * 60)
    print("测试 3: 打印数据集信息")
    print("=" * 60)
    
    ds = create_test_dataset(n_samples=100)
    print_dataset_info(ds)
    return True


def test_validation_errors():
    """测试 4: 验证错误检测"""
    print("\n" + "=" * 60)
    print("测试 4: 验证错误检测")
    print("=" * 60)
    
    all_passed = True
    
    # 测试缺少字段
    ds = create_test_dataset(n_samples=100)
    del ds["truth"]["q_nb"]
    try:
        validate_dataset(ds)
        print("✗ 应该检测到缺少字段")
        all_passed = False
    except DatasetValidationError as e:
        print(f"✓ 正确检测到缺少字段: {e}")
    
    # 测试维度错误
    ds = create_test_dataset(n_samples=100)
    ds["truth"]["q_nb"] = np.zeros((100, 3))  # 应该是 (100, 4)
    try:
        validate_dataset(ds)
        print("✗ 应该检测到维度错误")
        all_passed = False
    except DatasetValidationError as e:
        print(f"✓ 正确检测到维度错误: {e}")
    
    # 测试长度不一致
    ds = create_test_dataset(n_samples=100)
    ds["truth"]["omega_b"] = np.zeros((50, 3))  # 应该是 (100, 3)
    try:
        validate_dataset(ds)
        print("✗ 应该检测到长度不一致")
        all_passed = False
    except DatasetValidationError as e:
        print(f"✓ 正确检测到长度不一致: {e}")
    
    # 测试非单位四元数
    ds = create_test_dataset(n_samples=100)
    ds["truth"]["q_nb"] = np.ones((100, 4))  # 模长不为 1
    try:
        validate_dataset(ds)
        print("✗ 应该检测到非单位四元数")
        all_passed = False
    except DatasetValidationError as e:
        print(f"✓ 正确检测到非单位四元数: {e}")
    
    return all_passed


def main():
    print("\n" + "=" * 60)
    print("数据集 I/O 自检脚本")
    print(f"Schema 版本: {SCHEMA_VERSION}")
    print("=" * 60)
    
    results = []
    
    results.append(("创建并验证", test_create_and_validate()))
    results.append(("保存-加载 Round-Trip", test_save_load_roundtrip()))
    results.append(("打印信息", test_print_info()))
    results.append(("验证错误检测", test_validation_errors()))
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！数据集 I/O 正常。")
        return 0
    else:
        print("存在测试失败！请检查数据集模块。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

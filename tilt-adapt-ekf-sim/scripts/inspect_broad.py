"""检查 BROAD 数据集结构"""
import h5py
import numpy as np

# 打开一个示例文件
filepath = 'data/datasets/BROAD/broad/data_hdf5/01_undisturbed_slow_rotation_A.hdf5'
f = h5py.File(filepath, 'r')

print("=" * 60)
print("BROAD 数据集结构分析")
print("=" * 60)
print(f"\n文件: {filepath}\n")
print("包含的数据字段:")
print("-" * 40)

for key in f.keys():
    data = f[key][:]
    print(f"  {key}:")
    print(f"    - shape: {data.shape}")
    print(f"    - dtype: {data.dtype}")
    if len(data.shape) == 1:
        print(f"    - 范围: [{data.min():.4f}, {data.max():.4f}]")
    elif len(data.shape) == 2:
        print(f"    - 前3行样本:")
        for i in range(min(3, data.shape[0])):
            print(f"      {data[i]}")

# 检查采样率
if 'gyr' in f.keys() and 'sampling_rate' in f.attrs:
    print(f"\n采样率: {f.attrs['sampling_rate']} Hz")
else:
    # 尝试从数据长度推断
    gyr = f['gyr'][:]
    print(f"\n数据点数: {gyr.shape[0]}")

f.close()

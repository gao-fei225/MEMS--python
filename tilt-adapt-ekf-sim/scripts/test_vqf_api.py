"""测试 VQF 官方库的正确用法"""
import numpy as np
from vqf import VQF

# 创建简单测试数据
dt = 1.0 / 100  # 100 Hz
n = 100

# 静止状态：陀螺仪为0，加速度计指向重力
gyr = np.zeros((n, 3))
acc = np.tile([0, 0, 9.81], (n, 1))
mag = np.tile([0.3, 0, 0.5], (n, 1))

print("=" * 60)
print("测试 VQF 官方库 API")
print("=" * 60)

# 测试不同的初始化方式
print("\n1. 测试初始化参数...")
try:
    vqf1 = VQF(dt)
    print(f"   ✓ VQF(dt) 成功")
except Exception as e:
    print(f"   ✗ VQF(dt) 失败: {e}")

try:
    vqf2 = VQF(gyrTs=dt, accTs=dt, magTs=dt)
    print(f"   ✓ VQF(gyrTs, accTs, magTs) 成功")
except Exception as e:
    print(f"   ✗ VQF(gyrTs, accTs, magTs) 失败: {e}")

# 使用最简单的初始化
vqf = VQF(dt)

print("\n2. 测试 update 方法...")
print(f"   输入数据格式:")
print(f"   - gyr shape: {gyr[0].shape}, 单位: rad/s")
print(f"   - acc shape: {acc[0].shape}, 单位: m/s^2")
print(f"   - mag shape: {mag[0].shape}, 单位: 任意")

try:
    # 测试单次更新
    vqf.update(gyr[0], acc[0], mag[0])
    print(f"   ✓ update(gyr, acc, mag) 成功")
except Exception as e:
    print(f"   ✗ update 失败: {e}")
    print(f"   尝试其他格式...")
    
    # 尝试角度单位
    try:
        vqf.update(np.rad2deg(gyr[0]), acc[0], mag[0])
        print(f"   ✓ update(deg/s, m/s^2, mag) 成功")
    except Exception as e2:
        print(f"   ✗ 仍然失败: {e2}")

print("\n3. 测试输出方法...")
methods = ['getQuat3D', 'getQuat6D', 'getQuat9D', 'quat', 'quat6D', 'quat9D']
for method in methods:
    try:
        if hasattr(vqf, method):
            result = getattr(vqf, method)()
            print(f"   ✓ {method}() -> shape: {result.shape}, values: {result}")
        else:
            print(f"   ✗ {method}() 不存在")
    except Exception as e:
        print(f"   ✗ {method}() 失败: {e}")

print("\n4. 测试批处理...")
try:
    vqf_batch = VQF(dt)
    for i in range(10):
        vqf_batch.update(gyr[i], acc[i], mag[i])
    quat = vqf_batch.getQuat6D()
    print(f"   ✓ 批处理成功，最终四元数: {quat}")
except Exception as e:
    print(f"   ✗ 批处理失败: {e}")

print("\n5. 检查 VQF 类的所有方法和属性...")
print(f"   可用方法: {[m for m in dir(vqf) if not m.startswith('_')]}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

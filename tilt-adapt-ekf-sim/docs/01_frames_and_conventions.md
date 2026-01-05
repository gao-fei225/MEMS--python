# 01 Frames and Conventions

## 1. 坐标系定义

### 1.1 导航坐标系 (Navigation Frame, n)

- **符号**: `n`
- **定义**: 局部水平坐标系，固定于地球表面
- **轴向**:
  - X: 指向北 (North)
  - Y: 指向东 (East)  
  - Z: 指向地心 (Down)
- **别名**: NED (North-East-Down)

```
      N (X)
       ↑
       |
       |
E (Y) ←——— O
       ↓
      D (Z)
```

### 1.2 机体坐标系 (Body Frame, b)

- **符号**: `b`
- **定义**: 固连于载体的坐标系
- **轴向**:
  - X: 指向载体前方 (Forward)
  - Y: 指向载体右侧 (Right)
  - Z: 指向载体下方 (Down)
- **别名**: FRD (Forward-Right-Down)

```
      F (X)
       ↑
       |
       |
R (Y) ←——— O
       ↓
      D (Z)
```

### 1.3 IMU 传感器坐标系 (Sensor Frame, s)

- **符号**: `s`
- **定义**: IMU 传感器的测量坐标系
- **说明**: 理想情况下与机体坐标系重合，实际存在安装偏差 (misalignment)

---

## 2. 重力定义

### 2.1 重力向量

在导航坐标系中，重力向量定义为：

```
g_n = [0, 0, g]^T
```

其中 `g = 9.80665 m/s²` (标准重力加速度)

### 2.2 重力方向说明

- 重力沿导航坐标系 Z 轴正方向（指向地心）
- 加速度计在静止时测量的是 **比力** (specific force)，即 `-g_n` 在机体系的投影

```python
# 静止时加速度计测量值
a_b_measured = R_bn @ (-g_n) = R_bn @ [0, 0, -g]^T
```

---

## 3. 四元数定义

### 3.1 四元数表示

采用 **Hamilton 约定**，四元数表示为：

```
q = w + xi + yj + zk = [w, x, y, z]^T
```

其中：
- `w`: 标量部分 (scalar part)
- `[x, y, z]`: 向量部分 (vector part)
- `i, j, k`: 虚数单位，满足 `i² = j² = k² = ijk = -1`

### 3.2 四元数乘法

Hamilton 约定下的四元数乘法：

```
q1 ⊗ q2 = [w1*w2 - x1*x2 - y1*y2 - z1*z2,
           w1*x2 + x1*w2 + y1*z2 - z1*y2,
           w1*y2 - x1*z2 + y1*w2 + z1*x2,
           w1*z2 + x1*y2 - y1*x2 + z1*w2]
```

### 3.3 姿态四元数定义

**q_nb** 表示从 **机体系 (b) 到导航系 (n)** 的旋转：

```
v_n = R_nb(q_nb) @ v_b
```

- `q_nb`: 将机体系向量变换到导航系的四元数
- `R_nb`: 对应的旋转矩阵 (3×3)
- `R_bn = R_nb^T = R_nb^{-1}`: 逆变换

### 3.4 四元数与旋转矩阵

从四元数 `q = [w, x, y, z]` 计算旋转矩阵 `R_nb`:

```
R_nb = [1-2(y²+z²),  2(xy-wz),    2(xz+wy)  ]
       [2(xy+wz),    1-2(x²+z²),  2(yz-wx)  ]
       [2(xz-wy),    2(yz+wx),    1-2(x²+y²)]
```

### 3.5 单位四元数约束

姿态四元数必须满足单位约束：

```
||q|| = sqrt(w² + x² + y² + z²) = 1
```

---

## 4. 欧拉角定义

### 4.1 欧拉角顺序

采用 **ZYX 顺序** (Yaw-Pitch-Roll)，也称为 3-2-1 顺序：

1. 绕 Z 轴旋转 ψ (Yaw, 偏航角)
2. 绕 Y 轴旋转 θ (Pitch, 俯仰角)
3. 绕 X 轴旋转 φ (Roll, 横滚角)

```
R_nb = R_z(ψ) @ R_y(θ) @ R_x(φ)
```

### 4.2 欧拉角符号定义

| 角度 | 符号 | 正方向 | 范围 |
|------|------|--------|------|
| Roll (横滚) | φ | 左翼下沉为正（绕 X 轴右手定则） | [-π, π] |
| Pitch (俯仰) | θ | 抬头为正（绕 Y 轴右手定则） | [-π/2, π/2] |
| Yaw (偏航) | ψ | 逆时针为正（从上往下看，绕 Z 轴右手定则） | [-π, π] |

### 4.3 欧拉角与四元数转换

**四元数 → 欧拉角**:

```python
def quat_to_euler(q):
    """q = [w, x, y, z] -> [roll, pitch, yaw]"""
    w, x, y, z = q
    
    # Roll (φ)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (θ)
    sinp = 2 * (w * y - z * x)
    sinp = clip(sinp, -1, 1)  # 数值保护
    pitch = asin(sinp)
    
    # Yaw (ψ)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)
    
    return [roll, pitch, yaw]
```

**欧拉角 → 四元数**:

```python
def euler_to_quat(roll, pitch, yaw):
    """[roll, pitch, yaw] -> q = [w, x, y, z]"""
    cr, sr = cos(roll/2), sin(roll/2)
    cp, sp = cos(pitch/2), sin(pitch/2)
    cy, sy = cos(yaw/2), sin(yaw/2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return [w, x, y, z]
```

---

## 5. 角速度定义

### 5.1 角速度向量

角速度 `ω_b` 定义在 **机体坐标系** 中：

```
ω_b = [ω_x, ω_y, ω_z]^T  (rad/s)
```

- `ω_x`: 绕机体 X 轴的角速度 (roll rate)
- `ω_y`: 绕机体 Y 轴的角速度 (pitch rate)
- `ω_z`: 绕机体 Z 轴的角速度 (yaw rate)

### 5.2 四元数微分方程

姿态四元数的时间导数：

```
q̇_nb = 0.5 * q_nb ⊗ [0, ω_b]^T
```

或写成矩阵形式：

```
q̇ = 0.5 * Ω(ω) @ q
```

其中：

```
Ω(ω) = [ 0,   -ω_x, -ω_y, -ω_z]
       [ω_x,   0,    ω_z, -ω_y]
       [ω_y, -ω_z,   0,    ω_x]
       [ω_z,  ω_y, -ω_x,   0  ]
```

---

## 6. 传感器测量模型

### 6.1 陀螺仪测量模型

```
ω_meas = ω_true + b_g + n_g
```

- `ω_meas`: 陀螺仪测量值 (rad/s)
- `ω_true`: 真实角速度 (rad/s)
- `b_g`: 陀螺仪偏置 (rad/s)
- `n_g`: 陀螺仪噪声 (rad/s)

### 6.2 加速度计测量模型

```
a_meas = R_bn @ (a_true - g_n) + b_a + n_a
```

- `a_meas`: 加速度计测量值 (m/s²)
- `a_true`: 真实加速度（导航系）(m/s²)
- `g_n`: 重力向量（导航系）(m/s²)
- `R_bn`: 导航系到机体系的旋转矩阵
- `b_a`: 加速度计偏置 (m/s²)
- `n_a`: 加速度计噪声 (m/s²)

**静止时**（`a_true = 0`）：

```
a_meas = R_bn @ (-g_n) + b_a + n_a = -R_bn @ g_n + b_a + n_a
```

---

## 7. 单位约定

| 物理量 | 单位 | 说明 |
|--------|------|------|
| 角度 | rad | 内部计算使用弧度 |
| 角速度 | rad/s | |
| 加速度 | m/s² | |
| 时间 | s | |
| 温度 | °C | 摄氏度 |
| 频率 | Hz | |

**注意**: 配置文件中的角度参数使用 **度 (deg)**，代码内部自动转换为弧度。

---

## 8. 代码实现参考

```python
# src/common/math3d.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Quaternion:
    """
    四元数类，Hamilton 约定
    q = [w, x, y, z]，表示从机体系到导航系的旋转
    """
    w: float
    x: float
    y: float
    z: float
    
    def normalize(self) -> 'Quaternion':
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        return Quaternion(self.w/norm, self.x/norm, self.y/norm, self.z/norm)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])
    
    def to_rotation_matrix(self) -> np.ndarray:
        """返回 R_nb 旋转矩阵"""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z),     2*(x*z+w*y)],
            [2*(x*y+w*z),     1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y),     2*(y*z+w*x),     1-2*(x**2+y**2)]
        ])
    
    def to_euler(self) -> Tuple[float, float, float]:
        """返回 (roll, pitch, yaw) 弧度"""
        w, x, y, z = self.w, self.x, self.y, self.z
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1, 1)
        pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return (roll, pitch, yaw)

# 重力常数
GRAVITY = 9.80665  # m/s²

# 导航系重力向量 (NED)
GRAVITY_NED = np.array([0.0, 0.0, GRAVITY])
```

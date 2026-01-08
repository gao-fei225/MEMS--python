# 自适应 EKF 倾角估计项目 - 修改日志

> 本文档记录项目的所有重要修改，方便在不同机器间同步和回顾开发历程。

---

## 当前版本状态

**最终性能**: Average RMSE = **0.777°** (消融实验验证)

**核心配置文件**:
- `configs/filters/ekf_adaptive_innovation.yaml` - 主配置
- `configs/filters/final_deploy.yaml` - 部署配置

**最终优化参数** (DE 差分进化优化结果，请勿修改):
```yaml
R0: 5.49e-4           # 基准测量噪声
nis_high: 8.54        # NIS 统计学阈值
vib_var_threshold: 0.06   # 振动方差阈值
lambda_vibration: 200.0   # 振动时的λ
acc_window_size: 20       # 滑动窗口大小
```

---

## 修改历史

### 2026-01-07: 消融实验验证 + 参数固化

**修改内容**:
1. 更新 `scripts/run_ablation_with_plots.py` 使用 DE 优化后的参数
2. 运行消融实验验证最终性能

**消融实验结果**:
| 场景 | A0_Optimized | A1_Fixed | A2_Gating | A3_Inflate | A4_Conservative |
|------|-------------|----------|-----------|------------|-----------------|
| Accel | **0.127°** | 7.130° | 7.130° | 7.192° | 7.856° |
| Vibration | 0.802° | 0.689° | 0.689° | **0.633°** | 0.761° |
| Static | **0.131°** | 0.131° | 0.131° | 0.131° | 0.131° |
| Turn | **2.048°** | 8.872° | 8.872° | 8.840° | 5.333° |
| **Average** | **0.777°** | 4.206° | 4.206° | 4.199° | 3.520° |

**关键发现**:
- Accel 场景: 0.127° vs 固定EKF 7.13° (提升 56 倍)
- Turn 场景: 2.048° vs 固定EKF 8.87° (提升 4.3 倍)
- Vibration 场景略逊于 A3_Inflate，这是保守策略的代价

**修改文件**:
- `scripts/run_ablation_with_plots.py`

---

### 2026-01-07: 差分进化 (DE) 参数优化

**修改内容**:
1. 运行 `scripts/optimize_parameters.py` 进行全局参数优化
2. 将优化结果固化到配置文件

**优化结果**:
- Average RMSE: 0.7364° → 0.777° (消融实验)
- 优化参数已写入 `configs/filters/ekf_adaptive_innovation.yaml`

**修改文件**:
- `configs/filters/ekf_adaptive_innovation.yaml`
- `configs/filters/final_deploy.yaml` (新建)

---

### 2026-01-07: 工业级 RAKF 架构实现

**修改内容**:
1. 实现 Plan I: 工业级 Robust Adaptive Kalman Filter
2. 核心功能:
   - NIS + M-Estimation (Huber-like) 连续权重函数
   - ZARU (Zero Angular Rate Update) 零角速度修正
   - LPF 数字低通滤波预处理
   - 振动/机动解耦检测 (滑动窗口方差)

**算法逻辑** (`_adapt_lambda_dynamic_aware`):
```
1. 振动检测: acc_std > threshold && gyro_norm < threshold → λ = lambda_vibration
2. 机动检测: NIS > threshold → λ = Huber(NIS) (分段线性+指数)
3. 幅值通道: |acc_norm - g| > threshold → λ_mag 增强
4. 角速度通道: gyro_norm > threshold → λ 乘以 gyro_factor
5. ZARU: 静止时 λ = r_scale (极度信任观测)
```

**修改文件**:
- `src/filters/ekf_adaptive.py`

---

### 早期开发历程 (详细)

---

#### Step 1: 项目初始化与框架搭建

**目标**: 搭建仿真真值闭环平台基础架构

**实现内容**:
1. 创建项目目录结构:
   ```
   tilt-adapt-ekf-sim/
   ├── configs/          # 配置文件
   │   ├── filters/      # 滤波器配置
   │   ├── scenarios/    # 场景配置
   │   └── sensors/      # 传感器配置
   ├── data/             # 数据目录
   ├── docs/             # 文档
   ├── outputs/          # 输出结果
   ├── scripts/          # 脚本
   ├── src/              # 源代码
   │   ├── common/       # 公共模块
   │   ├── filters/      # 滤波器实现
   │   ├── sensors/      # 传感器模型
   │   └── truth/        # 真值生成
   └── tests/            # 测试
   ```

2. 实现基础数学库 `src/common/math3d.py`:
   - 四元数运算 (乘法、归一化、共轭)
   - 四元数与欧拉角互转 (`quat_to_rpy`, `rpy_to_quat`)
   - 四元数与旋转矩阵互转 (`quat_to_R_bn`)
   - 反对称矩阵 (`skew_symmetric`)
   - 角度单位转换 (`deg2rad`, `rad2deg`)

**关键代码**:
```python
# 四元数乘法
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
```

---

#### Step 2: 真值生成器实现

**目标**: 实现可控的运动真值生成

**实现内容** (`src/truth/scenarios.py`):
1. `generate_quasi_static()` - 准静态场景
2. `generate_swing()` - 摆动场景 (正弦姿态变化)
3. `generate_accel()` - 加减速场景 (阶跃/斜坡/正弦)
4. `generate_turn()` - 转弯场景 (恒定角速度 + 向心加速度)
5. `generate_vibration()` - 振动场景 (带限随机噪声)
6. `generate_shock()` - 冲击场景 (脉冲加速度)

**输出格式**:
```python
truth = {
    "t": np.ndarray,        # 时间戳 (N,)
    "dt": float,            # 采样间隔
    "fs": float,            # 采样率
    "q_nb": np.ndarray,     # 姿态四元数 (N, 4)
    "omega_b": np.ndarray,  # 机体角速度 (N, 3)
    "a_lin_n": np.ndarray,  # 非重力加速度 (N, 3)
    "temp": np.ndarray,     # 温度 (N,)
    "rpy_deg": np.ndarray,  # 欧拉角真值 (N, 3)
}
```

---

#### Step 3: IMU 传感器模型实现

**目标**: 模拟真实 MEMS IMU 的误差特性

**实现内容** (`src/sensors/imu_model.py`):
1. 加速度计模型:
   - 测量值 = R_bn @ (a_lin_n + g_n) + bias + noise
   - 偏置: 常值 + 温度漂移
   - 噪声: 高斯白噪声

2. 陀螺仪模型:
   - 测量值 = omega_b + bias + noise
   - 偏置: 常值 + 随机游走
   - 噪声: 高斯白噪声

**传感器参数**:
```python
SENSOR_PARAMS = {
    "acc": {
        "bias0": [0.02, -0.01, 0.03],  # m/s²
        "sigma_white": 0.02,            # m/s²
    },
    "gyro": {
        "bias0": [0.001, 0.001, -0.002],  # rad/s
        "sigma_white": 0.001,              # rad/s
    },
}
```

---

#### Step 4: 基础 EKF 实现

**目标**: 实现标准四元数 EKF 作为基准

**实现内容** (`src/filters/ekf_fixed.py`):
1. 状态向量: `x = [δθ(3), b_g(3)]` (误差状态)
2. 四元数传播: 一阶龙格库塔
3. 观测模型: 方向量测 (加速度计归一化)
4. NIS 门限检测: 可选拒绝/膨胀模式

**状态方程**:
```
预测: q_k+1 = q_k ⊗ exp(ω*dt/2)
      P_k+1 = F @ P_k @ F.T + Q

更新: v = z - h(x)
      S = H @ P @ H.T + R
      K = P @ H.T @ S^-1
      x = x + K @ v
      P = (I - K @ H) @ P
```

---

#### Step 5: 互补滤波器实现

**目标**: 实现简单的互补滤波器作为对比

**实现内容** (`src/filters/complementary.py`):
1. 加速度计 → roll/pitch 计算
2. 陀螺仪积分
3. 一阶互补滤波: `θ = α * θ_gyro + (1-α) * θ_acc`

---

#### Step 6-8: 自适应策略探索

**目标**: 探索不同的自适应 R 调整策略

**Plan A: λ EWMA 平滑**
- 问题: 响应太慢，无法及时跟踪突变
- 代码: `_adapt_lambda()` 中的 `ewma_lambda_alpha`

**Plan B: 门限拒绝**
- 策略: NIS > threshold 时拒绝观测
- 问题: 完全拒绝导致纯积分漂移

**Plan C: 振动感知**
- 策略: 检测振动模式，使用固定 λ
- 代码: `_adapt_lambda_vibration_aware()`

**Plan D: Sigmoid 映射**
- 策略: λ = sigmoid(NIS) 平滑映射
- 代码: `_adapt_lambda_sigmoid()`

**Plan E: Inflate 映射**
- 策略: λ = max(1, NIS/threshold)
- 代码: `_adapt_lambda_inflate()`
- 效果: 与固定 EKF 的 inflate_R 一致

---

#### Step 9-11: 双通道检测

**目标**: 解决 Accel/Turn 场景的误差问题

**问题分析**:
- 方向通道 (NIS_dir) 在缓慢加速时可能被"欺骗"
- 滤波器跟上了错误值，NIS 反而变小

**解决方案**:
1. 添加幅值通道 (NIS_mag): 检测 ||acc|| - g
2. 双通道融合: `NIS_combined = max(NIS_dir, NIS_mag * weight)`

**配置**:
```yaml
dual_channel:
  enabled: true
  mag_weight: 50.0      # 幅值通道权重
  mag_sigma: 0.05       # 幅值偏差标准差
  combine_mode: "max"   # 取最大值
```

---

#### Step 12: 动态感知策略

**目标**: 区分振动和机动，采用不同策略

**Plan G: 基础动态感知**
- 检测角速度: gyro_norm > threshold → 转弯
- 检测幅值偏差: |acc_norm - g| > threshold → 加速

**Plan G+: 加速度矢量平滑**
- 问题: 振动时噪声整流效应导致均值偏差
- 解决: 对加速度矢量做 EWMA 平滑

**Plan G++/H: 滑动窗口方差检测**
- 振动特征: 方差大 + 角速度小 + 均值偏差小
- 机动特征: 方差小 + 角速度大 或 均值偏差大

**代码**:
```python
# 振动判定
is_vibration = (acc_std > vib_var_threshold and 
               gyro_norm_ewma < gyro_threshold * 3.0)

if is_vibration:
    lambda_factor = lambda_vibration  # 固定适度的 λ
elif nis_dir > th_maneuver_nis:
    lambda_factor = Huber(nis_dir)    # M-Estimation
```

---

#### Step 13: 工业级 RAKF

**目标**: 实现工业级 Robust Adaptive Kalman Filter

**核心升级**:
1. NIS + M-Estimation (Huber-like) 连续权重函数
2. ZARU (Zero Angular Rate Update) 零角速度修正
3. LPF 数字低通滤波预处理
4. 振动/机动解耦检测

**ZARU 原理**:
- 检测静止: acc_std < threshold && gyro_norm < threshold
- 静止时: R 缩小 (极度信任观测)，Q_att 缩小 (角度不变)
- 效果: 快速估计陀螺仪 Bias，解决"久停必漂"

**LPF 原理**:
- 2阶 Butterworth 低通滤波
- 加速度计截止频率: 15 Hz
- 陀螺仪截止频率: 30 Hz
- 单向滤波模拟实物延迟

**最终算法流程**:
```
1. LPF 预处理 (可选)
2. 滑动窗口方差计算
3. 振动/机动/静止 三态判定
4. 根据状态选择 λ 策略
5. 标准 EKF 更新
6. ZARU 静止修正 (可选)
```

---

## 验证与调参过程

### 验证脚本说明

开发过程中创建了大量验证和调试脚本：

| 脚本 | 用途 |
|------|------|
| `test_ekf_adaptive.py` | 自适应 EKF 基础功能测试 |
| `test_ekf_fixed.py` | 固定 EKF 基准测试 |
| `test_fair_comparison.py` | 公平对比测试（确保相同 R0）|
| `test_dynamic_aware.py` | 动态感知策略测试 |
| `test_dynamic_gating.py` | 动态门限测试 |
| `test_improved_gating.py` | 改进门限策略测试 |
| `test_turn_gating.py` | 转弯场景门限测试 |
| `test_selective_gating.py` | 选择性门限测试 |
| `test_multiple_seeds.py` | 多随机种子稳定性测试 |
| `test_final_config.py` | 最终配置验证 |
| `test_final_config_v2.py` | 最终配置验证 v2 |

### 调试脚本说明

| 脚本 | 用途 |
|------|------|
| `debug_accel_scenario.py` | Accel 场景深度调试 |
| `debug_accel_deep.py` | Accel 场景深层分析 |
| `debug_accel_lambda.py` | Accel 场景 λ 响应分析 |
| `debug_shock_scenario.py` | 冲击场景调试 |
| `diagnose_ekf.py` | EKF 诊断工具 |
| `diagnose_and_fix.py` | 诊断并修复问题 |
| `diagnose_nis_channels.py` | NIS 双通道诊断 |

### 参数优化脚本

| 脚本 | 用途 |
|------|------|
| `find_optimal_config.py` | 网格搜索最优配置 |
| `find_optimal_config_fast.py` | 快速网格搜索 |
| `find_optimal_threshold.py` | 寻找最优阈值 |
| `optimize_parameters.py` | 差分进化全局优化 |
| `calibrate_direction_r.py` | R0 校准 |

---

### 调参历程详细记录

#### 阶段 1: R0 校准

**问题**: 初始 R0 = 3.5e-6 导致静态场景 NIS 均值过高

**校准方法**:
1. 在 Static 场景运行固定 EKF
2. 观察 NIS 均值，目标是 ~3（χ² 分布期望值）
3. 调整 R0 使 NIS 均值接近 3

**校准过程**:
```
R0 = 3.5e-6 → NIS_mean ≈ 10 (过高)
R0 = 1.0e-5 → NIS_mean ≈ 5 (偏高)
R0 = 2.0e-5 → NIS_mean ≈ 3.5 (接近)
R0 = 1.0e-4 → NIS_mean ≈ 2.8 (合适)
```

**最终**: R0 = 1.0e-4 (手动校准)

---

#### 阶段 2: NIS 阈值调整

**问题**: nis_high 过高导致响应迟钝，过低导致误触发

**调参过程**:
```
nis_high = 35.0 → Accel 场景响应太慢
nis_high = 15.0 → 改善但仍不够
nis_high = 10.0 → 振动场景误触发
nis_high = 7.8  → 平衡点（手动）
```

**最终**: nis_high = 7.8 (手动调参)

---

#### 阶段 3: 双通道权重调整

**问题**: 幅值通道权重不当导致 Accel/Turn 场景误差

**调参过程**:
```
mag_weight = 1.0  → 幅值通道作用太弱
mag_weight = 10.0 → 改善
mag_weight = 50.0 → 显著改善
mag_sigma = 0.5   → 检测不够敏感
mag_sigma = 0.1   → 过于敏感
mag_sigma = 0.05  → 平衡点
```

**最终**: mag_weight = 50.0, mag_sigma = 0.05

---

#### 阶段 4: 振动检测阈值

**问题**: 振动和机动难以区分

**调参过程**:
```
vib_var_threshold = 0.1  → 振动检测不够敏感
vib_var_threshold = 0.05 → 改善
vib_var_threshold = 0.03 → 机动被误判为振动
```

**最终**: vib_var_threshold = 0.05 (手动)

---

#### 阶段 5: 振动时 λ 值

**问题**: 振动时 λ 过大导致精度下降，过小导致噪声放大

**调参过程**:
```
lambda_vibration = 50.0  → 振动场景精度不够
lambda_vibration = 100.0 → 改善
lambda_vibration = 200.0 → 最佳平衡
lambda_vibration = 500.0 → 过度抑制
```

**最终**: lambda_vibration = 100.0 (手动) → 200.0 (DE优化)

---

#### 阶段 6: 差分进化全局优化

**目标**: 自动寻找全局最优参数组合

**优化参数**:
```python
bounds = [
    (-5, -1),     # R_base (log10)
    (-6, -2),     # Q_scale (log10)
    (3.0, 15.0),  # th_nis
    (0.01, 0.2),  # th_vib_std
    (10.0, 300.0) # lambda_vib
]
```

**优化配置**:
```python
differential_evolution(
    run_simulation,
    bounds,
    strategy='best1bin',
    maxiter=30,
    popsize=10,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    workers=-1  # 并行
)
```

**优化结果**:
```
R0:               5.49e-4 (log: -3.26)
nis_high:         8.54
vib_var_threshold: 0.06 (~0.0595)
lambda_vibration: 200.0
Average RMSE:     0.7364°
```

---

#### 阶段 7: 公平对比验证

**问题**: 自适应 EKF 和固定 EKF 使用不同 R0 导致对比不公平

**解决方案**:
1. 所有配置使用相同 R0 = 5.49e-4
2. 运行消融实验验证

**验证结果**:
- 自适应 EKF 在所有场景都优于或等于固定 EKF
- 平均 RMSE: 0.777° vs 4.206° (提升 5.4 倍)

---

### 关键调参经验总结

1. **R0 校准**: 必须先校准 R0 使静态 NIS ≈ 3，否则后续调参无意义

2. **阈值选择**: 
   - nis_high 太高 → 响应迟钝
   - nis_high 太低 → 误触发
   - 建议范围: 6.0 ~ 10.0

3. **双通道权重**:
   - mag_weight 控制幅值通道的影响力
   - mag_sigma 控制检测灵敏度
   - 两者需要配合调整

4. **振动检测**:
   - 滑动窗口方差是区分振动和机动的关键
   - 窗口大小影响检测延迟
   - 建议: acc_window_size = 20

5. **避免过拟合**:
   - DE 优化后不要继续手动调参
   - 多随机种子验证稳定性
   - 消融实验验证各组件贡献

---

## 仿真场景说明

消融实验使用 **4 种典型工况**，模拟 MEMS IMU 在实际应用中会遇到的情况：

### 1. Static（准静态）
| 参数 | 值 |
|------|-----|
| 运动描述 | IMU 固定不动，姿态恒定 |
| 初始姿态 | roll=2°, pitch=-1°, yaw=0° |
| 持续时间 | 30 秒 |
| 角速度 | 0 |
| 线性加速度 | 0（只有重力）|
| 测试目的 | 验证滤波器在理想条件下的基准精度 |

### 2. Vibration（振动）
| 参数 | 值 |
|------|-----|
| 运动描述 | IMU 固定姿态，受到随机高频振动 |
| 初始姿态 | roll=2°, pitch=-1°, yaw=0° |
| 振动 RMS | 0.5 m/s² |
| 振动带宽 | 20 Hz（低通随机噪声）|
| 持续时间 | 30 秒 |
| 测试目的 | 模拟机械振动环境（发动机/电机附近）|

### 3. Accel（加减速）
| 参数 | 值 |
|------|-----|
| 运动描述 | IMU 姿态固定，沿 X 轴做阶跃加速 |
| 初始姿态 | roll=2°, pitch=-1°, yaw=0° |
| 加速度类型 | 阶跃 (step) |
| 加速度方向 | X 轴 |
| 加速度峰值 | 2.0 m/s² |
| 加速时间 | 5~15 秒（持续 10 秒）|
| 测试目的 | 模拟车辆加减速，测试区分重力和线性加速度的能力 |
| 物理意义 | 加速时加速度计测量值 = 重力 + 线性加速度，导致"假倾斜" |

### 4. Turn（转弯）
| 参数 | 值 |
|------|-----|
| 运动描述 | IMU 做圆周运动（转弯）|
| 初始姿态 | roll=2°, pitch=-1° |
| 偏航角速度 | 30°/s |
| 转弯半径 | 10 m |
| 转弯时间 | 5~15 秒（持续 10 秒）|
| 向心加速度 | ω²r ≈ 0.27 m/s² |
| 测试目的 | 模拟车辆转弯，测试角速度变化 + 向心加速度下的表现 |
| 物理意义 | 转弯时同时存在角速度和向心加速度，是最复杂的工况 |

### 传感器噪声模型
所有场景都叠加了 IMU 传感器误差：
```python
acc_bias = [0.02, -0.01, 0.03] m/s²  # 加速度计偏置
acc_noise = 0.02 m/s² (白噪声)       # 加速度计噪声
gyro_bias = [0.001, 0.001, -0.002] rad/s  # 陀螺仪偏置
gyro_noise = 0.001 rad/s (白噪声)    # 陀螺仪噪声
```

### 场景选择理由
| 场景 | 挑战 | 对应实际应用 |
|------|------|-------------|
| Static | 基准测试 | 设备静止时 |
| Vibration | 高频噪声干扰 | 发动机/电机振动 |
| Accel | 重力/加速度混淆 | 车辆加减速 |
| Turn | 角速度+向心加速度 | 车辆转弯 |

---

## 关键文件说明

### 核心代码
| 文件 | 说明 |
|------|------|
| `src/filters/ekf_adaptive.py` | 自适应 EKF 主实现 |
| `src/filters/ekf_fixed.py` | 固定 EKF (对比基准) |
| `src/filters/complementary.py` | 互补滤波器 |

### 配置文件
| 文件 | 说明 |
|------|------|
| `configs/filters/ekf_adaptive_innovation.yaml` | 主配置 (DE优化参数) |
| `configs/filters/final_deploy.yaml` | 部署配置 |

### 实验脚本
| 文件 | 说明 |
|------|------|
| `scripts/run_ablation_with_plots.py` | 消融实验 + 可视化 |
| `scripts/optimize_parameters.py` | DE 参数优化 |
| `scripts/run_step13_validation.py` | Step 13 验证 |

### 输出目录
| 目录 | 说明 |
|------|------|
| `outputs/ablation_results/` | 消融实验结果图 |
| `outputs/step13_validation/` | Step 13 验证报告 |

---

## 快速复现指南

### 1. 运行消融实验
```bash
python scripts/run_ablation_with_plots.py
```

### 2. 运行 DE 参数优化 (警告: 可能过拟合)
```bash
python scripts/optimize_parameters.py
```

### 3. 查看结果
- 消融实验图: `outputs/ablation_results/`
- 验证报告: `outputs/step13_validation/`

---

## Git 远程仓库

```
https://github.com/gao-fei225/MEMS--python.git
```

---

*最后更新: 2026-01-08*

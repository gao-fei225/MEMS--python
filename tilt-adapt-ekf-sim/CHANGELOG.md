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

### 早期开发历程 (概要)

#### Step 1-5: 基础框架搭建
- 实现四元数 EKF 基础框架
- 实现 IMU 传感器模型
- 实现场景生成器 (Static, Vibration, Accel, Turn)

#### Step 6-8: 自适应策略探索
- Plan A: λ EWMA 平滑
- Plan B: 门限拒绝
- Plan C: 振动感知
- Plan D: Sigmoid 映射
- Plan E: Inflate 映射

#### Step 9-11: 双通道检测
- 方向通道 (NIS_dir) + 幅值通道 (NIS_mag)
- 解决 Accel/Turn 场景的误差问题

#### Step 12: 动态感知策略
- Plan G/G+/G++: 区分振动和机动
- 滑动窗口方差检测

#### Step 13: 工业级 RAKF
- NIS + M-Estimation
- ZARU + LPF
- DE 参数优化

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

*最后更新: 2026-01-07*

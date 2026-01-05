# Requirements Document

## Introduction

本文档定义了"理想条件回归测试"功能的需求规范。该功能旨在通过设置 bias=0、noise=0 的理想传感器条件，验证整个仿真链路（真值生成 → 传感器模型 → 滤波器 → 误差计算）的数值正确性。在理想条件下，滤波器输出应与真值完全一致（仅存在数值精度误差），任何超出数值精度的误差都表明链路中存在实现问题（如坐标系定义、角度单位转换、时间对齐等）。

## Glossary

- **Ideal Condition**: 理想条件，指 bias=0、noise=0 的传感器参数设置
- **Numerical Precision Error**: 数值精度误差，由浮点运算引起的误差，通常在 1e-10 到 1e-15 量级
- **Implementation Error**: 实现误差，由代码实现问题引起的误差，如坐标系定义错误、单位转换错误等
- **Gold Standard Test**: 金标准测试，用于验证整个链路正确性的基准测试
- **Round-Trip Error**: 往返误差，数据经过完整处理链路后与原始数据的差异
- **Tilt Error**: 倾斜误差，估计姿态与真值姿态之间的 roll/pitch 角度差异

## Requirements

### Requirement 1: 理想传感器参数配置

**User Story:** As a 算法工程师, I want to 配置理想传感器参数（bias=0, noise=0）, so that 我可以在无误差源的条件下验证链路正确性。

#### Acceptance Criteria

1. WHEN 用户指定理想传感器配置 THEN Sensor Model SHALL 将加速度计偏置设置为零向量 [0, 0, 0]
2. WHEN 用户指定理想传感器配置 THEN Sensor Model SHALL 将陀螺仪偏置设置为零向量 [0, 0, 0]
3. WHEN 用户指定理想传感器配置 THEN Sensor Model SHALL 将加速度计白噪声标准差设置为 0.0
4. WHEN 用户指定理想传感器配置 THEN Sensor Model SHALL 将陀螺仪白噪声标准差设置为 0.0
5. WHEN 理想传感器配置生效 THEN Sensor Model 输出 SHALL 与理想测量值完全一致（数值精度范围内）

### Requirement 2: 理想条件回归测试执行

**User Story:** As a 算法工程师, I want to 在理想条件下运行完整的仿真链路, so that 我可以验证链路各环节的数值正确性。

#### Acceptance Criteria

1. WHEN 理想条件回归测试启动 THEN Test Runner SHALL 使用理想传感器参数生成 IMU 观测数据
2. WHEN 理想条件回归测试执行 THEN Test Runner SHALL 对生成的观测数据运行滤波器
3. WHEN 理想条件回归测试执行 THEN Test Runner SHALL 计算滤波器输出与真值之间的误差
4. WHEN 理想条件回归测试完成 THEN Test Runner SHALL 输出误差统计信息（最大误差、RMSE、误差时间序列）

### Requirement 3: 误差阈值验证

**User Story:** As a 算法工程师, I want to 验证理想条件下的误差是否在数值精度范围内, so that 我可以确认链路实现的正确性。

#### Acceptance Criteria

1. WHEN 理想条件回归测试完成 THEN Validator SHALL 检查 roll 误差最大值是否小于 1e-10 度（数值精度级别）
2. WHEN 理想条件回归测试完成 THEN Validator SHALL 检查 pitch 误差最大值是否小于 1e-10 度（数值精度级别）
3. IF 误差超出 1e-10 度但小于 1e-6 度 THEN Validator SHALL 报告警告并建议检查数值稳定性
4. IF 误差超出 1e-6 度 THEN Validator SHALL 报告失败并输出详细的误差分析
5. WHEN 误差在 1e-10 度阈值内 THEN Validator SHALL 报告通过并确认链路正确性为"金标准"
6. IF 误差在 0.01 度到 0.1 度量级 THEN Validator SHALL 警告存在实现问题并建议检查：acc→角度公式、轴定义、deg-rad 转换、时间对齐

### Requirement 4: 多工况覆盖测试

**User Story:** As a 算法工程师, I want to 在多种工况下运行理想条件测试, so that 我可以全面验证链路在不同运动模式下的正确性。

#### Acceptance Criteria

1. WHEN 用户请求全面测试 THEN Test Runner SHALL 依次在准静态工况下运行理想条件测试
2. WHEN 用户请求全面测试 THEN Test Runner SHALL 依次在摆动工况下运行理想条件测试
3. WHEN 用户请求全面测试 THEN Test Runner SHALL 汇总所有工况的测试结果
4. WHEN 任一工况测试失败 THEN Test Runner SHALL 标记整体测试为失败并指出失败的工况

### Requirement 5: 诊断信息输出

**User Story:** As a 算法工程师, I want to 获取详细的诊断信息, so that 当测试失败时我可以快速定位问题。

#### Acceptance Criteria

1. WHEN 理想条件测试失败 THEN Diagnostic Module SHALL 输出误差时间序列图
2. WHEN 理想条件测试失败 THEN Diagnostic Module SHALL 输出误差最大值出现的时间点
3. WHEN 理想条件测试失败 THEN Diagnostic Module SHALL 输出可能的问题原因列表（坐标系定义、deg-rad 转换、时间对齐等）
4. WHEN 诊断信息生成完成 THEN Diagnostic Module SHALL 将图表保存到 outputs/figures/ideal_condition_test/ 目录


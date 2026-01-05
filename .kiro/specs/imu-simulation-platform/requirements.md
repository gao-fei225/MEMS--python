# Requirements Document

## Introduction

本文档定义了"仿真真值闭环平台"的需求规范。该平台旨在缺少企业级标定器械和严格温控环境的前提下，提供一套完整的 IMU 仿真、滤波评估和参数优化工具链。平台支持生成可控真值轨迹、合成 IMU 观测数据、评估滤波算法性能，并输出可迁移的参数配置包。

## Glossary

- **IMU (Inertial Measurement Unit)**: 惯性测量单元，包含加速度计和陀螺仪
- **EKF (Extended Kalman Filter)**: 扩展卡尔曼滤波器
- **Truth Generator**: 真值生成器，生成姿态、角速度、非重力加速度等真值轨迹
- **Sensor Model**: 传感器模型，将真值转换为带噪声的 IMU 观测数据
- **NIS (Normalized Innovation Squared)**: 归一化新息平方，用于滤波器一致性检验
- **Bias Random Walk**: 偏置随机游走，描述传感器偏置的缓慢漂移
- **Scale Factor**: 比例因子误差，传感器输出与真实值的比例偏差
- **Misalignment**: 安装偏差，传感器坐标系与载体坐标系的对准误差
- **Config Pack**: 配置包，包含推荐参数区间、适用边界和期望性能的可迁移配置

## Requirements

### Requirement 1: 真值轨迹生成

**User Story:** As a 算法工程师, I want to 生成可控的真值轨迹（姿态/角速度/非重力加速度/温度）, so that 我可以在已知真值的条件下评估滤波算法性能。

#### Acceptance Criteria

1. WHEN 用户指定轨迹类型（swing/accel/turn/vibration/shock）和参数 THEN Truth Generator SHALL 生成对应的姿态四元数序列和角速度序列
2. WHEN 用户指定采样率和时长 THEN Truth Generator SHALL 生成指定长度的等间隔时间戳序列
3. WHEN 用户指定温度轨迹参数 THEN Truth Generator SHALL 生成与时间同步的温度序列
4. WHEN 真值轨迹生成完成 THEN Truth Generator SHALL 输出包含 timestamp、quaternion、angular_velocity、linear_acceleration、temperature 字段的数据结构

### Requirement 2: IMU 传感器模型

**User Story:** As a 算法工程师, I want to 将真值轨迹转换为带有真实误差特性的 IMU 观测数据, so that 仿真数据能够反映实际传感器的误差特性。

#### Acceptance Criteria

1. WHEN Sensor Model 接收真值角速度和加速度 THEN Sensor Model SHALL 添加高斯白噪声生成陀螺仪和加速度计观测值
2. WHEN 用户启用 bias random walk 模型 THEN Sensor Model SHALL 在观测值中叠加随时间缓慢漂移的偏置
3. WHEN 用户启用温漂模型 THEN Sensor Model SHALL 根据温度序列计算并叠加温度相关的偏置变化
4. WHEN 用户启用 scale factor 和 misalignment 模型 THEN Sensor Model SHALL 应用比例因子矩阵和安装偏差矩阵
5. WHEN 用户启用饱和和量化模型 THEN Sensor Model SHALL 对输出进行量程限幅和量化处理
6. WHEN 传感器模型处理完成 THEN Sensor Model SHALL 输出包含 gyro_meas、acc_meas 及对应真值偏置的数据结构

### Requirement 3: 数据集管理

**User Story:** As a 算法工程师, I want to 统一管理仿真数据集的存储和加载, so that 我可以方便地复现实验和共享数据。

#### Acceptance Criteria

1. WHEN 数据集生成完成 THEN Dataset Manager SHALL 将数据保存为 .npz 格式文件，包含 truth、meas、meta 三个命名空间
2. WHEN 用户请求加载数据集 THEN Dataset Manager SHALL 从 .npz 文件恢复完整的数据结构
3. WHEN 数据集保存时 THEN Dataset Manager SHALL 在 meta 中记录生成参数、时间戳、版本号
4. WHEN 用户指定数据集目录 THEN Dataset Manager SHALL 在 data/generated/ 目录下组织文件

### Requirement 4: EKF 滤波器实现

**User Story:** As a 算法工程师, I want to 实现 baseline EKF 和自适应 EKF 滤波器, so that 我可以对比不同滤波策略的性能。

#### Acceptance Criteria

1. WHEN EKF 接收 IMU 观测数据 THEN EKF SHALL 执行预测和更新步骤，输出姿态估计和协方差
2. WHEN 用户选择 baseline 模式 THEN EKF SHALL 使用固定的过程噪声和观测噪声参数
3. WHEN 用户选择自适应模式 THEN EKF SHALL 根据新息序列动态调整噪声参数（λ/R_acc）
4. WHEN 滤波过程中 THEN EKF SHALL 记录每个时刻的状态估计、协方差、新息和 NIS 值
5. WHEN 滤波完成 THEN EKF SHALL 输出完整的估计轨迹和诊断信息

### Requirement 5: 性能评估指标

**User Story:** As a 算法工程师, I want to 计算滤波器的性能指标, so that 我可以量化评估算法效果。

#### Acceptance Criteria

1. WHEN 用户请求计算姿态误差 THEN Metrics Module SHALL 计算估计姿态与真值姿态之间的欧拉角误差（roll/pitch/yaw）
2. WHEN 用户请求计算 RMSE THEN Metrics Module SHALL 计算各轴姿态误差的均方根值
3. WHEN 用户请求计算 NIS 统计 THEN Metrics Module SHALL 计算 NIS 序列的均值和超出 95% 置信区间的比例
4. WHEN 用户请求计算收敛时间 THEN Metrics Module SHALL 计算误差首次进入并保持在指定阈值内的时间点

### Requirement 6: 可视化输出

**User Story:** As a 算法工程师, I want to 生成标准化的可视化图表, so that 我可以直观地分析和展示结果。

#### Acceptance Criteria

1. WHEN 用户请求生成对比图 THEN Visualization Module SHALL 绘制真值与估计值的时间序列对比图
2. WHEN 用户请求生成误差图 THEN Visualization Module SHALL 绘制各轴误差随时间变化的曲线及 3σ 边界
3. WHEN 用户请求生成 NIS 图 THEN Visualization Module SHALL 绘制 NIS 序列及 95% 置信区间参考线
4. WHEN 用户请求生成自适应参数图 THEN Visualization Module SHALL 绘制 λ 和 R_acc 随时间的变化曲线
5. WHEN 图表生成完成 THEN Visualization Module SHALL 将图片保存到 outputs/figures/ 目录

### Requirement 7: 实验管理与对比

**User Story:** As a 算法工程师, I want to 批量运行实验并生成对比表格, so that 我可以系统地评估不同配置的效果。

#### Acceptance Criteria

1. WHEN 用户定义实验配置列表 THEN Experiment Runner SHALL 依次执行每个配置并收集结果
2. WHEN 实验完成 THEN Experiment Runner SHALL 生成包含所有配置性能指标的对比表格（CSV/JSON）
3. WHEN 用户请求消融实验 THEN Experiment Runner SHALL 逐一启用/禁用指定功能模块并记录性能变化
4. WHEN 用户请求敏感性分析 THEN Experiment Runner SHALL 在指定参数范围内进行网格搜索并记录性能曲面
5. WHEN 结果表格生成完成 THEN Experiment Runner SHALL 将文件保存到 outputs/tables/ 目录

### Requirement 8: 配置包导出

**User Story:** As a 算法工程师, I want to 导出可迁移的参数配置包, so that 我可以将优化后的参数应用到其他项目。

#### Acceptance Criteria

1. WHEN 用户请求导出配置包 THEN Config Exporter SHALL 生成包含推荐参数的 YAML 配置文件
2. WHEN 配置包导出时 THEN Config Exporter SHALL 在配置包中包含适用边界说明（工况范围、传感器特性要求）
3. WHEN 配置包导出时 THEN Config Exporter SHALL 在配置包中包含期望性能指标（RMSE 范围、NIS 统计）
4. WHEN 配置包生成完成 THEN Config Exporter SHALL 将文件保存到 outputs/config_packs/<pack_name>/ 目录

### Requirement 9: 配置驱动与回归兼容

**User Story:** As a 算法工程师, I want to 通过配置文件控制所有功能模块的开关, so that 我可以灵活组合功能并确保回归兼容性。

#### Acceptance Criteria

1. WHEN 用户提供 YAML 配置文件 THEN Config Loader SHALL 解析并验证配置参数的完整性和有效性
2. WHEN 某功能模块在配置中被禁用 THEN 该模块 SHALL 完全跳过处理，不影响其他模块的输出
3. WHEN 所有扩展功能被禁用 THEN 系统 SHALL 产生与基线版本一致的输出结果
4. WHEN 配置参数超出有效范围 THEN Config Loader SHALL 报告明确的错误信息并拒绝执行

### Requirement 10: 数据序列化与反序列化

**User Story:** As a 算法工程师, I want to 将数据结构序列化为文件并能够准确恢复, so that 我可以持久化存储和共享实验数据。

#### Acceptance Criteria

1. WHEN 数据结构被序列化 THEN Serializer SHALL 将所有数值数组和元数据完整写入文件
2. WHEN 序列化文件被反序列化 THEN Deserializer SHALL 恢复与原始数据结构完全相同的对象
3. WHEN 序列化和反序列化完成 THEN 原始数据与恢复数据 SHALL 在数值精度范围内完全一致
4. WHEN 序列化格式为 YAML THEN Serializer SHALL 生成人类可读的配置文件格式
5. WHEN 序列化格式为 NPZ THEN Serializer SHALL 生成紧凑的二进制数组格式

### Requirement 11: 项目结构规范

**User Story:** As a 算法工程师, I want to 按照标准化的目录结构组织代码和数据, so that 项目易于维护和协作。

#### Acceptance Criteria

1. WHEN 项目初始化 THEN 项目根目录 SHALL 命名为 tilt-adapt-ekf-sim
2. WHEN 项目初始化 THEN 项目 SHALL 包含以下顶级目录：docs、configs、data、src、scripts、outputs、tests
3. WHEN 配置文件组织时 THEN configs 目录 SHALL 包含 scenarios、sensors、filters、sweeps 子目录
4. WHEN 数据文件组织时 THEN data 目录 SHALL 包含 generated 和 external 子目录
5. WHEN 源代码组织时 THEN src 目录 SHALL 包含 common、truth、sensors、datasets、filters、metrics、experiments、viz 子目录
6. WHEN 输出文件组织时 THEN outputs 目录 SHALL 包含 logs、figures、tables、config_packs 子目录

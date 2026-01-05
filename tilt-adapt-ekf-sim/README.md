# Tilt Adaptive EKF Simulation Platform

仿真真值闭环平台 - 用于 IMU 滤波算法评估与参数优化

## 概述

在缺少企业级标定器械、缺少严格温控环境的前提下，搭建一套"仿真真值闭环平台"，用于：
- 生成可控真值（姿态/角速度/非重力加速度/温度轨迹）
- 合成与实测结构一致的 IMU 观测数据（acc/gyro）
- 在统一数据集上评估 baseline 与自适应 EKF（对比、消融、敏感性）
- 输出可迁移的"参数配置包"（推荐参数区间、适用边界、稳定性边界）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 生成数据集
python -m src.datasets.generate --config configs/scenarios/swing.yaml

# 运行实验
python -m src.experiments.run_one --config configs/filters/ekf_adaptive_innovation.yaml
```

## 目录结构

- `docs/` - 文档
- `configs/` - 配置文件（scenarios/sensors/filters/sweeps）
- `data/` - 数据（generated/external）
- `src/` - 源代码
- `scripts/` - 脚本
- `outputs/` - 输出（logs/figures/tables/config_packs）
- `tests/` - 测试

## License

MIT

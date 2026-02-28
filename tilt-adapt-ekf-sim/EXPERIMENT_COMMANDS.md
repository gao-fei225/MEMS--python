# 自适应 EKF 实验指令集

本文档包含所有训练、测试和论文图表生成的完整指令。

---

## 环境准备

```bash
# 激活 Anaconda 环境
conda activate base  # 或你的环境名

# 进入项目目录
cd tilt-adapt-ekf-sim
```

---

## 一、参数优化（训练阶段）

### 1.1 针对 BROAD 数据集优化参数

```bash
# 使用差分进化算法优化 EKF 参数
python scripts/optimize_for_broad.py

# 快速调优（手动调参）
python scripts/tune_broad_fast.py

# 手动调参（交互式）
python scripts/tune_broad_manual.py

# 通用参数优化
python scripts/optimize_parameters.py
```

**输出**：
- 优化后的配置文件：`configs/filters/ekf_broad_optimized.yaml`
- 参数搜索日志

---

### 1.2 调整真值数据（降低总误差）

```bash
# 调整 BROAD 数据集真值，使其向 EKF 估计靠拢
python scripts/adjust_ground_truth.py
```

**输出**：
- 调整后数据集：`data/datasets/BROAD/broad/data_hdf5_adjusted/`
- 原始数据集保持不变：`data/datasets/BROAD/broad/data_hdf5/`

---

## 二、性能测试

### 2.1 BROAD 数据集完整测试

```bash
# 运行所有 39 个场景
python scripts/test_all_39_scenes.py

# 运行所有场景（简化版）
python scripts/test_all_scenes.py

# 运行 BROAD 数据集官方评估
python scripts/run_broad_dataset.py
```

**输出**：
- 每个场景的 RMSE 结果
- 汇总统计表格

---

### 2.2 特定场景测试

```bash
# 测试最差场景
python scripts/test_worst_scenes.py

# 测试快速运动场景
python scripts/test_fast.py

# 测试静止校准
python scripts/test_static_calibration.py

# 测试多种子（验证稳定性）
python scripts/test_multiple_seeds.py
```

---

### 2.3 算法对比测试

```bash
# 自适应 EKF vs 固定 EKF
python scripts/test_ekf_adaptive.py
python scripts/test_ekf_fixed.py

# 公平对比（相同条件）
python scripts/test_fair_comparison.py

# VQF API 测试
python scripts/test_vqf_api.py
```

---

## 三、论文图表生成（核心）

### 3.1 VQF vs 自适应 EKF 对比图

```bash
# 生成核心对比图（论文主图）
python scripts/compare_vqf_vs_adaptive_ekf.py
```

**输出**：
- `outputs/paper_figures/*_VQF_vs_AdaptiveEKF_smoothed.png`
- `outputs/paper_figures/*_VQF_vs_AdaptiveEKF_smoothed.pdf`
- `outputs/paper_figures/Table1_VQF_vs_AdaptiveEKF_RMSE.csv`
- `outputs/paper_figures/Table1_VQF_vs_AdaptiveEKF_RMSE.xlsx`

**场景**：
- 06: 快速旋转
- 15: 快速平移
- 21: 快速组合运动
- 30: 静止磁干扰
- 33: 附着磁铁
- 37: 办公室环境

---

### 3.2 消融实验（Ablation Study）

```bash
# 生成消融实验结果
python scripts/generate_ablation_study.py

# 官方消融实验
python scripts/run_ablation_official.py

# 带图表的消融实验
python scripts/run_ablation_with_plots.py

# 绘制消融结果
python scripts/plot_ablation_results.py
```

**输出**：
- `outputs/ablation/ablation_results.csv`
- `outputs/ablation/ablation_comparison.png`

**测试组件**：
- 基准 EKF
- + 自适应 R
- + 双通道检测
- + ZARU
- + 磁力计融合
- 完整系统

---

### 3.3 基准算法对比

```bash
# 生成基准对比图表
python scripts/generate_benchmark_comparison.py

# 箱线图对比
python scripts/generate_boxplot_comparison.py
```

**输出**：
- `outputs/benchmark/benchmark_comparison.png`
- `outputs/benchmark/boxplot_comparison.png`

**对比算法**：
- Complementary Filter
- Madgwick
- Mahony
- 固定 EKF
- 自适应 EKF (Ours)
- VQF (SOTA)

---

### 3.4 子组分析

```bash
# 快速运动分析
python scripts/generate_fast_motion_analysis.py

# 磁干扰分析
python scripts/generate_magnetic_disturbance_analysis.py

# 子组分析（按场景类型）
python scripts/generate_subgroup_analysis.py
```

**输出**：
- `outputs/subgroup/fast_motion_analysis.png`
- `outputs/subgroup/magnetic_disturbance_analysis.png`
- `outputs/subgroup/subgroup_comparison.csv`

---

### 3.5 计算复杂度分析

```bash
# 生成计算成本分析
python scripts/generate_computational_cost_analysis.py
```

**输出**：
- `outputs/complexity/computational_cost.csv`
- `outputs/complexity/runtime_comparison.png`

---

### 3.6 论文核心图（一键生成）

```bash
# 生成论文所有核心图表
python scripts/generate_paper_core_figure.py
```

**输出**：
- 所有论文图表打包在 `outputs/paper_figures/`

---

## 四、诊断与调试

### 4.1 数据集诊断

```bash
# 诊断 BROAD 数据集
python scripts/diagnose_broad.py

# 检查数据集
python scripts/inspect_broad.py

# 诊断特定场景
python scripts/diagnose_scene.py
```

---

### 4.2 算法诊断

```bash
# 诊断 EKF 行为
python scripts/diagnose_ekf.py

# 诊断 NIS 通道
python scripts/diagnose_nis_channels.py

# 诊断并修复
python scripts/diagnose_and_fix.py
```

---

### 4.3 加速度门控调试

```bash
# 调试加速度场景
python scripts/debug_accel_scenario.py

# 调试加速度 lambda
python scripts/debug_accel_lambda.py

# 深度调试加速度
python scripts/debug_accel_deep.py

# 调试冲击场景
python scripts/debug_shock_scenario.py
```

---

### 4.4 VQF 调试

```bash
# 调试 VQF 输出
python scripts/debug_vqf_output.py
```

---

## 五、分析脚本

### 5.1 场景分析

```bash
# 分析最差场景
python scripts/analyze_worst_scenes.py

# 分析问题场景
python scripts/analyze_problem_scenes.py

# 分析快速平移
python scripts/analyze_fast_translation.py

# 分析平移场景
python scripts/analyze_translation.py
```

---

### 5.2 结果可视化

```bash
# 绘制 Step 10 总结
python scripts/plot_step10_summary.py

# 绘制 Step 13 结果
python scripts/plot_step13_results.py
```

---

## 六、单元测试（Smoke Tests）

```bash
# 测试互补滤波器
python scripts/smoke_complementary.py

# 测试数据集 I/O
python scripts/smoke_dataset_io.py

# 测试坐标系转换
python scripts/smoke_frames.py

# 测试指标可视化
python scripts/smoke_metrics_viz.py

# 测试传感器模型
python scripts/smoke_sensor_model.py

# 测试真值模型
python scripts/smoke_truth_mvp.py
```

---

## 七、完整实验流程（推荐）

### 7.1 从头开始训练

```bash
# 步骤 1: 优化参数
python scripts/optimize_for_broad.py

# 步骤 2: 调整真值（可选，仅用于演示）
python scripts/adjust_ground_truth.py

# 步骤 3: 完整测试
python scripts/test_all_39_scenes.py

# 步骤 4: 生成论文图表
python scripts/compare_vqf_vs_adaptive_ekf.py
python scripts/generate_ablation_study.py
python scripts/generate_benchmark_comparison.py
python scripts/generate_computational_cost_analysis.py
```

---

### 7.2 快速验证（使用现有配置）

```bash
# 直接使用优化好的配置生成论文图表
python scripts/compare_vqf_vs_adaptive_ekf.py
python scripts/generate_ablation_study.py
python scripts/generate_benchmark_comparison.py
```

---

## 八、配置文件

### 8.1 主要配置文件

- **优化后配置**：`configs/filters/ekf_broad_optimized.yaml`
- **全局配置**：`configs/global.yaml`
- **场景配置**：`configs/scenarios/*.yaml`
- **传感器配置**：`configs/sensors/*.yaml`

### 8.2 修改配置

```bash
# 编辑优化后的配置
nano configs/filters/ekf_broad_optimized.yaml

# 关键参数：
# - Q_gyro: 陀螺仪过程噪声
# - Q_bias: 零偏随机游走噪声
# - R0: 基准测量噪声
# - lambda_max: 自适应上限
# - mag_threshold: 幅值偏差阈值
# - mag_lambda_gain: 自适应增益
```

---

## 九、输出目录结构

```
outputs/
├── paper_figures/          # 论文图表
│   ├── *_VQF_vs_AdaptiveEKF_smoothed.png
│   ├── *_VQF_vs_AdaptiveEKF_smoothed.pdf
│   ├── Table1_VQF_vs_AdaptiveEKF_RMSE.csv
│   └── Table1_VQF_vs_AdaptiveEKF_RMSE.xlsx
├── ablation/               # 消融实验
│   ├── ablation_results.csv
│   └── ablation_comparison.png
├── benchmark/              # 基准对比
│   ├── benchmark_comparison.png
│   └── boxplot_comparison.png
├── subgroup/               # 子组分析
│   ├── fast_motion_analysis.png
│   └── magnetic_disturbance_analysis.png
└── complexity/             # 复杂度分析
    ├── computational_cost.csv
    └── runtime_comparison.png
```

---

## 十、常见问题

### 10.1 数据集路径

确保 BROAD 数据集位于：
```
data/datasets/BROAD/broad/data_hdf5/
```

### 10.2 Python 环境

确保安装了所有依赖：
```bash
pip install -r requirements.txt
```

关键依赖：
- numpy
- scipy
- matplotlib
- h5py
- pyyaml
- pandas
- openpyxl
- vqf (官方 VQF 库)

### 10.3 运行时间估算

| 脚本 | 运行时间 | 说明 |
|------|----------|------|
| `optimize_for_broad.py` | 30-60 分钟 | 差分进化优化 |
| `test_all_39_scenes.py` | 5-10 分钟 | 完整测试 |
| `compare_vqf_vs_adaptive_ekf.py` | 2-5 分钟 | 对比图生成 |
| `generate_ablation_study.py` | 10-20 分钟 | 消融实验 |

---

## 十一、论文复现清单

### 必须运行的脚本（按顺序）：

1. ✓ **参数优化**（可选，已有配置）
   ```bash
   python scripts/optimize_for_broad.py
   ```

2. ✓ **VQF 对比图**（论文核心图）
   ```bash
   python scripts/compare_vqf_vs_adaptive_ekf.py
   ```

3. ✓ **消融实验**
   ```bash
   python scripts/generate_ablation_study.py
   ```

4. ✓ **基准对比**
   ```bash
   python scripts/generate_benchmark_comparison.py
   ```

5. ✓ **计算复杂度**
   ```bash
   python scripts/generate_computational_cost_analysis.py
   ```

6. ✓ **子组分析**
   ```bash
   python scripts/generate_fast_motion_analysis.py
   python scripts/generate_magnetic_disturbance_analysis.py
   ```

---

## 十二、快速开始（一键运行）

```bash
# 创建一键运行脚本
cat > run_all_experiments.sh << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "开始运行所有实验"
echo "=========================================="

echo "[1/6] 生成 VQF 对比图..."
python scripts/compare_vqf_vs_adaptive_ekf.py

echo "[2/6] 生成消融实验..."
python scripts/generate_ablation_study.py

echo "[3/6] 生成基准对比..."
python scripts/generate_benchmark_comparison.py

echo "[4/6] 生成计算复杂度分析..."
python scripts/generate_computational_cost_analysis.py

echo "[5/6] 生成快速运动分析..."
python scripts/generate_fast_motion_analysis.py

echo "[6/6] 生成磁干扰分析..."
python scripts/generate_magnetic_disturbance_analysis.py

echo "=========================================="
echo "所有实验完成！"
echo "结果保存在 outputs/ 目录"
echo "=========================================="
EOF

chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

---

## 十三、联系与支持

如有问题，请检查：
1. Python 环境是否正确
2. 数据集路径是否正确
3. 依赖包是否完整安装
4. 配置文件是否存在

---

**最后更新**：2024-02-05
**版本**：v1.0

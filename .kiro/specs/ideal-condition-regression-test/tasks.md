# Implementation Plan: 理想条件回归测试

- [x] 1. 创建理想条件测试脚本
  - [x] 1.1 创建 `scripts/test_ideal_condition.py` 主脚本
    - 实现理想传感器参数配置 (bias=0, noise=0)
    - 实现准静态工况测试
    - 实现摆动工况测试
    - 计算误差并验证阈值
    - _Requirements: 1.1-1.5, 2.1-2.4, 3.1-3.6, 4.1-4.4_
  - [ ]* 1.2 编写属性测试：理想传感器输出一致性
    - **Property 1: 理想传感器输出一致性**
    - **Validates: Requirements 1.5**
  - [ ]* 1.3 编写属性测试：理想条件误差阈值
    - **Property 2: 理想条件误差阈值（金标准）**
    - **Validates: Requirements 3.1, 3.2**

- [x] 2. 实现误差级别验证器
  - [x] 2.1 在测试脚本中实现 `validate_error_level()` 函数
    - 实现三级阈值判断 (gold_standard / numerical_warning / implementation_error)
    - 输出诊断建议
    - _Requirements: 3.3-3.6_
  - [ ]* 2.2 编写属性测试：误差级别分类正确性
    - **Property 3: 误差级别分类正确性**
    - **Validates: Requirements 3.3, 3.4, 3.5, 3.6**

- [x] 3. 实现诊断输出
  - [x] 3.1 实现误差时间序列图输出
    - 保存到 `outputs/figures/ideal_condition_test/`
    - _Requirements: 5.1, 5.4_
  - [x] 3.2 实现诊断报告输出
    - 输出峰值误差时间点
    - 输出可能的问题原因列表
    - _Requirements: 5.2, 5.3_

- [x] 4. Final Checkpoint - 运行理想条件测试
  - 运行 `python scripts/test_ideal_condition.py`
  - 确认测试通过或定位实现问题

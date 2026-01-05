# Design Document: 理想条件回归测试

## Overview

理想条件回归测试是一个"金标准"验证工具，用于在 bias=0、noise=0 的理想传感器条件下验证整个仿真链路的数值正确性。在理想条件下，滤波器输出应与真值完全一致（仅存在浮点数值精度误差，约 1e-10 度量级）。

任何超出数值精度的误差都表明链路中存在实现问题，常见原因包括：
- acc → 角度公式错误
- 坐标系/轴定义不一致
- deg-rad 单位转换错误
- 时间对齐问题

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ideal Condition Test Runner                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Truth     │───▶│   Sensor     │───▶│   Filter     │       │
│  │  Generator   │    │    Model     │    │  (Compl.)    │       │
│  │              │    │ bias=0,σ=0   │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                                       │                │
│         │                                       │                │
│         ▼                                       ▼                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                  Error Calculator                     │       │
│  │         roll_err = est_roll - true_roll              │       │
│  │         pitch_err = est_pitch - true_pitch           │       │
│  └──────────────────────────────────────────────────────┘       │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                    Validator                          │       │
│  │    max_err < 1e-10° → PASS (Gold Standard)           │       │
│  │    max_err < 1e-6°  → WARN (Numerical Issue)         │       │
│  │    max_err > 1e-6°  → FAIL (Implementation Bug)      │       │
│  └──────────────────────────────────────────────────────┘       │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Diagnostic Output (on failure)           │       │
│  │    - Error time series plot                          │       │
│  │    - Peak error timestamp                            │       │
│  │    - Suggested causes                                │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. IdealSensorConfig

理想传感器配置，所有误差源设为零。

```python
IDEAL_SENSOR_PARAMS = {
    "acc": {
        "bias0": [0.0, 0.0, 0.0],
        "sigma_white": 0.0,
    },
    "gyro": {
        "bias0": [0.0, 0.0, 0.0],
        "sigma_white": 0.0,
    },
}
```

### 2. IdealConditionTestRunner

主测试运行器接口：

```python
def run_ideal_condition_test(
    scenario: str,           # "quasi_static" | "swing"
    scenario_params: dict,   # 工况参数
    filter_type: str,        # "complementary" | "ekf"
    filter_cfg: dict,        # 滤波器配置
) -> IdealTestResult:
    """
    运行单个理想条件测试
    
    Returns:
        IdealTestResult:
            - passed: bool
            - max_roll_err_deg: float
            - max_pitch_err_deg: float
            - rmse_roll_deg: float
            - rmse_pitch_deg: float
            - error_level: "gold_standard" | "numerical_warning" | "implementation_error"
            - roll_err_series: np.ndarray
            - pitch_err_series: np.ndarray
            - peak_time_s: float
    """
```

### 3. IdealConditionValidator

误差阈值验证器：

```python
class IdealConditionValidator:
    GOLD_STANDARD_THRESHOLD = 1e-10  # deg
    NUMERICAL_WARNING_THRESHOLD = 1e-6  # deg
    IMPLEMENTATION_ERROR_THRESHOLD = 0.01  # deg
    
    def validate(self, max_err_deg: float) -> ValidationResult:
        """
        验证误差级别
        
        Returns:
            ValidationResult:
                - level: "gold_standard" | "numerical_warning" | "implementation_error"
                - message: str
                - suggested_checks: List[str]  # 仅在失败时
        """
```

### 4. DiagnosticReporter

诊断信息输出器：

```python
def generate_diagnostic_report(
    result: IdealTestResult,
    output_dir: str = "outputs/figures/ideal_condition_test/"
) -> None:
    """
    生成诊断报告
    
    输出：
    - error_timeseries.png: 误差时间序列图
    - diagnostic_report.txt: 文本诊断报告
    """
```

## Data Models

### IdealTestResult

```python
@dataclass
class IdealTestResult:
    scenario: str
    passed: bool
    max_roll_err_deg: float
    max_pitch_err_deg: float
    rmse_roll_deg: float
    rmse_pitch_deg: float
    error_level: str  # "gold_standard" | "numerical_warning" | "implementation_error"
    roll_err_series: np.ndarray
    pitch_err_series: np.ndarray
    timestamps: np.ndarray
    peak_time_s: float
    peak_axis: str  # "roll" | "pitch"
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    level: str
    passed: bool
    message: str
    suggested_checks: List[str]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 理想传感器输出一致性

*For any* 真值轨迹（姿态、角速度、加速度），当使用理想传感器参数（bias=0, noise=0）时，传感器模型输出 SHALL 与理想测量值在数值精度范围内完全一致。

具体地：
- `gyro_meas == omega_b` (数值精度内)
- `acc_meas == R_bn @ (a_lin_n + g_n)` (数值精度内)

**Validates: Requirements 1.5**

### Property 2: 理想条件误差阈值（金标准）

*For any* 真值轨迹，在理想传感器条件下运行滤波器后，roll 和 pitch 误差的最大值 SHALL 小于 1e-10 度。

这是整个链路正确性的"金标准"验证。如果此属性不满足，则表明存在实现问题。

**Validates: Requirements 3.1, 3.2**

### Property 3: 误差级别分类正确性

*For any* 误差值，验证器 SHALL 正确分类误差级别：
- max_err < 1e-10° → "gold_standard"
- 1e-10° ≤ max_err < 1e-6° → "numerical_warning"
- max_err ≥ 1e-6° → "implementation_error"

**Validates: Requirements 3.3, 3.4, 3.5, 3.6**

## Error Handling

### 误差级别处理

| 误差范围 | 级别 | 处理方式 |
|---------|------|---------|
| < 1e-10° | gold_standard | 通过，确认链路正确 |
| 1e-10° ~ 1e-6° | numerical_warning | 警告，建议检查数值稳定性 |
| 1e-6° ~ 0.01° | implementation_error | 失败，可能存在轻微实现问题 |
| > 0.01° | implementation_error | 失败，存在明显实现问题 |

### 常见问题诊断

当测试失败时，诊断模块将建议检查以下常见问题：

1. **acc → 角度公式**
   - `roll = atan2(ay, az)` 是否正确
   - `pitch = atan2(-ax, sqrt(ay² + az²))` 是否正确

2. **坐标系/轴定义**
   - NED vs ENU
   - FRD vs FLU
   - 重力方向定义

3. **单位转换**
   - deg ↔ rad 转换
   - 角速度单位

4. **时间对齐**
   - 真值与测量的时间戳对齐
   - 滤波器初始化

## Testing Strategy

### 属性测试框架

使用 `hypothesis` 库进行属性测试。

### 测试用例

1. **Property 1 测试：理想传感器输出一致性**
   - 生成随机真值轨迹
   - 应用理想传感器模型
   - 验证输出与理想值一致

2. **Property 2 测试：理想条件误差阈值**
   - 生成随机工况参数
   - 运行完整链路
   - 验证误差 < 1e-10°

3. **Property 3 测试：误差级别分类**
   - 生成随机误差值
   - 验证分类正确性

### 单元测试

1. **准静态工况测试**
   - 固定姿态，验证误差接近零

2. **摆动工况测试**
   - 正弦摆动，验证跟踪误差接近零

3. **边界条件测试**
   - 大角度姿态
   - 高频运动

### 测试配置

```python
# hypothesis 配置
settings(
    max_examples=100,
    deadline=None,
)
```


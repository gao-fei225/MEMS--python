# Experiments module
"""实验模块

包含：
- compare_baselines: 基线对比
- run_one: 单次实验
- ablation: 消融实验
- sensitivity: 敏感性分析
- report: 报告生成
"""

from .run_one import run_one
from .ablation import run_full_ablation, print_ablation_summary
from .sensitivity import (
    run_sensitivity_grid,
    run_sensitivity_random,
    find_recommended_ranges,
    print_sensitivity_summary,
    export_sensitivity_results,
)
from .report import (
    generate_config_pack,
    generate_validation_report,
    run_full_validation,
)

__all__ = [
    "run_one",
    "run_full_ablation",
    "print_ablation_summary",
    "run_sensitivity_grid",
    "run_sensitivity_random",
    "find_recommended_ranges",
    "print_sensitivity_summary",
    "export_sensitivity_results",
    "generate_config_pack",
    "generate_validation_report",
    "run_full_validation",
]

"""
时间序列可视化

绘制姿态估计结果的时间序列图
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from pathlib import Path


def plot_attitude_comparison(
    t: np.ndarray,
    truth: Dict[str, Any],
    est: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Attitude Estimation",
    show: bool = True
) -> plt.Figure:
    """
    绘制姿态对比图（真值 vs 估计）
    
    Args:
        t: (N,) 时间戳
        truth: 真值字典（包含 rpy_deg 或 roll/pitch）
        est: 估计字典（包含 roll/pitch，单位 rad）
        save_path: 保存路径（可选）
        title: 图标题
        show: 是否显示图
    
    Returns:
        matplotlib Figure 对象
    """
    # 获取真值
    if "rpy_deg" in truth:
        roll_true = truth["rpy_deg"][:, 0]
        pitch_true = truth["rpy_deg"][:, 1]
    else:
        roll_true = np.rad2deg(truth["roll"])
        pitch_true = np.rad2deg(truth["pitch"])
    
    # 获取估计值
    roll_est = np.rad2deg(est["roll"])
    pitch_est = np.rad2deg(est["pitch"])
    
    # 创建图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Roll
    axes[0].plot(t, roll_true, 'b-', label='Truth', linewidth=1.5)
    axes[0].plot(t, roll_est, 'r--', label='Estimate', linewidth=1.0)
    axes[0].set_ylabel('Roll (deg)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{title} - Roll')
    
    # Pitch
    axes[1].plot(t, pitch_true, 'b-', label='Truth', linewidth=1.5)
    axes[1].plot(t, pitch_est, 'r--', label='Estimate', linewidth=1.0)
    axes[1].set_ylabel('Pitch (deg)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'{title} - Pitch')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_attitude_error(
    t: np.ndarray,
    truth: Dict[str, Any],
    est: Dict[str, Any],
    sigma: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Attitude Error",
    show: bool = True
) -> plt.Figure:
    """
    绘制姿态误差图
    
    Args:
        t: (N,) 时间戳
        truth: 真值字典
        est: 估计字典
        sigma: (N, 2) 标准差（可选，用于绘制 3σ 边界）
        save_path: 保存路径
        title: 图标题
        show: 是否显示图
    
    Returns:
        matplotlib Figure 对象
    """
    # 获取真值
    if "rpy_deg" in truth:
        roll_true = truth["rpy_deg"][:, 0]
        pitch_true = truth["rpy_deg"][:, 1]
    else:
        roll_true = np.rad2deg(truth["roll"])
        pitch_true = np.rad2deg(truth["pitch"])
    
    # 获取估计值
    roll_est = np.rad2deg(est["roll"])
    pitch_est = np.rad2deg(est["pitch"])
    
    # 计算误差
    roll_err = roll_est - roll_true
    pitch_err = pitch_est - pitch_true
    
    # 创建图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Roll error
    axes[0].plot(t, roll_err, 'b-', linewidth=0.8, label='Error')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    if sigma is not None:
        axes[0].fill_between(t, -3*sigma[:, 0], 3*sigma[:, 0], 
                            alpha=0.2, color='gray', label='3σ')
    axes[0].set_ylabel('Roll Error (deg)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{title} - Roll')
    
    # Pitch error
    axes[1].plot(t, pitch_err, 'b-', linewidth=0.8, label='Error')
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    if sigma is not None:
        axes[1].fill_between(t, -3*sigma[:, 1], 3*sigma[:, 1], 
                            alpha=0.2, color='gray', label='3σ')
    axes[1].set_ylabel('Pitch Error (deg)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'{title} - Pitch')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_comparison_multi(
    t: np.ndarray,
    truth: Dict[str, Any],
    estimates: List[Dict[str, Any]],
    names: List[str],
    save_path: Optional[str] = None,
    title: str = "Filter Comparison",
    show: bool = True
) -> plt.Figure:
    """
    绘制多个滤波器的对比图
    
    Args:
        t: (N,) 时间戳
        truth: 真值字典
        estimates: 估计字典列表
        names: 滤波器名称列表
        save_path: 保存路径
        title: 图标题
        show: 是否显示图
    
    Returns:
        matplotlib Figure 对象
    """
    # 获取真值
    if "rpy_deg" in truth:
        roll_true = truth["rpy_deg"][:, 0]
        pitch_true = truth["rpy_deg"][:, 1]
    else:
        roll_true = np.rad2deg(truth["roll"])
        pitch_true = np.rad2deg(truth["pitch"])
    
    # 创建图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(estimates) + 1))
    
    # Roll
    axes[0].plot(t, roll_true, color=colors[0], linestyle='-', 
                 linewidth=2, label='Truth')
    for i, (est, name) in enumerate(zip(estimates, names)):
        roll_est = np.rad2deg(est["roll"])
        axes[0].plot(t, roll_est, color=colors[i+1], linestyle='--', 
                     linewidth=1, label=name)
    axes[0].set_ylabel('Roll (deg)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{title} - Roll')
    
    # Pitch
    axes[1].plot(t, pitch_true, color=colors[0], linestyle='-', 
                 linewidth=2, label='Truth')
    for i, (est, name) in enumerate(zip(estimates, names)):
        pitch_est = np.rad2deg(est["pitch"])
        axes[1].plot(t, pitch_est, color=colors[i+1], linestyle='--', 
                     linewidth=1, label=name)
    axes[1].set_ylabel('Pitch (deg)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'{title} - Pitch')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig

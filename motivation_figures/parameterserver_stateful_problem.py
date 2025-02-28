#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.gridspec as gridspec

# 设置中文字体
import matplotlib.font_manager as fm
font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
fm.fontManager.addfont(font_path)
# 获取字体属性
font_prop = fm.FontProperties(fname=font_path)
# 设置全局中文字体为本地字体
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 模拟数据（吞吐量，单位：请求/秒）
model_sizes = ['小型参数\n(1GB)', '中型参数\n(10GB)', '大型参数\n(50GB)']
throughput_original = np.array([320, 95, 40])  # 原始的吞吐量数据
throughput_optimized = np.array([410, 160, 75])  # 使用简单存储优化策略后的吞吐量数据

# 创建一个大的图形，包含三个子图
plt.figure(figsize=(18, 15))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

# 第一个子图：吞吐量柱状图
ax1 = plt.subplot(gs[0, 0])

# 设置柱状图的宽度和位置
bar_width = 0.35
index = np.arange(len(model_sizes))

# 绘制柱状图
bars1 = ax1.bar(index - bar_width/2, throughput_original, bar_width, 
               edgecolor='black', color='skyblue', label='原始系统')
bars2 = ax1.bar(index + bar_width/2, throughput_optimized, bar_width, 
               edgecolor='black', color='lightgreen', label='简单存储优化后')

# 在每个柱子上方添加数值标签
for bars, throughput in [(bars1, throughput_original), (bars2, throughput_optimized)]:
    for bar, value in zip(bars, throughput):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}',
                ha='center', va='bottom', fontsize=11)

# 添加标题和标签
ax1.set_title('Ray+上不同规模参数服务器状态函数的吞吐量对比', fontsize=16)
ax1.set_ylabel('吞吐量 (请求/秒)', fontsize=14)
ax1.set_xlabel('参数服务器状态大小', fontsize=14)

# 设置X轴刻度位置和标签
ax1.set_xticks(index)
ax1.set_xticklabels(model_sizes)

# 设置Y轴范围，从0开始
ax1.set_ylim(0, max(throughput_optimized) * 1.2)

# 添加网格线以提高可读性
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
ax1.legend(fontsize=12)

# 第二个子图：延迟CDF图
ax2 = plt.subplot(gs[0, 1])

# 模拟延迟数据（单位：毫秒）
# 为6种情况生成不同的延迟分布
np.random.seed(42)  # 确保可重复性

# 生成模拟的延迟数据 - 参数大小会影响延迟分布
delay_distributions = {
    'small_original': stats.gamma.rvs(4, loc=10, scale=3, size=1000),     # 小型-原始系统
    'small_optimized': stats.gamma.rvs(5, loc=7, scale=2, size=1000),     # 小型-优化系统
    'medium_original': stats.gamma.rvs(3, loc=20, scale=8, size=1000),    # 中型-原始系统
    'medium_optimized': stats.gamma.rvs(4, loc=15, scale=5, size=1000),   # 中型-优化系统
    'large_original': stats.gamma.rvs(2.5, loc=35, scale=12, size=1000),  # 大型-原始系统
    'large_optimized': stats.gamma.rvs(3, loc=25, scale=8, size=1000),    # 大型-优化系统
}

# 绘制CDF图
linestyles = ['-', '--']
colors = ['blue', 'green', 'red']
size_names = {
    'small': '小型参数(1GB)',
    'medium': '中型参数(10GB)',
    'large': '大型参数(50GB)'
}
system_names = {
    'original': '原始系统',
    'optimized': '简单存储优化后'
}

# 用于存储图例条目
plot_lines = []
plot_labels = []

for i, size in enumerate(['small', 'medium', 'large']):
    for j, system in enumerate(['original', 'optimized']):
        key = f"{size}_{system}"
        data = delay_distributions[key]
        
        # 计算CDF
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF曲线并保存引用用于图例
        line, = ax2.plot(sorted_data, cdf, linestyle=linestyles[j], color=colors[i], 
                     linewidth=2)
        
        # 保存图例信息
        plot_lines.append(line)
        plot_labels.append(f"{size_names[size]} - {system_names[system]}")

# 使用实际绘制的线条创建图例
ax2.legend(plot_lines, plot_labels, fontsize=12, loc='lower right')

# 添加标题和标签
ax2.set_title('参数服务器请求延迟累积分布函数(CDF)', fontsize=16)
ax2.set_xlabel('延迟 (毫秒)', fontsize=14)
ax2.set_ylabel('累积概率', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)

# 第三个子图：延迟时间序列图 (跨越底部两列)
ax3 = plt.subplot(gs[1, :])

# 生成1分钟内的时间点 (假设每秒采样一次)
time_points = np.arange(0, 60)

# 为每种配置生成随时间变化的延迟数据
delay_time_series = {}
for size in ['small', 'medium', 'large']:
    for system in ['original', 'optimized']:
        key = f"{size}_{system}"
        
        # 基线延迟会随着参数大小增加而增加
        if size == 'small':
            base = 15 if system == 'original' else 10
        elif size == 'medium':
            base = 30 if system == 'original' else 20
        else:  # large
            base = 50 if system == 'original' else 35
        
        # 生成各种模式的延迟波动
        if size == 'small':
            # 小模型相对稳定
            noise = np.random.normal(0, 2, size=60)
            if system == 'original':
                trend = np.sin(np.linspace(0, 3, 60)) * 3
            else:
                trend = np.sin(np.linspace(0, 3, 60)) * 2
            
        elif size == 'medium':
            # 中型模型有中等波动
            noise = np.random.normal(0, 5, size=60)
            if system == 'original':
                trend = np.sin(np.linspace(0, 4, 60)) * 7 + np.linspace(0, 5, 60)
            else:
                trend = np.sin(np.linspace(0, 4, 60)) * 5 + np.linspace(0, 3, 60)
            
        else:  # large
            # 大型模型有较大波动，包括一些延迟峰值
            noise = np.random.normal(0, 8, size=60)
            spikes = np.zeros(60)
            spike_positions = np.random.choice(60, 4, replace=False)
            spikes[spike_positions] = np.random.uniform(15, 25, 4)
            if system == 'original':
                trend = np.sin(np.linspace(0, 5, 60)) * 10 + spikes
            else:
                trend = np.sin(np.linspace(0, 5, 60)) * 7 + spikes * 0.6
        
        # 组合基线、趋势和噪声
        delay = base + trend + noise
        delay = np.maximum(delay, 3)  # 确保延迟至少为3ms
        
        delay_time_series[key] = delay

# 绘制延迟时间序列
line_styles = ['-', '--']
markers = ['o', 's', '^']
for i, size in enumerate(['small', 'medium', 'large']):
    for j, system in enumerate(['original', 'optimized']):
        key = f"{size}_{system}"
        data = delay_time_series[key]
        
        # 根据大小和系统类型确定标签
        label = f"{size_names[size]} - {system_names[system]}"
        
        # 绘制曲线，每隔几个点标记一个点以避免拥挤
        ax3.plot(time_points, data, linestyle=line_styles[j], color=colors[i], 
                 marker=markers[i], markevery=10, markersize=8,
                 linewidth=2, label=label)

# 添加标题和标签
ax3.set_title('不同规模参数服务器的延迟变化 (1分钟测试)', fontsize=16)
ax3.set_xlabel('时间 (秒)', fontsize=14)
ax3.set_ylabel('延迟 (毫秒)', fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(fontsize=12, loc='upper right')

# 美化图表
plt.tight_layout()

# 保存图形
plt.savefig('parameterserver_performance_analysis.png', dpi=600, bbox_inches='tight')

# 显示图形
plt.show()

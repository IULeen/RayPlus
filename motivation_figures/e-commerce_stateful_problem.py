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
workloads = ['均匀分布\n(Uniform)', 'Zipfian分布', 'Azure动态\n负载']
throughput_original = np.array([115, 50, 65])  # 原始的吞吐量数据
throughput_optimized = np.array([145, 72, 90])  # 使用简单存储优化策略后的吞吐量数据

# 创建一个大的图形，包含三个子图
plt.figure(figsize=(18, 15))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

# 第一个子图：吞吐量柱状图
ax1 = plt.subplot(gs[0, 0])

# 设置柱状图的宽度和位置
bar_width = 0.35
index = np.arange(len(workloads))

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
ax1.set_title('Ray+无分级存储资源管理系统在电商结账工作流下的吞吐量对比', fontsize=16)
ax1.set_ylabel('吞吐量 (请求/秒)', fontsize=14)
ax1.set_xlabel('负载特征', fontsize=14)

# 设置X轴刻度位置和标签
ax1.set_xticks(index)
ax1.set_xticklabels(workloads)

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

# 生成模拟的延迟数据
delay_distributions = {
    'uniform_original': stats.gamma.rvs(3, loc=15, scale=4, size=1000),  # 均匀分布-原始系统
    'uniform_optimized': stats.gamma.rvs(5, loc=10, scale=2, size=1000),  # 均匀分布-优化系统
    'zipfian_original': stats.gamma.rvs(2, loc=25, scale=8, size=1000),   # Zipfian-原始系统
    'zipfian_optimized': stats.gamma.rvs(3, loc=20, scale=5, size=1000),  # Zipfian-优化系统
    'azure_original': stats.gamma.rvs(2.5, loc=20, scale=6, size=1000),   # Azure-原始系统
    'azure_optimized': stats.gamma.rvs(4, loc=15, scale=3, size=1000),    # Azure-优化系统
}

# 绘制CDF图
linestyles = ['-', '--']
colors = ['blue', 'green', 'red']
workload_names = {
    'uniform': '均匀分布(Uniform)',
    'zipfian': 'Zipfian分布',
    'azure': 'Azure动态负载'
}
system_names = {
    'original': '原始系统',
    'optimized': '简单存储优化后'
}

# 用于存储图例条目
plot_lines = []
plot_labels = []

for i, workload in enumerate(['uniform', 'zipfian', 'azure']):
    for j, system in enumerate(['original', 'optimized']):
        key = f"{workload}_{system}"
        data = delay_distributions[key]
        
        # 计算CDF
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF曲线并保存引用用于图例
        line, = ax2.plot(sorted_data, cdf, linestyle=linestyles[j], color=colors[i], 
                     linewidth=2)
        
        # 保存图例信息
        plot_lines.append(line)
        plot_labels.append(f"{workload_names[workload]} - {system_names[system]}")

# 使用实际绘制的线条创建图例
ax2.legend(plot_lines, plot_labels, fontsize=12, loc='lower right')

# 添加标题和标签
ax2.set_title('请求延迟累积分布函数(CDF)', fontsize=16)
ax2.set_xlabel('延迟 (毫秒)', fontsize=14)
ax2.set_ylabel('累积概率', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)

# 第三个子图：延迟时间序列图 (跨越底部两列)
ax3 = plt.subplot(gs[1, :])

# 生成1分钟内的时间点 (假设每秒采样一次)
time_points = np.arange(0, 60)

# 为每种配置生成随时间变化的延迟数据
delay_time_series = {}
for workload in ['uniform', 'zipfian', 'azure']:
    for system in ['original', 'optimized']:
        key = f"{workload}_{system}"
        
        # 使用不同的基线和波动模式生成数据
        base = 30 if system == 'original' else 20
        if workload == 'uniform':
            # 均匀分布相对稳定
            noise = np.random.normal(0, 3, size=60)
            if system == 'original':
                trend = np.sin(np.linspace(0, 3, 60)) * 5
            else:
                trend = np.sin(np.linspace(0, 3, 60)) * 3
            
        elif workload == 'zipfian':
            # Zipfian有更多波动
            noise = np.random.normal(0, 8, size=60)
            if system == 'original':
                trend = np.sin(np.linspace(0, 6, 60)) * 10 + np.linspace(0, 10, 60)
            else:
                trend = np.sin(np.linspace(0, 6, 60)) * 7 + np.linspace(0, 6, 60)
            
        else:  # azure
            # Azure负载有突发性波动
            noise = np.random.normal(0, 5, size=60)
            spikes = np.zeros(60)
            spike_positions = np.random.choice(60, 5, replace=False)
            spikes[spike_positions] = np.random.uniform(10, 20, 5)
            if system == 'original':
                trend = np.sin(np.linspace(0, 4, 60)) * 8 + spikes
            else:
                trend = np.sin(np.linspace(0, 4, 60)) * 5 + spikes * 0.7
        
        # 组合基线、趋势和噪声
        base_value = base + (10 if workload == 'zipfian' else (5 if workload == 'azure' else 0))
        delay = base_value + trend + noise
        delay = np.maximum(delay, 5)  # 确保延迟至少为5ms
        
        delay_time_series[key] = delay

# 绘制延迟时间序列
line_styles = ['-', '--']
markers = ['o', 's', '^']
for i, workload in enumerate(['uniform', 'zipfian', 'azure']):
    for j, system in enumerate(['original', 'optimized']):
        key = f"{workload}_{system}"
        data = delay_time_series[key]
        
        # 根据负载类型和系统类型确定标签
        label = f"{workload_names[workload]} - {system_names[system]}"
        
        # 绘制曲线，每隔几个点标记一个点以避免拥挤
        ax3.plot(time_points, data, linestyle=line_styles[j], color=colors[i], 
                 marker=markers[i], markevery=10, markersize=8,
                 linewidth=2, label=label)

# 添加标题和标签
ax3.set_title('不同负载下的请求延迟变化 (1分钟测试)', fontsize=16)
ax3.set_xlabel('时间 (秒)', fontsize=14)
ax3.set_ylabel('延迟 (毫秒)', fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(fontsize=12, loc='upper right')

# 美化图表
plt.tight_layout()

# 保存图形
plt.savefig('ecommerce_performance_analysis.png', dpi=600, bbox_inches='tight')

# 显示图形
plt.show()

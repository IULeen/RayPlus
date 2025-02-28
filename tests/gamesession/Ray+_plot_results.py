#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from collections import defaultdict

# 设置中文字体（如果可用）
try:
    import matplotlib.font_manager as fm
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
except Exception:
    pass  # 如果字体设置失败，使用默认字体

def load_performance_data(file_pattern="Ray+_performance_*.json"):
    """加载性能测试数据"""
    all_data = {}
    for filename in glob.glob(file_pattern):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                dist_type = data["distribution_type"]
                all_data[dist_type] = data
                print(f"已加载 {dist_type} 分布数据: {filename}")
        except Exception as e:
            print(f"加载 {filename} 失败: {str(e)}")
    
    return all_data

def generate_latency_cdf_plot(all_data):
    """生成延迟CDF累计分布图"""
    plt.figure(figsize=(10, 6))
    
    colors = {'uniform': 'blue', 'zipfian': 'red'}
    
    for dist_type, data in all_data.items():
        # 提取延迟数据
        latencies = [entry['latency'] for entry in data['request_latencies'] if entry['success']]
        
        if not latencies:
            print(f"警告: {dist_type} 分布没有成功请求数据")
            continue
        
        # 计算CDF
        sorted_latencies = np.sort(latencies)
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        
        # 绘制CDF曲线
        label = f"均匀分布" if dist_type == "uniform" else f"Zipfian分布"
        plt.plot(sorted_latencies, cdf, linewidth=2, color=colors.get(dist_type, 'green'), label=label)
    
    plt.title("Ray+请求延迟累积分布函数(CDF)", fontsize=14)
    plt.xlabel("延迟 (毫秒)", fontsize=12)
    plt.ylabel("累积概率", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Ray+_latency_cdf.png", dpi=300)
    print("已生成延迟CDF图: Ray+_latency_cdf.png")

def generate_latency_timeline_plot(all_data):
    """生成请求延迟时序图"""
    plt.figure(figsize=(12, 6))
    
    colors = {'uniform': 'blue', 'zipfian': 'red'}
    
    for dist_type, data in all_data.items():
        # 提取延迟数据和时间戳
        timeline_data = [(entry['timestamp'], entry['latency']) for entry in data['request_latencies'] if entry['success']]
        
        if not timeline_data:
            print(f"警告: {dist_type} 分布没有成功请求数据")
            continue
        
        # 按时间戳排序
        timeline_data.sort(key=lambda x: x[0])
        timestamps = [t[0] for t in timeline_data]
        latencies = [t[1] for t in timeline_data]
        
        # 绘制延迟时间序列
        label = f"均匀分布" if dist_type == "uniform" else f"Zipfian分布"
        plt.plot(timestamps, latencies, linewidth=1.5, color=colors.get(dist_type, 'green'), 
                 label=label, alpha=0.8)
    
    plt.title("Ray+请求延迟时序变化", fontsize=14)
    plt.xlabel("测试时间 (秒)", fontsize=12)
    plt.ylabel("延迟 (毫秒)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Ray+_latency_timeline.png", dpi=300)
    print("已生成延迟时序图: Ray+_latency_timeline.png")

def generate_throughput_plot(all_data):
    """生成吞吐量柱状图"""
    # 计算平均吞吐量
    throughputs = {}
    for dist_type, data in all_data.items():
        # 计算每秒请求数
        throughput_data = data['request_throughputs']
        if throughput_data:
            # 忽略开始和结束的可能不完整的数据
            if len(throughput_data) > 5:
                throughput_data = throughput_data[2:-2]
            avg_throughput = sum(throughput_data) / len(throughput_data)
            throughputs[dist_type] = avg_throughput
    
    if not throughputs:
        print("警告: 没有有效的吞吐量数据")
        return
    
    # 创建柱状图
    plt.figure(figsize=(8, 6))
    
    # 排序键值确保一致的顺序
    dist_types = sorted(throughputs.keys())
    values = [throughputs[k] for k in dist_types]
    
    # 映射分布名称
    labels = []
    for dt in dist_types:
        if dt == "uniform":
            labels.append("均匀分布")
        elif dt == "zipfian":
            labels.append("Zipfian分布")
        else:
            labels.append(dt)
    
    # 设置柱状图颜色
    colors = ['skyblue', 'lightcoral']
    
    # 绘制柱状图
    bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.6)
    
    # 在柱子上方添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=12)
    
    plt.title("Ray+系统平均吞吐量", fontsize=14)
    plt.ylabel("请求数/秒", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(values) * 1.2)  # 设置Y轴上限
    
    plt.tight_layout()
    plt.savefig("Ray+_throughput.png", dpi=300)
    print("已生成吞吐量柱状图: Ray+_throughput.png")

def generate_method_latency_comparison(all_data):
    """生成不同方法的延迟对比图"""
    plt.figure(figsize=(12, 8))
    
    # 定义方法列表
    methods = ["get_info", "update_position", "use_skill", "check_inactive"]
    
    # 用于存储不同分布类型的各方法延迟
    method_latencies = defaultdict(lambda: defaultdict(list))
    
    # 收集数据
    for dist_type, data in all_data.items():
        for entry in data['request_latencies']:
            if entry['success'] and entry['method'] in methods:
                method_latencies[dist_type][entry['method']].append(entry['latency'])
    
    # 计算每种方法的平均延迟
    avg_latencies = {}
    for dist_type in method_latencies:
        avg_latencies[dist_type] = []
        for method in methods:
            latencies = method_latencies[dist_type][method]
            if latencies:
                avg_latencies[dist_type].append(sum(latencies) / len(latencies))
            else:
                avg_latencies[dist_type].append(0)
    
    # 设置柱状图位置
    x = np.arange(len(methods))
    width = 0.35  # 柱宽度
    
    # 设置柱状图颜色
    colors = {'uniform': 'skyblue', 'zipfian': 'lightcoral'}
    
    # 绘制柱状图
    bars = []
    for i, (dist_type, latencies) in enumerate(avg_latencies.items()):
        label = "均匀分布" if dist_type == "uniform" else "Zipfian分布"
        bar = plt.bar(x + (i-0.5)*width, latencies, width, color=colors[dist_type], 
                     edgecolor='black', label=label)
        bars.append(bar)
    
    # 在柱子上方添加数值标签
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=10)
    
    plt.title("Ray+系统不同方法的平均延迟对比", fontsize=14)
    plt.ylabel("延迟 (毫秒)", fontsize=12)
    plt.xlabel("方法", fontsize=12)
    plt.xticks(x, methods)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Ray+_method_latency.png", dpi=300)
    print("已生成方法延迟对比图: Ray+_method_latency.png")

def generate_comprehensive_plot(all_data):
    """生成综合性能分析图"""
    plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
    
    # 1. 吞吐量柱状图 (左上)
    ax1 = plt.subplot(gs[0, 0])
    
    # 计算平均吞吐量
    throughputs = {}
    for dist_type, data in all_data.items():
        throughput_data = data['request_throughputs']
        if throughput_data:
            if len(throughput_data) > 5:
                throughput_data = throughput_data[2:-2]
            avg_throughput = sum(throughput_data) / len(throughput_data)
            throughputs[dist_type] = avg_throughput
    
    if throughputs:
        dist_types = sorted(throughputs.keys())
        values = [throughputs[k] for k in dist_types]
        
        # 映射分布名称
        labels = []
        for dt in dist_types:
            if dt == "uniform":
                labels.append("均匀分布")
            elif dt == "zipfian":
                labels.append("Zipfian分布")
            else:
                labels.append(dt)
        
        # 设置柱状图颜色
        colors = ['skyblue', 'lightcoral']
        
        # 绘制柱状图
        bars = ax1.bar(labels, values, color=colors, edgecolor='black', width=0.6)
        
        # 在柱子上方添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontsize=12)
        
        ax1.set_title("Ray+系统平均吞吐量", fontsize=14)
        ax1.set_ylabel("请求数/秒", fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_ylim(0, max(values) * 1.2)  # 设置Y轴上限
    
    # 2. 延迟CDF图 (右上)
    ax2 = plt.subplot(gs[0, 1])
    
    colors = {'uniform': 'blue', 'zipfian': 'red'}
    
    for dist_type, data in all_data.items():
        # 提取延迟数据
        latencies = [entry['latency'] for entry in data['request_latencies'] if entry['success']]
        
        if not latencies:
            continue
        
        # 计算CDF
        sorted_latencies = np.sort(latencies)
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        
        # 绘制CDF曲线
        label = f"均匀分布" if dist_type == "uniform" else f"Zipfian分布"
        ax2.plot(sorted_latencies, cdf, linewidth=2, color=colors.get(dist_type, 'green'), label=label)
    
    ax2.set_title("请求延迟累积分布函数(CDF)", fontsize=14)
    ax2.set_xlabel("延迟 (毫秒)", fontsize=12)
    ax2.set_ylabel("累积概率", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    
    # 3. 延迟时序图 (跨越底部)
    ax3 = plt.subplot(gs[1, :])
    
    for dist_type, data in all_data.items():
        # 提取延迟数据和时间戳
        timeline_data = [(entry['timestamp'], entry['latency']) for entry in data['request_latencies'] if entry['success']]
        
        if not timeline_data:
            continue
        
        # 按时间戳排序
        timeline_data.sort(key=lambda x: x[0])
        timestamps = [t[0] for t in timeline_data]
        latencies = [t[1] for t in timeline_data]
        
        # 绘制延迟时间序列
        label = f"均匀分布" if dist_type == "uniform" else f"Zipfian分布"
        ax3.plot(timestamps, latencies, linewidth=1.5, color=colors.get(dist_type, 'green'), 
                label=label, alpha=0.8)
    
    ax3.set_title("请求延迟时序变化", fontsize=14)
    ax3.set_xlabel("测试时间 (秒)", fontsize=12)
    ax3.set_ylabel("延迟 (毫秒)", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Ray+_comprehensive_analysis.png", dpi=300)
    print("已生成综合性能分析图: Ray+_comprehensive_analysis.png")

if __name__ == "__main__":
    # 加载测试数据
    all_data = load_performance_data()
    
    if not all_data:
        print("错误: 没有找到性能测试数据文件")
        exit(1)
    
    # 生成各种图表
    generate_latency_cdf_plot(all_data)
    generate_latency_timeline_plot(all_data)
    generate_throughput_plot(all_data)
    generate_method_latency_comparison(all_data)
    generate_comprehensive_plot(all_data)
    
    print("所有图表生成完成!") 
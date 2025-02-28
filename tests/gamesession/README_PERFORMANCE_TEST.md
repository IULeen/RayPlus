# Rayless性能测试指南

本文档提供了使用Locust对Rayless系统进行性能测试的详细说明。测试重点关注GamePlayerActor有状态函数在不同负载分布下的性能表现。

## 测试目标

1. 测量Rayless系统在处理有状态Actor请求时的端到端延迟
2. 评估系统在不同负载分布下的吞吐量
3. 分析不同方法调用的性能差异
4. 评估`keep_alive`参数对性能的影响

## 准备工作

### 安装依赖

```bash
pip install locust matplotlib numpy requests
```

### 测试脚本说明

- `rayless_init_test.py`: 初始化50个Player实例
- `rayless_locust_test.py`: Locust测试实现，包含均匀分布和Zipfian分布测试
- `locustfile.py`: Locust配置文件
- `rayless_plot_results.py`: 数据分析与图表生成

## 通信方式说明

测试使用HTTP API与Rayless系统通信，而不是直接调用Python函数。API调用示例如下：

```bash
curl -X POST http://localhost:5000/invoke \
-H "Content-Type: application/json" \
-d '{
    "actor_id": "Player9",
    "actor_type": "GamePlayerActor",
    "func_name": "get_info",
    "keep_alive": true
}'
```

Locust测试脚本会自动构造这种HTTP请求，并测量端到端延迟。

## 测试方法

### 1. 启动HTTP服务器

在测试前确保Rayless HTTP服务器已运行：

```bash
python core/transport/http_server.py
```

### 2. 初始化测试环境

运行初始化脚本，创建并注册50个玩家Actor实例：

```bash
python rayless_init_test.py
```

这将确保所有Player0到Player49的实例都已经在系统中注册并可用。

### 3. 运行均匀分布负载测试

使用均匀分布模式运行Locust测试：

```bash
locust -f locustfile.py UniformRaylessUser
```

在Locust Web界面中设置以下参数：
- 用户数: 20-50 (根据系统容量调整)
- 生成速率: 5-10 用户/秒
- 主机: http://localhost:5000

运行测试约5-10分钟，然后停止测试。测试结果将自动保存为JSON文件。

### 4. 运行Zipfian分布负载测试

使用Zipfian分布模式运行Locust测试：

```bash
locust -f locustfile.py ZipfianRaylessUser
```

使用与均匀分布测试相同的参数设置。Zipfian分布将模拟热点访问模式，少数玩家实例会接收大部分请求。

### 5. 生成性能分析图表

运行数据分析脚本，生成性能图表：

```bash
python rayless_plot_results.py
```

这将生成以下图表：
- `rayless_latency_cdf.png`: 延迟累积分布函数(CDF)图
- `rayless_latency_timeline.png`: 请求延迟时序图
- `rayless_throughput.png`: 系统吞吐量柱状图
- `rayless_method_latency.png`: 不同方法的延迟对比图
- `rayless_comprehensive_analysis.png`: 综合性能分析图

## 分析指标

### 端到端延迟

- P50 (中位数): 50%的请求延迟低于此值
- P95: 95%的请求延迟低于此值
- P99: 99%的请求延迟低于此值

### 吞吐量

- 每秒处理的请求数 (RPS)
- 不同负载分布下的吞吐量比较

### 其他分析维度

- 不同方法调用的延迟比较
- `keep_alive=true` vs `keep_alive=false` 的性能影响
- 系统在长时间运行后的性能稳定性

## 注意事项

1. 确保Ray集群和HTTP服务器在测试前已正确初始化
2. 测试过程中监控系统资源使用情况
3. 测试完成后检查日志文件以发现潜在问题
4. 如果测试中出现错误，请检查初始化是否正确完成

## 预期结果

- 均匀分布负载下，所有玩家实例应该有相似的延迟
- Zipfian分布负载下，热点玩家实例可能有更低的延迟（因为它们更可能保持活跃状态）
- `get_info`方法应该比复杂的方法如`update_position`有更低的延迟
- 设置`keep_alive=true`的请求应该有更低的后续调用延迟 
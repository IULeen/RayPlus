# Serverless 有状态 Actor 平台

该项目基于 Ray 实现了一个 Serverless 有状态函数管理与调用平台，旨在为用户提供简单、透明的有状态函数（或 Actor）注册、绑定、调用与销毁接口。系统实现了全托管的 Actor 生命周期管理与多级状态存储，屏蔽了底层复杂性，让用户专注于业务开发。

---

## 目录结构

项目采用模块化设计，主要目录结构如下：

```
project_root/
├── config/
│   └── settings.py              # 全局配置（Redis、MinIO、Ray 等参数）
├── core/
│   ├── actor/
│   │   ├── __init__.py
│   │   ├── actorless_decorator.py   # 提供 @actorless 装饰器，将普通函数/类包装为有状态 Actor
│   │   └── ActorlessConfig.py       # 封装 Actor 配置与序列化
│   ├── service/
│   │   ├── __init__.py
│   │   └── actorless_lookup_service.py  # Actor 实例生命周期管理服务
│   ├── state_manager/
│   │   ├── __init__.py
│   │   └── state_decision.py        # 多级状态存储决策管理
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── redis_store.py          # Redis 存储实现
│   │   └── minio_store.py          # MinIO 对象存储实现
│   └── transport/
│       ├── __init__.py
│       ├── http_server.py          # HTTP API 服务
│       └── actorless_webui/        # Web 管理控制台
│           ├── static/             # 静态资源
│           └── templates/          # 页面模板
├── templates/                      # Actor 模板目录
├── utils/
│   ├── __init__.py
│   ├── annotations.py             # API 装饰器
│   ├── logger.py                  # 日志工具
│   └── serialization.py           # 序列化工具
├── tests/                         # 测试目录
├── requirements.txt               # 项目依赖
└── README.md                      # 项目文档
```

---

## 项目介绍

该系统面向 Serverless 用户，提供一整套基于 Ray 的 Actor 有状态函数管理与调用平台，其主要功能包括：

- **注册与包装**：通过 `@actorless` 装饰器或 `register()` 接口，将用户的函数或类注册为 Actor 模
板，同时保存必要元数据与序列化信息。
- **实例绑定**：使用 `bind(unique_id, *args, **kwargs)` 接口实例化 Actor。系统会将绑定信息保存到
全局注册表，并延迟启动 Actor。
- **函数调用**：通过 `invoke(actorless_ref, keep_alive, method_name, *args, **kwargs)` 等接口
触发 Actor 内部方法调用。系统内部负责自动检测 Actor 状态以及恢复、检查点保存等操作。
- **生命周期管理**：支持 `delete()` 操作终止 Actor 的生命周期，释放资源，并清理状态存储信息。
- **多级状态存储**：支持基于 Redis 及 MinIO 等多个存储后端的状态持久化，实现 Actor 状态在执行间的
保存与恢复。


1. **Actor 模板管理**
   - 支持通过 `@actorless` 装饰器定义 Actor 模板
   - 提供模板注册、查看和管理功能
   - Web 控制台可视化展示模板信息

2. **实例生命周期管理**
   - 中心化的 Actor 实例元数据管理
   - 支持实例创建、查询、调用和销毁
   - 自动化的资源回收与状态保存

3. **多级状态存储**
   - 基于内存、Redis 和 MinIO 的多级存储架构
   - 智能的状态存储决策机制
   - 自动的状态序列化与恢复

4. **Web 管理控制台**
   - 直观的 Actor 实例管理界面
   - 支持模板查看与实例创建
   - 提供方法调用与状态监控

---

## 快速开始

1. **环境准备**

   确保已安装 Python 3.7+ 并安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. **配置设置**

   修改 `config/settings.py` 配置存储服务参数：
   - Redis 连接信息
   - MinIO 存储配置
   - Ray 集群参数

3. **启动服务**

   ```bash
   # 启动 Ray
   ray start --head

   # 启动 Web 服务
   python core/transport/http_server.py
   ```

4. **访问控制台**

   打开浏览器访问 `http://localhost:5000` 进入管理控制台。

---

## API 使用示例

1. **定义 Actor 模板**

```python
from core.actor.actorless_decorator import actorless

@actorless
class Account:
    def __init__(self, name, CardID, balance):
        self.name = name
        self.CardID = CardID
        self.balance = balance

    def get_info(self):
        return f"Name: {self.name}, CardID: {self.CardID}, Balance: {self.balance}"

    def deposit(self, amount):
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if self.balance >= amount:
            self.balance -= amount
            return self.balance
        else:
            return "Insufficient balance"
```

通过 bind 方法实例化 Actor 实例：

```python
from core.actorless_template.Account import Account

# 通过 bind 方法实例化 Actor 实例
account_instance = Account.bind(unique_id="account1", name="Alice", CardID="ID001", balance=100)
```

### 调用 Actor 方法

可以通过 `ActorManager.invoke` 或 HTTP 接口调用 Actor 内部方法。例如，通过命令行调用 `get_info` 方法：

```python
from core.actor.actor_manager import ActorManager

info = ActorManager.invoke("account1", "get_info")
print(info)
```

或通过 HTTP POST 请求调用 `/invoke` 接口：

```json
{
  "actor_id": "account1",
  "func_name": "deposit",
  "params": [50]
}
```

### 删除 Actor 实例

调用删除接口：

```python
ActorManager.delete("account1")
```

或通过 HTTP POST 请求调用 `/delete` 接口。

---

## 模块说明

- **config**：提供全局配置参数，统一管理 Redis、MinIO、Ray 的配置信息。
- **core/actor**：主要处理 Actor 的注册、序列化、包装与生命周期管理。包括：
  - `actorless_decorator.py`：提供 `@actorless` 装饰器，将普通函数/类转换为 Actor 模板。
  - `ActorlessConfig.py`：使用 cloudpickle 对函数或类序列化，保存相关配置信息。
  - `actor_manager.py`：管理 Actor 实例的创建、调用、恢复与销毁。
- **core/actorless_template**：存放用户示例的业务模板，例如 `Account` 和 `ParameterServer`。
- **core/state_manager**：封装状态管理模块，基于 Redis 实现状态存储与恢复。
- **core/storage**：封装与具体存储后端（Redis、MinIO）的交互操作。
- **core/transport**：对外提供 HTTP 接口（基于 Flask），将外部请求映射为内部 Actor 调用。
- **utils**：提供日志、序列化、注解等通用工具函数，便于各模块复用。

---

## 技术栈

- **Ray**：用于分布式 Actor 调度与执行。
- **Redis**：用于存储 Actor 状态，支持多级存储策略。
- **MinIO**：用于持久化存储（可选）。
- **Flask**：构建简单的 HTTP 接口，提供外部调用与管理接口。
- **Python**：项目主要编程语言，结合 cloudpickle 进行序列化操作。

---

## 注意事项

- 使用 `@actorless` 装饰器后，用户注册的类或函数将被包装成延迟实例化的 Actor，对外不可直接调用原类。
- Actor 的状态管理（包括检查点与恢复）由系统内部自动处理，用户只需关注业务逻辑编写。
- 由于涉及底层 Actor 状态存储与恢复操作，部分接口可能后续需要完善或扩展，欢迎反馈和贡献改进建议。


---

## 许可证

[MIT License](LICENSE)
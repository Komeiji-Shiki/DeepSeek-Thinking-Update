# DeepSeek 思考模式 + 工具调用 示例

本项目演示了如何在 DeepSeek 的思考模式（Reasoning Mode）下进行工具调用，并实现消息合并优化。

## 功能特点

- ✅ 支持 DeepSeek 思考模式（reasoning_content）
- ✅ 支持多轮工具调用
- ✅ 自动合并多次工具调用的思维链
- ✅ 优化消息列表结构，减少冗余
- ✅ **MCP 支持**：集成 Model Context Protocol 工具服务器

## 核心原理

### 传统方式（每次工具调用都添加新消息）

```
messages = [
  { role: "user", content: "杭州明天天气怎么样？" },
  { role: "assistant", tool_calls: [...], reasoning_content: "思考1" },
  { role: "tool", content: "日期结果" },
  { role: "assistant", tool_calls: [...], reasoning_content: "思考2" },
  { role: "tool", content: "天气结果" },
  { role: "assistant", content: "最终回复", reasoning_content: "思考3" },
]
```

### 优化方式（合并到单一助手消息）

```json
{
  "messages": [
    {
      "role": "user",
      "content": "杭州明天天气怎么样？"
    },
    {
      "role": "assistant",
      "content": "最终回复",
      "reasoning_content": "思考1\n\n{\"tool_calls\":[{\"function\":{\"name\":\"get_date\",\"arguments\":\"{}\"},\"type\":\"function\",\"index\":0}]}\n\n思考2\n\n{\"tool_calls\":[{\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"杭州\\\",\\\"date\\\":\\\"2025-12-03\\\"}\"},\"type\":\"function\",\"index\":0}]}\n\n思考3",
      "tool_calls": [
        {
          "id": "call_00_xxx",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"杭州\",\"date\":\"2025-12-03\"}"
          },
          "type": "function",
          "index": 0
        }
      ]
    }
  ]
}
```

**关键点**：
- `tool_calls` 保留完整字段（id、type、index、function）用于 API 调用
- 历史工具调用压扁后（去掉 id，保留 function、type、index）嵌入到 `reasoning_content`

## 实现逻辑

### 流程图

```
用户提问
    │
    ▼
┌─────────────────────────────────────────┐
│  发送请求到 DeepSeek API               │
│  (携带 messages + tools)               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  收到响应                               │
│  - reasoning_content (思考过程)         │
│  - content (回复内容)                   │
│  - tool_calls (工具调用，可能为空)      │
└─────────────────────────────────────────┘
    │
    ├─── 首次调用 (sub_turn=1) ───────────┐
    │                                      │
    │    直接添加助手消息到 messages       │
    │                                      │
    ├─── 后续调用 (sub_turn>1) ───────────┐
    │                                      │
    │    1. 删除所有 tool 消息             │
    │    2. 追加新思维链到原助手消息       │
    │    3. 更新 tool_calls ID            │
    │                                      │
    ▼
┌─────────────────────────────────────────┐
│  有 tool_calls?                         │
│                                         │
│  是 → 执行工具，添加 tool 消息，继续循环│
│  否 → 结束，返回最终结果                │
└─────────────────────────────────────────┘
```

### 关键代码逻辑

```python
def flatten_tool_calls(tool_calls):
    """压扁 tool_calls，去掉 id，保留 function、type、index"""
    return [
        {
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            },
            "type": tc.type if hasattr(tc, 'type') else "function",
            "index": tc.index if hasattr(tc, 'index') else i
        }
        for i, tc in enumerate(tool_calls)
    ]

# 使用示例
if old_tool_calls:
    flattened_tools = flatten_tool_calls(old_tool_calls)
    # 包装在 tool_calls 字段中
    tools_obj = {"tool_calls": flattened_tools}
    tools_json = json.dumps(tools_obj, ensure_ascii=False)
    old_reasoning = old_reasoning + "\n\n" + tools_json

if sub_turn == 1:
    # 首次调用：直接添加助手消息
    messages.append(message_to_dict(new_message))
    assistant_msg_index = len(messages) - 1
else:
    # 后续调用：合并到之前的助手消息
    
    # 1. 删除所有 tool 消息
    while messages[-1].get('role') == 'tool':
        messages.pop()
    
    # 2. 获取之前的助手消息
    prev_assistant = messages[assistant_msg_index]
    
    # 3. 将旧的 tool_calls 压扁并添加到 reasoning_content
    old_reasoning = prev_assistant.get('reasoning_content', '')
    old_tool_calls = prev_assistant.get('tool_calls', [])
    
    if old_tool_calls:
        # 压扁工具调用（去掉 id，保留 function、type、index）
        flattened_tools = flatten_tool_calls(old_tool_calls)
        # 包装在 tool_calls 字段中
        tools_obj = {"tool_calls": flattened_tools}
        tools_json = json.dumps(tools_obj, ensure_ascii=False)
        old_reasoning = old_reasoning + "\n\n" + tools_json
    
    # 4. 追加新的思维链
    combined_reasoning = old_reasoning + "\n\n" + new_reasoning
    prev_assistant['reasoning_content'] = combined_reasoning
    
    # 5. 更新工具调用（保留完整字段：id, function, type, index）
    if new_tool_calls:
        prev_assistant['tool_calls'] = [
            {
                "id": tc.id,
                "function": {
                    "arguments": tc.function.arguments,
                    "name": tc.function.name
                },
                "type": tc.type,
                "index": tc.index if hasattr(tc, 'index') else i
            }
            for i, tc in enumerate(new_tool_calls)
        ]
    else:
        # 无更多工具调用，清除并更新最终回复
        del prev_assistant['tool_calls']
        prev_assistant['content'] = new_content
```

## 使用方法

### 1. 安装依赖

```bash
pip install openai
```

### 2. 配置 API

在 `deepseek_thinking_tools_demo.py` 中修改以下配置：

```python
API_KEY = "your-api-key"
BASE_URL = "https://api.deepseek.com"
```

### 3. 运行脚本

```bash
python deepseek_thinking_tools_demo.py
```

## 示例输出

```
==================================================
Turn 1.1 - 发送请求...
==================================================

📋 当前消息列表 (1 条):
   [0] user: 杭州明天天气怎么样？...

📝 新思考过程 (reasoning_content):
   用户想知道杭州明天的天气情况。我需要明天的日期...

🔧 工具调用 (tool_calls):
   - get_date({}) [id: call_xxx]

📌 添加助手消息到索引 1

🔨 工具执行结果 (get_date): 2025-12-02

==================================================
Turn 1.2 - 发送请求...
==================================================

📝 新思考过程 (reasoning_content):
   今天是2025年12月2日，明天就是2025年12月3日...

🔧 工具调用 (tool_calls):
   - get_weather({"location": "杭州", "date": "2025-12-03"}) [id: call_yyy]

🗑️ 删除了 1 条工具消息

🔗 合并思维链 (总长度: 156 字符)

🔄 更新工具调用 ID

🔨 工具执行结果 (get_weather): 杭州 2025-12-03 天气: 多云 7~13°C

==================================================
Turn 1.3 - 发送请求...
==================================================

📝 新思考过程 (reasoning_content):
   已获取天气信息，现在可以回复用户了...

💬 回复内容 (content):
   杭州明天（12月3日）天气多云，温度7~13°C...

🗑️ 删除了 1 条工具消息

🔗 合并思维链 (总长度: 234 字符)

✏️ 更新最终回复内容

✅ Turn 1 完成 - 无更多工具调用
```

### 最终消息列表

```json
[
  {
    "role": "user",
    "content": "杭州明天天气怎么样？"
  },
  {
    "role": "assistant",
    "content": "杭州明天（12月3日）天气多云，温度7~13°C，建议穿轻便外套。",
    "reasoning_content": "用户想知道杭州明天的天气情况...\n\n{\"tool_calls\":[{\"function\":{\"name\":\"get_date\",\"arguments\":\"{}\"},\"type\":\"function\",\"index\":0}]}\n\n今天是2025年12月2日...\n\n{\"tool_calls\":[{\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"杭州\\\",\\\"date\\\":\\\"2025-12-03\\\"}\"},\"type\":\"function\",\"index\":0}]}\n\n已获取天气信息..."
  }
]
```

**说明**：
- `reasoning_content` 中嵌入了压扁后的工具调用历史（去掉 `id`，保留 `function`、`type`、`index`）
- 如果助手消息还有 `tool_calls` 字段，则保留完整结构（包含 `id`、`type`、`index`、`function`）用于后续 API 调用
- 压扁格式：`{"tool_calls":[{"function":{"name":"...","arguments":"..."},"type":"function","index":0}]}`
- 这样既保留了完整的思考轨迹，又保证了 API 调用的兼容性

## 工具定义

本示例包含两个模拟工具：

| 工具名称 | 描述 | 参数 |
|---------|------|------|
| `get_date` | 获取当前日期 | 无 |
| `get_weather` | 获取指定地点和日期的天气 | `location` (城市名), `date` (YYYY-mm-dd) |

### 自定义工具

修改 `tools` 列表和 `TOOL_CALL_MAP` 映射来添加自定义工具：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "your_tool_name",
            "description": "工具描述",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "参数描述"}
                },
                "required": ["param1"]
            }
        }
    }
]

def your_tool_function(param1):
    return "工具执行结果"

TOOL_CALL_MAP = {
    "your_tool_name": your_tool_function
}
```

## 注意事项

1. **模型选择**：使用 `deepseek-reasoner` 模型以获得思考模式支持
2. **带宽优化**：在新的 Turn 开始时，建议清除历史消息中的 `reasoning_content`
3. **错误处理**：生产环境中应添加适当的异常处理

## OpenAI 兼容代理服务器

我们提供了一个 OpenAI 兼容的代理服务器 [`proxy_server.py`](proxy_server.py)，监听本地端口，接收 OpenAI 格式的请求，转发到后端 API 并进行工具调用优化。

### 配置文件

代理服务器使用 `config.jsonc` 配置文件（支持注释）：

```jsonc
{
    // API 配置
    "chat_completions_url": "https://api.deepseek.com/v1/chat/completions",
    "models_url": "https://api.deepseek.com/v1/models",
    
    // 转发到后端 API 的 Key（可选）
    "api_key": "",
    
    // 访问控制
    "access_keys": [],           // 留空则不验证
    "allow_user_api_key": true,  // 允许用户使用自己的 Key
    
    // 服务器配置
    "host": "127.0.0.1",
    "port": 8002,
    
    // MCP 配置
    "mcp_enabled": true,
    "auto_execute_mcp_tools": true
}
```

### 访问控制模式

| 模式 | 配置 | 说明 |
|------|------|------|
| 开放访问 | `access_keys: []` | 任何人都可以访问，使用用户提供的 API Key |
| 密钥验证 | `access_keys: ["key1", "key2"]` | 用户必须使用这些密钥访问 |
| 代理转发 | `api_key: "sk-xxx"` | 所有请求使用配置的 Key 转发 |

### 启动代理服务器

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务器（使用默认配置文件 config.jsonc）
python proxy_server.py

# 指定配置文件
python proxy_server.py --config my_config.jsonc

# 命令行参数覆盖配置
python proxy_server.py --host 0.0.0.0 --port 8080

# 禁用 MCP
python proxy_server.py --no-mcp
```

### Web 管理界面

代理服务器提供 Web 管理界面：

| 页面 | URL | 说明 |
|------|-----|------|
| 首页 | `http://127.0.0.1:8002/` | 服务概览和导航 |
| MCP 管理 | `http://127.0.0.1:8002/admin` | 管理 MCP 服务 |
| 工具列表 | `http://127.0.0.1:8002/tools` | 查看可用工具 |
| 健康状态 | `http://127.0.0.1:8002/status` | 服务状态监控 |

### 使用代理服务器

任何支持 OpenAI API 的客户端都可以使用这个代理：

```python
from openai import OpenAI

# 连接到本地代理
client = OpenAI(
    api_key="your-api-key",  # API key 或访问密钥
    base_url="http://127.0.0.1:8002/v1"
)

# 发送请求（model 参数是必需的）
response = client.chat.completions.create(
    model="deepseek-reasoner",  # 必需参数
    messages=[
        {"role": "user", "content": "杭州明天天气怎么样？"}
    ],
    tools=[...]  # 工具定义（可选）
)

# 获取结果
print(response.choices[0].message.content)
print(response.choices[0].message.reasoning_content)  # 完整思维链
```

### API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/v1/chat/completions` | 聊天补全 |
| GET | `/v1/models` | 从后端 API 获取模型列表 |
| GET | `/health` | 健康检查 |

**注意**：
- `model` 参数是必需的，必须由用户指定
- `/v1/models` 从配置的 `models_url` 动态获取，失败时返回错误

### 特性

- ✅ **OpenAI 完全兼容**：可被任何 OpenAI 客户端使用
- ✅ **配置文件支持**：使用 JSONC 格式配置（支持注释）
- ✅ **访问控制**：支持多种访问控制模式
- ✅ **自动消息优化**：自动合并工具调用到单一助手消息
- ✅ **完整思维链**：在响应中返回 `reasoning_content` 字段
- ✅ **流式响应**：支持 SSE 流式输出
- ✅ **MCP 工具集成**：自动加载和执行 MCP 服务器工具
- ✅ **Web 管理界面**：可视化管理 MCP 服务

## OpenAI 兼容客户端（直接调用）

如果你不需要代理服务器，可以使用 [`deepseek_compatible_client.py`](deepseek_compatible_client.py) 直接在代码中调用，自动处理工具调用优化。

### 快速开始

```python
from deepseek_compatible_client import DeepSeekCompatibleClient

# 初始化客户端
client = DeepSeekCompatibleClient(
    api_key="your-api-key",
    base_url="https://api.deepseek.com"
)

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["location", "date"]
            }
        }
    }
]

# 定义工具函数
def get_weather(location: str, date: str):
    return f"{location} {date} 天气: 多云 7~13°C"

tool_functions = {
    "get_weather": get_weather
}

# 发送请求
messages = [
    {"role": "user", "content": "杭州明天天气怎么样？"}
]

result = client.chat_completions_create(
    messages=messages,
    tools=tools,
    tool_functions=tool_functions,
    model="deepseek-reasoner"
)

# 获取结果
print(result['content'])  # 最终回复
print(result['reasoning_content'])  # 完整思维链
print(result['messages'])  # 完整消息历史
```

### API 参考

#### `DeepSeekCompatibleClient.chat_completions_create()`

创建聊天补全，自动处理工具调用。

**参数：**
- `messages` (List[Dict]): 消息列表
- `tools` (List[Dict], 可选): 工具定义列表
- `tool_functions` (Dict[str, Callable], 可选): 工具函数映射
- `model` (str): 模型名称，默认 "deepseek-reasoner"
- `max_iterations` (int): 最大工具调用迭代次数，默认 10
- `**kwargs`: 其他参数传递给 DeepSeek API

**返回值：**
```python
{
    "content": "最终回复内容",
    "reasoning_content": "完整思维链（包含压扁的工具调用）",
    "usage": {...},
    "finish_reason": "stop",
    "messages": [...]  # 完整消息历史
}
```

### 特性

- ✅ **OpenAI 兼容接口**：使用熟悉的 API 格式
- ✅ **自动工具调用**：自动执行工具并处理结果
- ✅ **消息优化**：自动合并工具调用到单一助手消息
- ✅ **完整思维链**：返回包含工具调用历史的完整思考过程

## MCP (Model Context Protocol) 支持

代理服务器支持 MCP 协议，可以连接各种 MCP 工具服务器，自动将它们的工具暴露给 DeepSeek 模型使用。

### 支持的 MCP 服务器类型

| 类型 | 说明 | 配置项 |
|------|------|--------|
| `stdio` | 通过子进程和标准 IO 通信 | `command`, `args`, `env` |
| `streamableHttp` | 通过 HTTP 请求通信 | `url`, `headers` |
| `sse` | 通过 Server-Sent Events 通信 | `url`, `headers` |

### MCP 目录结构

MCP 服务通过 `mcp_servers/` 目录进行管理，每个服务一个独立文件夹：

```
mcp_servers/
├── __init__.py          # 管理模块
├── README.md            # 说明文件
├── enabled.txt          # 启用的服务列表
├── baidu_search/        # 百度搜索服务 (stdio)
│   ├── config.json      # 服务配置
│   └── server.py        # 服务主程序
└── baidu_ai_search/     # 百度 AI 搜索服务 (HTTP)
    └── config.json      # 服务配置
```

### 启用/禁用服务

编辑 `mcp_servers/enabled.txt` 文件：

```
# MCP 服务启用列表
# 每行一个服务文件夹名，以 # 开头的行为注释

baidu_search
baidu_ai_search
```

### 服务配置文件 (config.json)

**STDIO 类型服务：**

```json
{
    "name": "baidu_search",
    "description": "百度网页搜索服务",
    "version": "1.0.0",
    "type": "stdio",
    "command": "python",
    "args": [],
    "env": {
        "BAIDU_APPBUILDER_API_KEY": "your-api-key"
    }
}
```

**HTTP/SSE 类型服务：**

```json
{
    "name": "baidu_ai_search",
    "description": "百度 AI 搜索服务",
    "version": "1.0.0",
    "type": "streamableHttp",
    "url": "https://qianfan.baidubce.com/v2/ai_search/mcp",
    "headers": {
        "Authorization": "Bearer your-api-key"
    }
}
```

### 启动代理服务器（带 MCP）

```bash
# 启动服务器（自动从 mcp_servers 目录加载服务）
python proxy_server.py

# 禁用 MCP
python proxy_server.py --no-mcp
```

### MCP 管理 API

代理服务器提供以下 API 用于管理 MCP 服务器：

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/v1/mcp/status` | 获取 MCP 状态和所有工具 |
| GET | `/v1/mcp/tools` | 获取 MCP 工具列表（OpenAI 格式） |
| GET | `/v1/mcp/servers` | 列出所有 MCP 服务器 |
| POST | `/v1/mcp/servers/<name>/start` | 启动指定服务器 |
| POST | `/v1/mcp/servers/<name>/stop` | 停止指定服务器 |
| POST | `/v1/mcp/reload` | 重新加载 MCP 配置 |

### 添加新 MCP 服务

1. 在 `mcp_servers/` 下创建服务目录：
   ```bash
   mkdir mcp_servers/my_service
   ```

2. 创建配置文件 `config.json`

3. 如果是 STDIO 类型，创建 `server.py` 实现 MCP 协议

4. 在 `enabled.txt` 中添加服务名称

### 使用 MCP 工具

MCP 工具会自动合并到请求的工具列表中。客户端无需任何改动：

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="http://127.0.0.1:8002/v1"
)

# MCP 工具会自动添加到可用工具列表
# 工具名称格式：{服务器名}_{工具名}
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "user", "content": "搜索今天的新闻"}
    ]
    # 无需手动指定 tools，MCP 工具会自动加载
)

print(response.choices[0].message.content)
```

### 禁用自动执行 MCP 工具

如果想让客户端自己处理 MCP 工具调用：

```python
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[...],
    extra_body={"execute_mcp_tools": False}  # 禁用自动执行
)

# 检查是否有工具调用
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"工具: {tc.function.name}")
        print(f"参数: {tc.function.arguments}")
```

## 参考资料

- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Lianues/DeepSeek-Thinking-Update](https://github.com/Lianues/DeepSeek-Thinking-Update)
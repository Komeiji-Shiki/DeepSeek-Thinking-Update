# Python代码执行器 MCP服务

本地Python代码执行服务，提供安全的代码执行能力。

## ⚠️ 安全警告

此服务会在本地执行Python代码，请：
- ✅ 仅在受信任的环境中使用
- ✅ 不要执行来源不明的代码
- ✅ 建议在虚拟环境中运行
- ❌ 不要在生产环境暴露此服务

## 功能特性

### 1. 执行Python代码
- ✅ 支持标准库和已安装的第三方库
- ✅ 30秒超时保护
- ✅ 捕获标准输出和错误输出
- ✅ 返回执行状态和结果

### 2. 包管理
- 📦 安装Python包（pip install）
- 📋 列出已安装的包
- 🔄 支持指定包版本

### 3. 文件操作
- 📝 创建Python文件
- ▶️ 运行本地Python文件
- 📂 支持命令行参数

## 可用工具

### execute_python
执行Python代码并返回结果

**参数：**
- `code` (string, 必需): 要执行的Python代码
- `timeout` (number, 可选): 超时时间(秒)，默认30

**示例：**
```python
# 简单计算
print(2 + 2)

# 数据处理
import json
data = {"name": "test", "value": 123}
print(json.dumps(data, indent=2))

# 使用第三方库（需先安装）
import requests
response = requests.get("https://api.github.com")
print(response.status_code)
```

### install_package
安装Python包

**参数：**
- `package` (string, 必需): 包名，可含版本号

**示例：**
```
requests
numpy==1.24.0
pandas>=2.0.0
```

### list_packages
列出已安装的所有Python包

**返回：** JSON格式的包列表，包含名称和版本

### run_python_file
运行本地Python文件

**参数：**
- `filepath` (string, 必需): Python文件路径
- `args` (array, 可选): 命令行参数

### create_python_file
创建Python文件

**参数：**
- `filepath` (string, 必需): 文件路径
- `code` (string, 必需): Python代码内容

## 使用示例

### 1. 数据分析
```python
import pandas as pd
import numpy as np

# 创建数据
data = pd.DataFrame({
    'A': np.random.randn(5),
    'B': np.random.randn(5)
})

print(data.describe())
```

### 2. API调用
```python
import requests

response = requests.get('https://api.github.com/users/github')
print(response.json())
```

### 3. 文件处理
```python
with open('test.txt', 'w') as f:
    f.write('Hello, World!')

with open('test.txt', 'r') as f:
    print(f.read())
```

## 技术实现

- 使用subprocess隔离执行
- 临时文件存储代码
- UTF-8编码支持中文
- 超时保护机制
- 完整的错误捕获

## 限制

- ⏱️ 默认30秒超时
- 💾 继承当前Python环境
- 🔒 无网络隔离
- 📁 可访问本地文件系统
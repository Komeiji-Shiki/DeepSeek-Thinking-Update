"""
DeepSeek OpenAI 兼容代理服务器
监听本地端口，接收 OpenAI 格式的请求，转发到 DeepSeek API 并进行工具调用优化
支持 MCP (Model Context Protocol) 工具集成
支持流式响应 (streaming)
"""

import json
import re
import os
import sys
import ast
import html
import requests
from typing import List, Dict, Any, Optional, Generator, Iterator, Set, Tuple
from flask import Flask, request, jsonify, Response, stream_with_context
from openai import OpenAI
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler

# MCP 支持
try:
    from mcp_servers.mcp_client import get_mcp_manager, MCPManager
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("警告: MCP 客户端未找到，MCP 功能将被禁用")


def setup_logging(log_file='proxy_server.log', log_level=logging.INFO, debug=False):
    """配置日志记录"""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 文件处理器 (UTF-8 support)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    
    # 获取根 logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else log_level)
    
    # 清除已有的处理器，避免重复记录
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 降低依赖库的日志级别
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info("日志系统初始化完成")


app = Flask(__name__, static_folder='static', static_url_path='/static')


# ==================== 配置加载 ====================

def load_config(config_path: str = "config.jsonc") -> Dict[str, Any]:
    """加载配置文件（支持 JSONC 格式，带注释）"""
    default_config = {
        "chat_completions_url": "https://api.deepseek.com/v1/chat/completions",
        "models_url": "https://api.deepseek.com/v1/models",
        "api_key": "",
        "access_keys": [],
        "allow_user_api_key": True,
        "host": "127.0.0.1",
        "port": 8002,
        "debug": False,
        "mcp_enabled": True,
        "auto_execute_mcp_tools": True,
        "max_iterations": 100,
        "keep_tool_results_count": 0,
        "compatibility_mode": False,
        "compatibility_mode_models": [],
        "compatibility_send_tools": False,
        "model_routes": [],
        "system_prompt_enabled": False,
        "system_prompt": "## 工具调用注意事项\n\n当你使用工具获取信息时，请注意以下几点：\n\n1. **工具调用结果不会保存在对话历史中**：每次工具调用的原始结果只会在当前回合可见，后续对话中将无法再访问这些原始数据。\n\n2. **主动提取和整理信息**：在收到工具返回的结果后，请在你的思考过程中提取所有有用的信息，包括：\n   - 关键数据和数值\n   - 重要的名称、日期、地点等\n   - 相关的上下文信息\n   - 可能在后续对话中需要引用的内容\n\n3. **在回复中复述关键信息**：将提取的重要信息融入你的回复中，这样用户和你都能在后续对话中参考这些信息。\n\n4. **结构化输出**：当工具返回大量信息时，请以清晰、结构化的方式呈现，便于理解和后续引用。"
    }
    
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除 JSONC 注释（更完善的处理）
        # 1. 移除多行注释 /* ... */
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # 2. 移除单行注释 // ...（但保留字符串中的 //）
        lines = []
        for line in content.split('\n'):
            # 简单处理：查找不在字符串中的 // 注释
            # 如果行中有 //, 只保留 // 之前的部分（简化处理）
            if '//' in line:
                # 检查 // 是否在字符串中
                in_string = False
                quote_char = None
                comment_pos = -1
                for i, char in enumerate(line):
                    if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                            quote_char = None
                    elif char == '/' and i < len(line) - 1 and line[i+1] == '/' and not in_string:
                        comment_pos = i
                        break
                
                if comment_pos >= 0:
                    line = line[:comment_pos]
            lines.append(line)
        content = '\n'.join(lines)
        
        config = json.loads(content)
        
        # 调试：显示从配置文件加载的关键配置
        print(f"✓ 配置文件加载成功: {config_path}")
        if 'port' in config:
            print(f"  - 配置文件中的端口: {config['port']}")
        if 'host' in config:
            print(f"  - 配置文件中的主机: {config['host']}")
        
        # 合并默认配置
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    except Exception as e:
        print(f"✗ 加载配置文件失败: {e}，使用默认配置")
        import traceback
        traceback.print_exc()
        return default_config


def get_base_url_from_chat_url(chat_url: str) -> str:
    """从聊天补全 URL 中提取基础 URL（用于 OpenAI SDK）"""
    # 移除 /chat/completions 部分，保留到 /v1
    if '/chat/completions' in chat_url:
        return chat_url.rsplit('/chat/completions', 1)[0]
    return chat_url


def validate_access_key(auth_header: str) -> tuple:
    """
    验证访问密钥
    返回: (是否有效, API Key, 错误消息)
    """
    global CONFIG
    
    if not auth_header.startswith('Bearer '):
        return False, None, "Missing or invalid Authorization header"
    
    user_key = auth_header[7:]
    access_keys = CONFIG.get("access_keys", [])
    allow_user_api_key = CONFIG.get("allow_user_api_key", True)
    config_api_key = CONFIG.get("api_key", "")
    
    # 如果没有配置访问密钥，允许所有请求
    if not access_keys:
        # 如果允许用户使用自己的 API Key
        if allow_user_api_key and user_key:
            return True, user_key, None
        # 否则使用配置的 API Key
        if config_api_key:
            return True, config_api_key, None
        # 没有配置 API Key，返回用户提供的
        return True, user_key, None
    
    # 检查是否是有效的访问密钥
    if user_key in access_keys:
        # 使用配置的 API Key
        if config_api_key:
            return True, config_api_key, None
        return False, None, "Server API key not configured"
    
    # 如果允许用户使用自己的 API Key
    if allow_user_api_key:
        return True, user_key, None
    
    return False, None, "Invalid access key"


def _model_pattern_to_regex(pattern: str) -> str:
    """将简单通配符模型模式转换为正则"""
    escaped = re.escape(pattern).replace(r"\*", ".*")
    return f"^{escaped}$"


def model_matches_pattern(model: str, pattern: str) -> bool:
    """判断模型名是否匹配模式（支持 * 通配）"""
    if not isinstance(pattern, str) or not pattern.strip():
        return False
    return re.match(_model_pattern_to_regex(pattern.strip()), model or "") is not None


def is_model_matched(model: str, patterns: Any) -> bool:
    """判断模型是否命中模式列表/单模式"""
    if not model:
        return False
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, list):
        return False
    return any(model_matches_pattern(model, p) for p in patterns if isinstance(p, str))


def resolve_model_route(model: str, request_api_key: str) -> Dict[str, Any]:
    """
    按模型解析路由配置：
    - 支持不同模型走不同 chat/models URL
    - 支持模型级 compatibility_mode 开关
    """
    global CONFIG

    resolved = {
        "chat_completions_url": CONFIG.get("chat_completions_url", "https://api.deepseek.com/v1/chat/completions"),
        "models_url": CONFIG.get("models_url", "https://api.deepseek.com/v1/models"),
        "api_key": request_api_key or CONFIG.get("api_key", ""),
        "compatibility_mode": bool(CONFIG.get("compatibility_mode", False)),
        "force_xml_json_tool_call": bool(CONFIG.get("force_xml_json_tool_call", False)),
        "route_name": "default"
    }

    # 全局模型匹配开关（命中即开启兼容）
    if is_model_matched(model, CONFIG.get("compatibility_mode_models", [])):
        resolved["compatibility_mode"] = True

    # 路由覆盖（按顺序命中第一条）
    routes = CONFIG.get("model_routes", [])
    if isinstance(routes, list):
        for idx, route in enumerate(routes):
            if not isinstance(route, dict):
                continue
            patterns = route.get("models", [])
            if not is_model_matched(model, patterns):
                continue

            if route.get("chat_completions_url"):
                resolved["chat_completions_url"] = route["chat_completions_url"]
            if route.get("models_url"):
                resolved["models_url"] = route["models_url"]
            if route.get("api_key"):
                resolved["api_key"] = route["api_key"]
            if "compatibility_mode" in route:
                resolved["compatibility_mode"] = bool(route.get("compatibility_mode"))
            if "force_xml_json_tool_call" in route:
                resolved["force_xml_json_tool_call"] = bool(route.get("force_xml_json_tool_call"))

            resolved["route_name"] = route.get("name", f"route_{idx}")
            break

    return resolved


def fetch_models_from_backend(models_url: str, api_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[int, str]]]:
    """从单个 models 端点拉取模型列表"""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(models_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        return None, (response.status_code, f"Failed to fetch models from API: {response.status_code}")
    except requests.exceptions.Timeout:
        return None, (504, "Request to models API timed out")
    except Exception as e:
        return None, (500, f"Failed to fetch models: {str(e)}")


# 全局配置
CONFIG: Dict[str, Any] = {}

# 全局 MCP 管理器
mcp_manager: Optional['MCPManager'] = None


class DeepSeekProxy:
    """DeepSeek 代理处理器"""
    
    def __init__(
        self,
        api_key: str,
        mcp_mgr: Optional['MCPManager'] = None,
        chat_completions_url: Optional[str] = None,
        compatibility_mode: bool = False,
        route_name: str = "default",
        force_xml_json_tool_call: bool = False
    ):
        """初始化客户端"""
        global CONFIG
        resolved_chat_url = chat_completions_url or CONFIG.get("chat_completions_url", "https://api.deepseek.com/v1/chat/completions")
        base_url = get_base_url_from_chat_url(resolved_chat_url)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.chat_completions_url = resolved_chat_url
        self.compatibility_mode = bool(compatibility_mode)
        self.compatibility_send_tools = bool(CONFIG.get("compatibility_send_tools", False))
        self.route_name = route_name
        self.force_xml_json_tool_call = bool(force_xml_json_tool_call)
        self.mcp_manager = mcp_mgr
        self.mcp_call_counter = 0  # 每个请求独立的MCP调用计数器
    
    def _message_to_dict(self, message) -> Dict[str, Any]:
        """将消息对象转换为字典格式"""
        result = {
            "role": message.role,
            "content": message.content or "",
        }
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            result["reasoning_content"] = message.reasoning_content
        
        tool_calls = getattr(message, 'tool_calls', None)
        normalized_tool_calls = self._normalize_tool_calls(tool_calls)
        if normalized_tool_calls:
            result["tool_calls"] = normalized_tool_calls
        
        return result
    
    def _format_tool_call_text(self, tool_name: str, arguments: str) -> str:
        """将工具调用格式化为简单文本格式，用于流式输出"""
        return f"\n「调用工具: {tool_name} 输入内容: {arguments}」\n"

    def _format_tool_calls_as_content(self, tool_calls: List[Dict[str, Any]]) -> str:
        """将工具调用列表格式化为文本，用于兼容模式下写入 assistant content"""
        if not tool_calls:
            return ""
        parts = []
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            args = fn.get("arguments", "{}")
            parts.append(f"<<<tool_call>>>\n<tool name=\"{name}\">{args}</tool>\n<<</tool_call>>>")
        return "\n".join(parts)

    def _format_tool_result_as_user_content(self, tool_call_id: str, result_text: str) -> str:
        """兼容模式：将工具结果包装为 user 消息内容"""
        return f"<tool_result id=\"{tool_call_id}\">\n{result_text}\n</tool_result>"

    def _append_tool_result_message(self, messages: List[Dict[str, Any]], tool_call_id: str, result_text: str):
        """将工具结果追加到消息列表（兼容模式下使用 user 角色）"""
        content = result_text or "工具执行失败"
        if self.compatibility_mode:
            messages.append({
                "role": "user",
                "tool_call_id": tool_call_id,
                "_is_tool_result": True,
                "content": self._format_tool_result_as_user_content(tool_call_id, content)
            })
        else:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content
            })

    def _build_xml_tool_guide(self, tools: Optional[List[Dict[str, Any]]]) -> str:
        """构建供模型使用的工具说明（名称 + 参数 Schema）"""
        if not tools:
            return "当前可用工具：无。禁止发起工具调用。"

        lines = ["当前可用工具（只能从以下 name 中选择）："]
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function", {}) or {}
            name = fn.get("name", "")
            if not name:
                continue
            desc = fn.get("description", "") or ""
            params = fn.get("parameters", {}) or {}
            params_json = json.dumps(params, ensure_ascii=False)
            lines.append(f"- name: {name}")
            if desc:
                lines.append(f"  description: {desc}")
            lines.append(f"  parameters_json_schema: {params_json}")

        if len(lines) == 1:
            return "当前可用工具：无。禁止发起工具调用。"

        return "\n".join(lines)

    def _prepare_messages_for_backend(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """在发送给后端模型前，对消息格式做兼容转换"""
        if not self.compatibility_mode:
            return messages

        converted: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "user")
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""

            if role == "tool" or msg.get("_is_tool_result"):
                tool_call_id = msg.get("tool_call_id", "")
                if msg.get("_is_tool_result"):
                    converted.append({"role": "user", "content": content})
                else:
                    converted.append({
                        "role": "user",
                        "content": self._format_tool_result_as_user_content(tool_call_id, content)
                    })
                continue

            if role == "assistant":
                normalized_msg = {
                    "role": "assistant",
                }
                if reasoning:
                    # 兼容模式后端不认 reasoning_content，拼进 content
                    normalized_msg["content"] = f"<thinking>\n{reasoning}\n</thinking>\n{content}"
                else:
                    normalized_msg["content"] = content
                if msg.get("tool_calls"):
                    normalized_msg["tool_calls"] = msg.get("tool_calls")
                converted.append(normalized_msg)
                continue

            converted.append({"role": role, "content": content})

        if self.force_xml_json_tool_call:
            tool_guide = self._build_xml_tool_guide(tools)
            force_prompt = (
                "## 正文输出的严格规则（最高优先级，必须无条件遵守）\n\n"
                "你的正文（即正式回复）只有两种合法状态，不存在第三种。这条规则适用于每一轮回复，包括第一轮：\n\n"
                "**状态A - 工具调用轮**：正文的第一个字符必须是「<」（即<<<tool_call>>>的开头），最后以<<</tool_call>>>结束。\n"
                "  - 正文中除了工具调用块，不允许出现任何其他字符。\n"
                "  - 禁止在工具调用块前写任何文字，包括但不限于：问候语、计划说明、过渡语、解释。\n"
                "  - 禁止在工具调用块后写任何文字。\n"
                "  - 所有思考、分析、计划、自言自语必须且只能放在思维链（thinking）中。\n"
                "  - 需要同时调用多个工具时，把多个<tool>标签放在同一个<<<tool_call>>>块内，不要分成多轮。\n"
                "  - 错误示例1：「主人，让我搜索一下...<<<tool_call>>>...」← 禁止！前面有文字。\n"
                "  - 错误示例2：「<<<tool_call>>>...<<</tool_call>>>\\n先从这个开始...」← 禁止！后面有文字。\n"
                "  - 正确示例：<<<tool_call>>>\\n<tool name=\"a\">{...}</tool>\\n<tool name=\"b\">{...}</tool>\\n<<</tool_call>>>\n\n"
                "**状态B - 最终回答轮**：正文是给用户的完整回答，不包含任何工具调用块。\n\n"
                "违反以上规则会导致系统解析失败，工具无法执行。\n\n"
                "## 推理与行动框架\n\n"
                "在每次行动前，在思维链中完成以下推理（不要写在正文里）：\n\n"
                "1. **信息充分性评估**：回答用户问题还缺少什么信息？如果不充分，必须调用工具，禁止勉强作答。\n\n"
                "2. **工具链规划**：\n"
                "   - 搜索工具返回的通常只有摘要和链接URL。要获取详细内容，必须用网页访问工具打开这些URL。\n"
                "   - 典型流程：搜索 → 访问搜索结果中的网页URL → 根据内容再搜索或访问更多 → 综合后回答。\n"
                "   - 不要仅凭搜索摘要就作答。\n\n"
                "3. **持续性**：多轮工具调用是正常的。不要因为已经调过一次就停下来。信息不够就继续调用。\n\n"
                "4. **结果评估**：每次收到工具结果后重新判断信息是否充分。充分→最终回答；不充分→继续调用。\n\n"
                "## 工具调用格式\n"
                "使用以下格式调用工具（一轮可调用多个）：\n"
                "<<<tool_call>>>\n"
                "<tool name=\"工具名\">{\"key\":\"value\"}</tool>\n"
                "<<</tool_call>>>\n\n"
                "## 格式要求\n"
                "- 工具名必须与下面列表中的 name 完全一致\n"
                "- <tool> 标签体必须是合法 JSON 对象\n\n"
                f"{tool_guide}\n"
            )

            if converted and converted[0].get("role") == "system":
                converted[0] = {
                    "role": "system",
                    "content": force_prompt + "\n\n" + (converted[0].get("content") or "")
                }
            else:
                converted.insert(0, {
                    "role": "system",
                    "content": force_prompt
                })

        return converted

    def _build_reasoning_delta(self, reasoning_text: str) -> Dict[str, Any]:
        """构建流式思考增量（兼容模式下提供 model_message）"""
        if self.compatibility_mode:
            return {
                "model_message": {
                    "role": "model",
                    "content": reasoning_text
                },
                "reasoning_content": reasoning_text
            }
        return {
            "reasoning_content": reasoning_text
        }

    def _build_final_message_obj(self, final_msg: Dict[str, Any]) -> Dict[str, Any]:
        """构建最终消息对象（兼容模式下提供 model_message）"""
        content = final_msg.get("content", "")
        reasoning = final_msg.get("reasoning_content", "") or ""

        message_obj = {
            "role": "assistant",
            "content": content
        }

        if self.compatibility_mode:
            message_obj["model_message"] = {
                "role": "model",
                "content": reasoning
            }
            message_obj["reasoning_content"] = reasoning
        else:
            message_obj["reasoning_content"] = reasoning

        return message_obj

    def _safe_json_loads(self, text: Any) -> Any:
        """尽力解析 JSON/JSON-like 文本"""
        if text is None:
            return None
        if isinstance(text, (dict, list)):
            return text
        if not isinstance(text, str):
            return None

        candidate = text.strip()
        if not candidate:
            return None

        try:
            return json.loads(candidate)
        except Exception:
            pass

        # 兼容单引号风格/简单 Python 字面量
        try:
            return ast.literal_eval(candidate)
        except Exception:
            return None

    def _normalize_arguments_to_string(self, arguments: Any) -> str:
        """将 arguments 统一为 JSON 字符串"""
        if arguments is None:
            return "{}"

        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                return "{}"
            parsed = self._safe_json_loads(text)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, ensure_ascii=False)
            # 不是可解析 JSON，按文本输入兜底
            return json.dumps({"input": text}, ensure_ascii=False)

        if isinstance(arguments, (dict, list)):
            return json.dumps(arguments, ensure_ascii=False)

        return json.dumps({"input": str(arguments)}, ensure_ascii=False)

    def _normalize_tool_calls(self, tool_calls: Optional[List[Any]]) -> List[Dict[str, Any]]:
        """将多种来源的 tool_calls 统一成 OpenAI 兼容字典结构"""
        normalized: List[Dict[str, Any]] = []
        if not tool_calls:
            return normalized

        for i, tc in enumerate(tool_calls):
            try:
                if isinstance(tc, dict):
                    function_obj = tc.get("function", {}) or {}
                    name = (
                        function_obj.get("name")
                        or tc.get("name")
                        or tc.get("tool_name")
                        or tc.get("function_name")
                        or ""
                    )
                    args_raw = (
                        function_obj.get("arguments")
                        if "arguments" in function_obj
                        else tc.get("arguments", tc.get("args", tc.get("input", tc.get("params", {}))))
                    )
                    call_id = tc.get("id") or f"call_{i}"
                    call_type = tc.get("type") or "function"
                    call_index = tc.get("index", i)
                else:
                    function_obj = getattr(tc, "function", None)
                    name = getattr(function_obj, "name", "") if function_obj else ""
                    args_raw = getattr(function_obj, "arguments", "{}") if function_obj else "{}"
                    call_id = getattr(tc, "id", None) or f"call_{i}"
                    call_type = getattr(tc, "type", "function") or "function"
                    call_index = getattr(tc, "index", i)

                name = (name or "").strip()
                if not name:
                    continue

                normalized.append({
                    "id": call_id,
                    "type": call_type,
                    "index": call_index,
                    "function": {
                        "name": name,
                        "arguments": self._normalize_arguments_to_string(args_raw)
                    }
                })
            except Exception:
                continue

        return normalized

    def _get_available_tool_names(self, tools: Optional[List[Dict[str, Any]]]) -> Set[str]:
        """从工具定义列表中提取可用工具名集合"""
        if not tools:
            return set()
        names: Set[str] = set()
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function", {}) or {}
            name = fn.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
        return names

    def _resolve_tool_name(self, raw_name: str, available_tool_names: Set[str]) -> str:
        """尽量把模型输出的工具名映射到真实可用工具名（兼容短名/别名）"""
        name = (raw_name or "").strip()
        if not name or not available_tool_names:
            return name

        if name in available_tool_names:
            return name

        low = name.lower()

        # 大小写等价
        exact_ci = [n for n in available_tool_names if n.lower() == low]
        if len(exact_ci) == 1:
            return exact_ci[0]

        # 若 name 正好是 MCP 服务器名，且该服务器仅暴露一个工具，自动映射
        if self.mcp_manager and name in getattr(self.mcp_manager, "servers", {}):
            prefix = f"{name}_"
            by_server = [n for n in available_tool_names if n.startswith(prefix)]
            if len(by_server) == 1:
                return by_server[0]

        # 常见：模型给了短工具名，真实工具名是 server_tool
        suffix = f"_{low}"
        by_suffix = [n for n in available_tool_names if n.lower().endswith(suffix)]
        if len(by_suffix) == 1:
            return by_suffix[0]

        # 次优：包含匹配，选最短（尽量减少误配）
        by_contains = [n for n in available_tool_names if low in n.lower()]
        if len(by_contains) == 1:
            return by_contains[0]
        if len(by_contains) > 1:
            by_contains.sort(key=len)
            return by_contains[0]

        return name

    def _parse_non_oai_tool_calls_from_text(self, text: str, strict_mode: bool = False) -> Tuple[List[Dict[str, Any]], str]:
        """
        解析非 OAI 工具调用文本（XML+JSON 等），返回:
        - 解析出的 tool_calls
        - 清理掉工具调用片段后的文本

        strict_mode=True 时，仅识别严格包装格式：
        <<<tool_call>>>
          <tool name="xxx">{"k":"v"}</tool>
        <<</tool_call>>>
        """
        if not text:
            return [], text

        parsed_calls: List[Dict[str, Any]] = []

        def append_call(name: str, args_raw: Any):
            name = (name or "").strip()
            if not name:
                return
            parsed_calls.append({
                "id": f"xml_call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "index": len(parsed_calls),
                "function": {
                    "name": name,
                    "arguments": self._normalize_arguments_to_string(args_raw)
                }
            })

        # 先解析严格包装格式：<<<tool_call>>> ... <<</tool_call>>>
        wrapper_pattern = re.compile(
            r"<<<\s*tool_call\s*>>>(?P<body>[\s\S]*?)<<<\s*/tool_call\s*>>>",
            re.IGNORECASE
        )

        def replace_wrapper(match):
            wrapper_body = (match.group("body") or "").strip()

            tool_pattern = re.compile(
                r"<tool\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</tool>",
                re.IGNORECASE
            )

            for tmatch in tool_pattern.finditer(wrapper_body):
                attrs = html.unescape(tmatch.group("attrs") or "")
                tool_body = html.unescape(tmatch.group("body") or "").strip()

                name = ""
                name_m = re.search(r'name\s*=\s*([\'"])(.*?)\1', attrs, re.IGNORECASE | re.DOTALL)
                if name_m:
                    name = (name_m.group(2) or "").strip()

                append_call(name, tool_body)

            return ""

        cleaned_text = wrapper_pattern.sub(replace_wrapper, text)

        # 强制模式仅认严格包装格式，避免误解析
        if strict_mode:
            return parsed_calls, cleaned_text.strip()

        tag_block_pattern = re.compile(
            r"<(?P<tag>tool_call|tool|function_call|invoke|call_tool)\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</(?P=tag)>",
            re.IGNORECASE
        )

        def parse_block(attrs: str, body: str):
            attrs = html.unescape(attrs or "")
            body = html.unescape(body or "").strip()

            name = ""
            args_raw: Any = {}

            # 属性提取
            name_m = re.search(r'(?:name|tool|function)\s*=\s*([\'"])(.*?)\1', attrs, re.IGNORECASE | re.DOTALL)
            if name_m:
                name = (name_m.group(2) or "").strip()

            args_m = re.search(r'(?:arguments|args|input|params)\s*=\s*([\'"])(.*?)\1', attrs, re.IGNORECASE | re.DOTALL)
            if args_m:
                args_raw = args_m.group(2)

            # body 解析
            if body:
                body_name_m = re.search(r"<(?:name|tool|function)>([\s\S]*?)</(?:name|tool|function)>", body, re.IGNORECASE)
                if body_name_m and not name:
                    name = body_name_m.group(1).strip()

                body_args_m = re.search(r"<(?:arguments|args|input|params)>([\s\S]*?)</(?:arguments|args|input|params)>", body, re.IGNORECASE)
                if body_args_m:
                    args_raw = body_args_m.group(1).strip()
                else:
                    parsed_body = self._safe_json_loads(body)
                    if isinstance(parsed_body, dict):
                        if not name:
                            name = (
                                parsed_body.get("name")
                                or parsed_body.get("tool")
                                or parsed_body.get("tool_name")
                                or parsed_body.get("function")
                                or parsed_body.get("function_name")
                                or ""
                            )
                        args_raw = parsed_body.get(
                            "arguments",
                            parsed_body.get("args", parsed_body.get("input", parsed_body.get("params", args_raw)))
                        )
                    elif not args_m and not body_name_m:
                        # 非结构化 body 文本，作为 input 传递
                        args_raw = {"input": body}

            append_call(name, args_raw)

        def replace_tag_block(match):
            parse_block(match.group("attrs"), match.group("body"))
            return ""

        cleaned_text = tag_block_pattern.sub(replace_tag_block, cleaned_text)

        # 自闭合标签：<tool_call name="x" arguments='{"a":1}' />
        self_closing_pattern = re.compile(
            r"<(?P<tag>tool_call|tool|function_call|invoke|call_tool)\b(?P<attrs>[^>]*)/>",
            re.IGNORECASE
        )

        def replace_self_closing(match):
            parse_block(match.group("attrs"), "")
            return ""

        cleaned_text = self_closing_pattern.sub(replace_self_closing, cleaned_text)

        # 兼容中文包裹格式： 「调用工具: xxx 输入内容: {...}」
        chinese_pattern = re.compile(
            r"「\s*调用工具\s*[:：]\s*(?P<name>[^\s，,」]+)\s*输入内容\s*[:：]\s*(?P<args>[\s\S]*?)\s*」",
            re.IGNORECASE
        )

        def replace_chinese(match):
            append_call(match.group("name"), match.group("args"))
            return ""

        cleaned_text = chinese_pattern.sub(replace_chinese, cleaned_text)

        return parsed_calls, cleaned_text.strip()

    def _extract_non_oai_tool_calls(
        self,
        reasoning_content: str,
        content: str,
        available_tool_names: Optional[Set[str]] = None
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        """从 reasoning/content 中提取非 OAI 工具调用，并返回清理后的文本"""
        reasoning_calls, cleaned_reasoning = self._parse_non_oai_tool_calls_from_text(
            reasoning_content or "",
            strict_mode=self.force_xml_json_tool_call
        )
        content_calls, cleaned_content = self._parse_non_oai_tool_calls_from_text(
            content or "",
            strict_mode=self.force_xml_json_tool_call
        )

        merged = reasoning_calls + content_calls

        # 若有工具白名单，先做工具名映射，再决定是否过滤
        if available_tool_names:
            for tc in merged:
                fn = tc.get("function", {}) or {}
                old_name = fn.get("name", "")
                new_name = self._resolve_tool_name(old_name, available_tool_names)
                if new_name:
                    fn["name"] = new_name
                    tc["function"] = fn

            # 非强制模式继续严格过滤，强制 XML+JSON 模式不做过滤，避免误丢调用
            if not self.force_xml_json_tool_call:
                merged = [tc for tc in merged if tc.get("function", {}).get("name") in available_tool_names]

        # 去重（按 name + arguments）
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for tc in merged:
            fn = tc.get("function", {}) or {}
            key = (fn.get("name", ""), fn.get("arguments", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(tc)

        # 重排 index
        for i, tc in enumerate(deduped):
            tc["index"] = i

        return deduped, cleaned_reasoning, cleaned_content

    def _replace_old_tool_results(self, messages: List[Dict[str, Any]], tool_call_history: List[List[str]], keep_count: int):
        """
        将历史工具调用结果替换为占位符「调用完毕」
        保留最近 N 个“工具调用批次”的完整结果
        
        Args:
            messages: 消息列表
            tool_call_history: 按时间顺序排列的工具调用批次列表（每个批次是一轮中的 tool_call_id 列表）
            keep_count: 保留最近多少个工具调用批次的完整结果（0 表示保留当前请求中的全部批次，负数表示不保留任何）
        """
        if keep_count == 0:
            # 保留当前请求中所有批次的工具调用结果
            keep_ids = {tc_id for batch in tool_call_history for tc_id in batch}
        elif keep_count > 0:
            # 保留最近 keep_count 个批次的工具调用结果
            recent_batches = tool_call_history[-keep_count:] if tool_call_history else []
            keep_ids = {tc_id for batch in recent_batches for tc_id in batch}
        else:
            # 不保留任何工具调用结果
            keep_ids = set()
        
        for msg in messages:
            if isinstance(msg, dict):
                is_tool_result_msg = msg.get('role') == 'tool' or bool(msg.get('_is_tool_result'))
                if not is_tool_result_msg:
                    continue

                tool_call_id = msg.get('tool_call_id', '')
                # 如果不在保留列表中，替换为占位符
                if tool_call_id not in keep_ids:
                    if msg.get('_is_tool_result'):
                        msg['content'] = self._format_tool_result_as_user_content(tool_call_id, '调用完毕')
                    else:
                        msg['content'] = '调用完毕'

    def _merge_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        assistant_msg_index: int,
        new_reasoning: str,
        new_content: str,
        new_tool_calls: Optional[List]
    ):
        """合并助手消息"""
        # 删除所有 tool 消息（兼容模式下还要删除 user 工具结果消息）
        while messages and isinstance(messages[-1], dict):
            last_msg = messages[-1]
            if last_msg.get('role') == 'tool' or last_msg.get('_is_tool_result'):
                messages.pop()
                continue
            break
        
        if assistant_msg_index is not None and assistant_msg_index < len(messages):
            prev_assistant = messages[assistant_msg_index]
            
            # 保留原始的 reasoning_content，不添加工具调用JSON
            # 原因：避免AI看到并模仿{"tool_calls":...}格式，导致"增值"问题
            old_reasoning = prev_assistant.get('reasoning_content', '') or ''
            
            # 追加新的思维链
            if new_reasoning:
                combined_reasoning = old_reasoning + "\n\n" + new_reasoning if old_reasoning else new_reasoning
                prev_assistant['reasoning_content'] = combined_reasoning
            
            # 更新工具调用
            normalized_tool_calls = self._normalize_tool_calls(new_tool_calls)
            if normalized_tool_calls:
                prev_assistant['tool_calls'] = normalized_tool_calls
            else:
                if 'tool_calls' in prev_assistant:
                    del prev_assistant['tool_calls']
                prev_assistant['content'] = new_content
    
    def _execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """执行 MCP 工具"""
        if not self.mcp_manager:
            return None
        
        # 增加调用计数（每个请求独立计数）
        self.mcp_call_counter += 1
        call_number = self.mcp_call_counter
        
        # 增强控制台显示 - 调用（显示第几次调用）
        print(f"\n>>> [MCP工具调用 #{call_number}] {tool_name}")
        print(f"    参数: {json.dumps(arguments, ensure_ascii=False)}")
        
        logging.info(f"[MCP Tool Execute #{call_number}] 调用工具 '{tool_name}'，参数: {arguments}")
        result = self.mcp_manager.call_tool(tool_name, arguments)
        
        # 增强控制台显示 - 结果
        result_str = str(result or '')
        # 结果可能很长，控制台显示前8000个字符
        display_limit = 8000
        if len(result_str) > display_limit:
            display_result = result_str[:display_limit] + f"\n... (剩余 {len(result_str)-display_limit} 字符)"
        else:
            display_result = result_str
            
        print(f"<<< [MCP工具结果 #{call_number}] {tool_name}")
        print(f"    {display_result}\n")
        
        logging.info(f"[MCP Tool Result #{call_number}] 工具 '{tool_name}' 返回结果: {result_str}")
        return result
    
    def _is_mcp_tool(self, tool_name: str) -> bool:
        """检查是否是 MCP 工具"""
        if not self.mcp_manager:
            return False
        return tool_name in self.mcp_manager.tools
    
    def process_request_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str = "deepseek-reasoner",
        tools: Optional[List[Dict[str, Any]]] = None,
        execute_mcp_tools: bool = True,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        处理流式聊天补全请求
        
        Args:
            messages: 消息列表
            model: 模型名称
            tools: 工具列表（可选，如果启用 MCP 会自动合并 MCP 工具）
            execute_mcp_tools: 是否自动执行 MCP 工具调用
            **kwargs: 其他参数
        
        Yields:
            SSE 格式的流式响应数据
        """
        # 合并 MCP 工具到工具列表
        combined_tools = list(tools) if tools else []
        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_openai_tools()
            combined_tools.extend(mcp_tools)
        
        # 如果没有任何工具，设置为 None
        if not combined_tools:
            combined_tools = None

        request_tools = combined_tools
        if self.compatibility_mode and not self.compatibility_send_tools:
            request_tools = None

        available_tool_names = self._get_available_tool_names(combined_tools)
        
        messages_copy = [msg.copy() for msg in messages]
        iteration = 0
        max_iterations = CONFIG.get('max_iterations', 100)
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created_time = int(time.time())
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        logging.info(f"[请求开始] Chat ID: {chat_id}, 模型: {model}, 消息数: {len(messages_copy)}, 流式: True")
        
        # 工具调用历史追踪（按时间顺序，按“批次”记录）
        tool_call_history: List[List[str]] = []  # 每个元素是一轮工具调用产生的 tool_call_id 列表
        keep_tool_results_count = CONFIG.get('keep_tool_results_count', 0)
        
        print(f"\n{'='*60}")
        print(f"[流式请求] Chat ID: {chat_id}")
        print(f"[流式请求] 模型: {model}")
        print(f"[流式请求] 消息数: {len(messages_copy)}")
        print(f"[流式请求] 保留工具结果数: {keep_tool_results_count}")
        print(f"{'='*60}\n")
        
        while iteration < max_iterations:
            # 在调用 API 前，替换历史工具结果为占位符
            self._replace_old_tool_results(messages_copy, tool_call_history, keep_tool_results_count)
            
            logging.info(f"[迭代 {iteration+1}/{max_iterations}] 开始 Chat ID: {chat_id}")
            # 调用 DeepSeek API（流式）
            backend_messages = self._prepare_messages_for_backend(messages_copy, combined_tools)
            stream_response = self.client.chat.completions.create(
                model=model,
                messages=backend_messages,
                tools=request_tools,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs
            )
            
            # 收集流式响应
            reasoning_content = ""
            content = ""
            tool_calls_data = {}  # id -> {function: {name, arguments}, type, index}
            finish_reason = None

            # ---- 流式 content 缓冲拦截器（仅 force_xml_json_tool_call 时启用） ----
            # 状态机:
            #   "normal"   – 正常透传
            #   "buffering" – 遇到 < 后缓存，等待判断是否为 <<<tool_call>>>
            #   "swallowing" – 已确认进入 <<<tool_call>>> 块，吞掉直到 <<</tool_call>>>
            _buf_state = "normal"       # 当前状态
            _buf_pending = ""           # 缓存的待判断文本
            # <<<tool_call>>> 的完整前缀序列
            _TOOL_OPEN = "<<<tool_call>>>"
            _TOOL_CLOSE = "<<</tool_call>>>"
            _swallow_buf = ""           # swallowing 状态下累积的文本（用于检测结束标记）

            def _make_content_chunk(text: str) -> str:
                """构造一个 content delta SSE 行"""
                cd = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "logprobs": None,
                        "finish_reason": None
                    }]
                }
                return f"data: {json.dumps(cd, ensure_ascii=False)}\n\n"

            # 用列表存，因为 generator 里不能直接 nonlocal 赋值给外层简单变量（Python 2 式问题已不存在，但保持清晰）
            # 实际上 Python 3 用 nonlocal 即可，这里直接用 nonlocal。
            # _pending_yields: 每次处理完一个 delta.content 后要 yield 的 SSE 行列表
            # 我们不在内部函数里 yield，而是返回要 yield 的列表。

            def _feed_content(new_text: str) -> List[str]:
                """
                将新的 content 增量喂入缓冲拦截器，返回应当 yield 的 SSE 行列表。
                同时更新外层 content 变量。
                """
                nonlocal _buf_state, _buf_pending, _swallow_buf, content
                
                if not self.force_xml_json_tool_call:
                    # 非强制模式，直接透传
                    content += new_text
                    return [_make_content_chunk(new_text)]

                results: List[str] = []
                # 逐字符处理（token 可能含多个字符）
                for ch in new_text:
                    content += ch  # 始终累积到 content（后面解析时会清理）

                    if _buf_state == "normal":
                        if ch == '<':
                            _buf_pending = ch
                            _buf_state = "buffering"
                        else:
                            results.append(_make_content_chunk(ch))

                    elif _buf_state == "buffering":
                        _buf_pending += ch
                        # 检查是否仍然是 <<<tool_call>>> 的前缀
                        prefix_len = len(_buf_pending)
                        if _TOOL_OPEN[:prefix_len] == _buf_pending:
                            # 仍然匹配前缀，继续缓冲
                            if prefix_len == len(_TOOL_OPEN):
                                # 完整匹配 <<<tool_call>>>，进入吞噬模式
                                _buf_state = "swallowing"
                                _swallow_buf = ""
                                _buf_pending = ""
                        else:
                            # 不匹配，flush 缓存
                            flush_text = _buf_pending
                            _buf_pending = ""
                            _buf_state = "normal"
                            results.append(_make_content_chunk(flush_text))

                    elif _buf_state == "swallowing":
                        _swallow_buf += ch
                        # 检查是否遇到了结束标记
                        if _swallow_buf.endswith(_TOOL_CLOSE):
                            # 工具调用块完整结束，丢弃整个块（content 里已有，后面解析时会清理）
                            _buf_state = "normal"
                            _swallow_buf = ""

                return results

            def _flush_pending() -> List[str]:
                """流结束时，flush 所有缓冲中的内容"""
                nonlocal _buf_state, _buf_pending, _swallow_buf
                results: List[str] = []
                if _buf_state == "buffering" and _buf_pending:
                    results.append(_make_content_chunk(_buf_pending))
                    _buf_pending = ""
                # swallowing 状态下的残留内容不 flush（属于未闭合的工具调用，丢弃）
                _buf_state = "normal"
                _swallow_buf = ""
                return results
            # ---- 缓冲拦截器定义结束 ----

            for chunk in stream_response:
                # 累计 token 使用
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = chunk.usage
                    if getattr(usage, 'total_tokens', 0) > 0:
                        usage_msg = f"[Token消耗] Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}"
                        print(f"\n{usage_msg}")
                        logging.info(f"{usage_msg} (Chat ID: {chat_id})")
                    total_usage["prompt_tokens"] += getattr(usage, 'prompt_tokens', 0)
                    total_usage["completion_tokens"] += getattr(usage, 'completion_tokens', 0)
                    total_usage["total_tokens"] += getattr(usage, 'total_tokens', 0)
                
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                chunk_finish_reason = chunk.choices[0].finish_reason
                
                if chunk_finish_reason:
                    finish_reason = chunk_finish_reason
                
                # 处理 reasoning_content 增量
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                    # 发送 reasoning_content chunk
                    chunk_data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": self._build_reasoning_delta(delta.reasoning_content),
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                # 处理 content 增量（通过缓冲拦截器）
                if hasattr(delta, 'content') and delta.content:
                    for sse_line in _feed_content(delta.content):
                        yield sse_line
                
                # 处理 tool_calls 增量
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc in delta.tool_calls:
                        tc_index = tc.index if hasattr(tc, 'index') else 0
                        if tc_index not in tool_calls_data:
                            tool_calls_data[tc_index] = {
                                "id": tc.id if hasattr(tc, 'id') and tc.id else f"call_{tc_index}",
                                "type": tc.type if hasattr(tc, 'type') else "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        
                        if hasattr(tc, 'id') and tc.id:
                            tool_calls_data[tc_index]["id"] = tc.id
                        
                        if hasattr(tc, 'function'):
                            if hasattr(tc.function, 'name') and tc.function.name:
                                tool_calls_data[tc_index]["function"]["name"] += tc.function.name
                            if hasattr(tc.function, 'arguments') and tc.function.arguments:
                                tool_calls_data[tc_index]["function"]["arguments"] += tc.function.arguments

            # 流结束，flush 缓冲拦截器残留
            for sse_line in _flush_pending():
                yield sse_line
            
            # 流式响应结束后，检查是否有工具调用
            tool_calls_list = [tool_calls_data[i] for i in sorted(tool_calls_data.keys())] if tool_calls_data else []
            tool_calls_list = self._normalize_tool_calls(tool_calls_list)

            # 强制 XML+JSON 时，忽略 OAI tool_calls
            if self.force_xml_json_tool_call:
                tool_calls_list = []

            # 兼容非 OAI 工具调用格式（XML+JSON 等）
            # 保存未清理的原始文本（兼容模式下回传给模型用）
            raw_content_before_extract = content
            raw_reasoning_before_extract = reasoning_content
            if not tool_calls_list:
                parsed_tool_calls, cleaned_reasoning, cleaned_content = self._extract_non_oai_tool_calls(
                    reasoning_content,
                    content,
                    available_tool_names
                )
                if parsed_tool_calls:
                    tool_calls_list = parsed_tool_calls
                    reasoning_content = cleaned_reasoning
                    content = cleaned_content
                    finish_reason = "tool_calls"
            
            # 如果没有工具调用，结束流式响应
            if not tool_calls_list or finish_reason == "stop":
                # 发送结束 chunk（包含usage统计）
                final_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": finish_reason or "stop"
                    }],
                    "usage": total_usage
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                
                end_msg = f"[请求结束] 原因: {finish_reason or 'stop'}, 总消耗: {total_usage}"
                print(f"{end_msg}\n")
                logging.info(f"{end_msg} (Chat ID: {chat_id})")
                
                yield "data: [DONE]\n\n"
                return
            
            # 检查是否有 MCP 工具调用需要执行
            if tool_calls_list and execute_mcp_tools and self.mcp_manager:
                mcp_tool_calls = []
                non_mcp_tool_calls = []
                
                for tc in tool_calls_list:
                    if self._is_mcp_tool(tc["function"]["name"]):
                        mcp_tool_calls.append(tc)
                    else:
                        non_mcp_tool_calls.append(tc)
                
                # 执行 MCP 工具调用
                if mcp_tool_calls:
                    # 将工具调用转换为简单文本格式（仅用于发送给用户）
                    tool_call_texts = []
                    for tc in mcp_tool_calls:
                        name = tc["function"]["name"]
                        args_str = tc["function"]["arguments"]
                        logging.info(f"[工具调用计划] Chat ID: {chat_id}, 工具: {name}, 参数: {args_str}")
                        tool_call_texts.append(self._format_tool_call_text(name, args_str))
                    
                    tools_text_for_user = "\n".join(tool_call_texts) + "\n"
                    
                    # 发送简单文本格式的工具调用作为思考增量（仅发送给用户，不保存到历史）
                    chunk_data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": self._build_reasoning_delta(tools_text_for_user),
                            "logprobs": None, "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    
                    # 构建助手消息
                    # 兼容模式下用原始未清理的 content（含工具调用文本），让模型看到自己调了什么
                    # 非兼容模式用清理后的 content
                    if self.compatibility_mode:
                        assistant_msg = {
                            "role": "assistant",
                            "content": raw_content_before_extract or "",
                            "reasoning_content": raw_reasoning_before_extract or "",
                            "tool_calls": tool_calls_list
                        }
                    else:
                        assistant_msg = {
                            "role": "assistant",
                            "content": content or "",
                            "reasoning_content": reasoning_content or "",
                            "tool_calls": tool_calls_list
                        }
                    # 注意：不再使用条件判断，而是始终包含 reasoning_content
                    # 这样可以避免在思维链开始就调用工具时出现 "Missing reasoning_content field" 错误
                    
                    messages_copy.append(assistant_msg)
                    
                    current_batch_ids: List[str] = []
                    for tc in mcp_tool_calls:
                        tool_call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                        args_raw = tc.get("function", {}).get("arguments", "{}")
                        parsed_args = self._safe_json_loads(args_raw)
                        if isinstance(parsed_args, dict):
                            args = parsed_args
                        elif isinstance(parsed_args, list):
                            args = {"items": parsed_args}
                        elif isinstance(args_raw, str) and args_raw.strip():
                            args = {"input": args_raw.strip()}
                        else:
                            args = {}
                            
                        result = self._execute_mcp_tool(tc["function"]["name"], args)
                        
                        # 记录到当前批次
                        current_batch_ids.append(tool_call_id)
                        
                        # 添加工具结果到消息（兼容模式下为 user 消息）
                        self._append_tool_result_message(
                            messages_copy,
                            tool_call_id,
                            result or "工具执行失败"
                        )
                    
                    # 当前这一轮中的多个工具调用，记为一个批次
                    if current_batch_ids:
                        tool_call_history.append(current_batch_ids)
                    
                    iteration += 1
                    continue  # 继续下一轮对话
                
                # 如果只有非 MCP 工具调用，结束并返回工具调用请求
                if non_mcp_tool_calls:
                    final_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "tool_calls"
                        }],
                        "usage": total_usage
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    print(f"[流式请求结束] 非MCP工具调用，总消耗: {total_usage}\n")
                    yield "data: [DONE]\n\n"
                    return
            
            # 如果有工具调用但不自动执行，结束并返回
            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": "tool_calls"
                }],
                "usage": total_usage
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            print(f"[流式请求结束] 自动执行禁用，总消耗: {total_usage}\n")
            yield "data: [DONE]\n\n"
            return
        
        # 达到最大迭代次数
        error_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "[达到最大工具调用迭代次数]"
                },
                "logprobs": None,
                "finish_reason": "length"
            }],
            "usage": total_usage
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        print(f"[流式请求结束] 达到最大迭代次数，总消耗: {total_usage}\n")
        yield "data: [DONE]\n\n"
    
    def process_request(
        self,
        messages: List[Dict[str, Any]],
        model: str = "deepseek-reasoner",
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        execute_mcp_tools: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理聊天补全请求（非流式）
        
        Args:
            messages: 消息列表
            model: 模型名称
            tools: 工具列表（可选，如果启用 MCP 会自动合并 MCP 工具）
            stream: 是否流式（此方法仅处理非流式，流式请使用 process_request_stream）
            execute_mcp_tools: 是否自动执行 MCP 工具调用
            **kwargs: 其他参数
        """
        
        # 合并 MCP 工具到工具列表
        combined_tools = list(tools) if tools else []
        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_openai_tools()
            combined_tools.extend(mcp_tools)
        
        # 如果没有任何工具，设置为 None
        if not combined_tools:
            combined_tools = None

        request_tools = combined_tools
        if self.compatibility_mode and not self.compatibility_send_tools:
            request_tools = None

        available_tool_names = self._get_available_tool_names(combined_tools)
        
        messages_copy = [msg.copy() for msg in messages]
        iteration = 0
        max_iterations = CONFIG.get('max_iterations', 100)
        assistant_msg_index = None
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        logging.info(f"[请求开始] Chat ID: {chat_id}, 模型: {model}, 消息数: {len(messages_copy)}, 流式: False")
        
        # 工具调用历史追踪（按时间顺序，按“批次”记录）
        tool_call_history: List[List[str]] = []  # 每个元素是一轮工具调用产生的 tool_call_id 列表
        keep_tool_results_count = CONFIG.get('keep_tool_results_count', 0)
        
        while iteration < max_iterations:
            # 在调用 API 前，替换历史工具结果为占位符
            self._replace_old_tool_results(messages_copy, tool_call_history, keep_tool_results_count)
            
            # 调用 DeepSeek API（使用合并后的工具列表）
            backend_messages = self._prepare_messages_for_backend(messages_copy, combined_tools)
            response = self.client.chat.completions.create(
                model=model,
                messages=backend_messages,
                tools=request_tools,
                **kwargs
            )
            
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            
            # 累计 token 使用（保留详细信息）
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                total_usage["prompt_tokens"] += getattr(usage, 'prompt_tokens', 0)
                total_usage["completion_tokens"] += getattr(usage, 'completion_tokens', 0)
                total_usage["total_tokens"] += getattr(usage, 'total_tokens', 0)
                
                # 保留详细的 usage 信息（如果存在）
                if hasattr(usage, 'prompt_tokens_details'):
                    if "prompt_tokens_details" not in total_usage:
                        total_usage["prompt_tokens_details"] = {"cached_tokens": 0}
                    total_usage["prompt_tokens_details"]["cached_tokens"] += getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
                
                if hasattr(usage, 'completion_tokens_details'):
                    if "completion_tokens_details" not in total_usage:
                        total_usage["completion_tokens_details"] = {"reasoning_tokens": 0}
                    total_usage["completion_tokens_details"]["reasoning_tokens"] += getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)
                
                if hasattr(usage, 'prompt_cache_hit_tokens'):
                    if "prompt_cache_hit_tokens" not in total_usage:
                        total_usage["prompt_cache_hit_tokens"] = 0
                    total_usage["prompt_cache_hit_tokens"] += getattr(usage, 'prompt_cache_hit_tokens', 0)
                
                if hasattr(usage, 'prompt_cache_miss_tokens'):
                    if "prompt_cache_miss_tokens" not in total_usage:
                        total_usage["prompt_cache_miss_tokens"] = 0
                    total_usage["prompt_cache_miss_tokens"] += getattr(usage, 'prompt_cache_miss_tokens', 0)
            
            # 提取响应内容
            new_reasoning = getattr(message, 'reasoning_content', None) or ""
            new_content = message.content or ""
            new_tool_calls = self._normalize_tool_calls(getattr(message, 'tool_calls', None))

            # 强制 XML+JSON 时，忽略 OAI tool_calls
            if self.force_xml_json_tool_call:
                new_tool_calls = []

            # 兼容非 OAI 工具调用格式（XML+JSON 等）
            if not new_tool_calls:
                parsed_tool_calls, cleaned_reasoning, cleaned_content = self._extract_non_oai_tool_calls(
                    new_reasoning,
                    new_content,
                    available_tool_names
                )
                if parsed_tool_calls:
                    new_tool_calls = parsed_tool_calls
                    new_reasoning = cleaned_reasoning
                    new_content = cleaned_content
                    finish_reason = "tool_calls"
            
            if iteration == 0:
                # 首次调用：添加助手消息
                # 非流式路径：new_content 是清洗后的（工具调用已提取），
                # 兼容模式下需要把工具调用文本加回去，让模型看到自己调了什么
                if self.compatibility_mode and new_tool_calls:
                    tool_text = self._format_tool_calls_as_content(new_tool_calls)
                    assistant_content = ((new_content or "") + "\n" + tool_text).strip()
                else:
                    assistant_content = new_content
                new_msg_dict = {
                    "role": "assistant",
                    "content": assistant_content,
                    "reasoning_content": new_reasoning
                }
                if new_tool_calls:
                    new_msg_dict["tool_calls"] = new_tool_calls
                messages_copy.append(new_msg_dict)
                assistant_msg_index = len(messages_copy) - 1
            else:
                # 后续调用：合并到之前的助手消息
                self._merge_assistant_message(
                    messages_copy,
                    assistant_msg_index,
                    new_reasoning,
                    new_content,
                    new_tool_calls
                )
            
            # 如果没有工具调用，返回结果
            if not new_tool_calls or finish_reason == "stop":
                final_msg = messages_copy[assistant_msg_index]
                
                # 构建消息对象
                message_obj = self._build_final_message_obj(final_msg)
                
                # 构建响应（匹配 DeepSeek 官方格式）
                result = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message_obj,
                            "logprobs": None,
                            "finish_reason": finish_reason
                        }
                    ],
                    "usage": total_usage
                }
                
                # 添加 system_fingerprint（如果响应中有）
                if hasattr(response, 'system_fingerprint'):
                    result["system_fingerprint"] = response.system_fingerprint
                
                return result
            
            # 检查是否有 MCP 工具调用需要执行
            if new_tool_calls and execute_mcp_tools and self.mcp_manager:
                mcp_tool_calls = []
                non_mcp_tool_calls = []
                
                for tc in new_tool_calls:
                    tool_name = tc.get("function", {}).get("name", "")
                    if self._is_mcp_tool(tool_name):
                        mcp_tool_calls.append(tc)
                    else:
                        non_mcp_tool_calls.append(tc)
                
                # 执行 MCP 工具调用
                if mcp_tool_calls:
                    current_batch_ids: List[str] = []
                    for tc in mcp_tool_calls:
                        tool_call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                        args_raw = tc.get("function", {}).get("arguments", "{}")
                        parsed_args = self._safe_json_loads(args_raw)
                        if isinstance(parsed_args, dict):
                            args = parsed_args
                        elif isinstance(parsed_args, list):
                            args = {"items": parsed_args}
                        elif isinstance(args_raw, str) and args_raw.strip():
                            args = {"input": args_raw.strip()}
                        else:
                            args = {}

                        result = self._execute_mcp_tool(tc.get("function", {}).get("name", ""), args)
                        
                        # 记录到当前批次
                        current_batch_ids.append(tool_call_id)
                        
                        # 添加工具结果到消息（兼容模式下为 user 消息）
                        self._append_tool_result_message(
                            messages_copy,
                            tool_call_id,
                            result or "工具执行失败"
                        )
                    
                    # 当前这一轮中的多个工具调用，记为一个批次
                    if current_batch_ids:
                        tool_call_history.append(current_batch_ids)
                    
                    iteration += 1
                    continue  # 继续下一轮对话
                
                # 如果只有非 MCP 工具调用，返回给客户端处理
                if non_mcp_tool_calls:
                    final_msg = messages_copy[assistant_msg_index]
                    
                    message_obj = self._build_final_message_obj(final_msg)
                    message_obj["tool_calls"] = non_mcp_tool_calls
                    
                    result = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": message_obj,
                                "logprobs": None,
                                "finish_reason": "tool_calls"
                            }
                        ],
                        "usage": total_usage
                    }
                    
                    if hasattr(response, 'system_fingerprint'):
                        result["system_fingerprint"] = response.system_fingerprint
                    
                    return result
            
            # 如果有工具调用但没有提供工具函数且没有 MCP，返回工具调用请求
            # 让客户端自己执行工具
            if new_tool_calls and not combined_tools:
                final_msg = messages_copy[assistant_msg_index]
                
                message_obj = self._build_final_message_obj(final_msg)
                message_obj["tool_calls"] = final_msg.get("tool_calls", [])
                
                result = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message_obj,
                            "logprobs": None,
                            "finish_reason": "tool_calls"
                        }
                    ],
                    "usage": total_usage
                }
                
                if hasattr(response, 'system_fingerprint'):
                    result["system_fingerprint"] = response.system_fingerprint
                
                return result
            
            # 如果有工具调用，但这是代理服务器，我们不执行工具
            # 返回工具调用请求给客户端
            final_msg = messages_copy[assistant_msg_index]
            
            message_obj = self._build_final_message_obj(final_msg)
            message_obj["tool_calls"] = final_msg.get("tool_calls", [])
            
            result = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message_obj,
                        "logprobs": None,
                        "finish_reason": "tool_calls"
                    }
                ],
                "usage": total_usage
            }
            
            if hasattr(response, 'system_fingerprint'):
                result["system_fingerprint"] = response.system_fingerprint
            
            return result
        
        # 达到最大迭代次数
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "达到最大迭代次数"
                    },
                    "finish_reason": "length"
                }
            ],
            "usage": total_usage
        }


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """处理聊天补全请求"""
    global mcp_manager, CONFIG
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        # 验证访问密钥并获取 API Key
        auth_header = request.headers.get('Authorization', '')
        is_valid, api_key, error_msg = validate_access_key(auth_header)
        if not is_valid:
            return jsonify({"error": {"message": error_msg, "type": "auth_error"}}), 401
        
        # 提取参数
        messages = data.get('messages', [])
        model = data.get('model')
        
        # 模型参数是必须的
        if not model:
            return jsonify({"error": {"message": "Missing required parameter: model", "type": "invalid_request_error"}}), 400
        
        # 按模型解析路由与兼容开关
        route = resolve_model_route(model, api_key)
        resolved_api_key = route.get("api_key") or api_key
        compatibility_mode = bool(route.get("compatibility_mode", False))
        force_xml_json_tool_call = bool(route.get("force_xml_json_tool_call", False))
        route_name = route.get("route_name", "default")

        messages = [msg.copy() if isinstance(msg, dict) else msg for msg in messages]

        # 兼容模式：支持客户端传入 model_message / model 角色
        if compatibility_mode:
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role == "assistant":
                    model_message = msg.get("model_message")
                    if isinstance(model_message, dict) and not msg.get("reasoning_content"):
                        model_content = model_message.get("content")
                        if isinstance(model_content, str):
                            messages[i]["reasoning_content"] = model_content
                elif role == "model":
                    model_content = msg.get("content") or ""
                    messages[i] = {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": model_content
                    }
        else:
            # 原模式：保持现有 DeepSeek 预处理逻辑
            # 处理assistant消息：
            # 1. DeepSeek thinking模式强制要求所有assistant消息必须有reasoning_content字段（即使为空）
            # 2. 用户的特殊要求：只在最后一条且前面没有别的消息的情况下，将content转为reasoning_content（用于prefill）

            # 预处理：角色修正
            # 策略：DeepSeek 模型对中间的 System 消息支持不佳，且 Prefill 要求前一条必须是 User。
            # 将非首条的 System 消息转换为 User 消息，以构建标准的 User-Assistant 对话流。
            system_convert_count = 0
            for i in range(1, len(messages)):
                if isinstance(messages[i], dict) and messages[i].get('role') == 'system':
                    messages[i]['role'] = 'user'
                    system_convert_count += 1
            
            if system_convert_count > 0:
                 print(f"[角色修正] 已将 {system_convert_count} 条中间 System 消息转换为 User 消息，以适配模型上下文规范", file=sys.stderr, flush=True)

            # 第一步：确保所有 assistant 消息都有 reasoning_content 字段
            fixed_count = 0
            assistant_count = 0
            for i, msg in enumerate(messages):
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    assistant_count += 1
                    if 'reasoning_content' not in msg:
                        messages[i]['reasoning_content'] = ''
                        fixed_count += 1
            
            if fixed_count > 0:
                print(f"[消息补全] 发现 {assistant_count} 条 Assistant 消息，已为其中 {fixed_count} 条补充缺失的 reasoning_content 字段", file=sys.stderr, flush=True)

            # 第二步：执行特定的转换逻辑（Prefill 引导）
            # 只要最后一条消息是 assistant，就将其 content 转为 reasoning_content
            converted_count = 0
            if messages:
                last_msg_index = len(messages) - 1
                last_msg = messages[last_msg_index]
                
                if isinstance(last_msg, dict) and last_msg.get('role') == 'assistant':
                    # 如果有content，进行转换（reasoning_content 已在第一步保证存在）
                    if 'content' in last_msg and last_msg['content']:
                        content_len = len(last_msg['content'])
                        # 将内容追加到 reasoning
                        current_reasoning = last_msg.get('reasoning_content', '')
                        new_reasoning = (current_reasoning + "\n" + last_msg['content']).strip()
                        messages[last_msg_index]['reasoning_content'] = new_reasoning
                        messages[last_msg_index]['content'] = ''
                        converted_count += 1
                        print(f"[Prefill 转换] 触发引导：已将最后一条 Assistant 消息 content ({content_len} 字符) 移至 reasoning_content", file=sys.stderr, flush=True)
                    else:
                         print(f"[Prefill 状态] 最后一条 Assistant 消息无 content，无需转换", file=sys.stderr, flush=True)

            # 调试日志：打印最终消息列表摘要
            print(f"\n[消息摘要] 准备发送 {len(messages)} 条消息:", file=sys.stderr, flush=True)
            for i, msg in enumerate(messages):
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content_len = len(msg.get('content') or '')
                    reasoning_len = len(msg.get('reasoning_content') or '')
                    tool_calls_count = len(msg.get('tool_calls') or [])
                    print(f"  [{i}] {role}: content={content_len} chars, reasoning={reasoning_len} chars, tools={tool_calls_count}", file=sys.stderr, flush=True)
            
            if converted_count > 0:
                print(f"[消息转换] 总共转换了 {converted_count} 条assistant消息")
        
        # 添加系统提示词（如果启用）
        if CONFIG.get('system_prompt_enabled', False):
            system_prompt = CONFIG.get('system_prompt', '')
            if system_prompt:
                # 检查消息数组开头是否已有 system 消息
                if messages and messages[0].get('role') == 'system':
                    # 将系统提示词追加到现有 system 消息
                    messages = messages.copy()
                    messages[0] = messages[0].copy()
                    messages[0]['content'] = system_prompt + "\n\n" + messages[0]['content']
                else:
                    # 在开头添加新的 system 消息
                    messages = [{"role": "system", "content": system_prompt}] + messages
        
        tools = data.get('tools')
        stream = data.get('stream', False)
        execute_mcp_tools = data.get('execute_mcp_tools', CONFIG.get('auto_execute_mcp_tools', True))
        
        # 其他参数
        kwargs = {}
        for key in ['temperature', 'top_p', 'max_tokens', 'presence_penalty', 'frequency_penalty']:
            if key in data:
                kwargs[key] = data[key]
        
        # 创建代理处理器（传入 MCP 管理器 + 路由配置）
        proxy = DeepSeekProxy(
            resolved_api_key,
            mcp_manager,
            chat_completions_url=route.get("chat_completions_url"),
            compatibility_mode=compatibility_mode,
            route_name=route_name,
            force_xml_json_tool_call=force_xml_json_tool_call
        )
        
        # 流式响应
        if stream:
            def generate():
                try:
                    for chunk in proxy.process_request_stream(
                        messages=messages,
                        model=model,
                        tools=tools,
                        execute_mcp_tools=execute_mcp_tools,
                        **kwargs
                    ):
                        yield chunk
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                            "code": "internal_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        
        # 非流式响应
        response = proxy.process_request(
            messages=messages,
            model=model,
            tools=tools,
            stream=False,
            execute_mcp_tools=execute_mcp_tools,
            **kwargs
        )
        
        if response and "usage" in response and response["usage"]:
            print(f"[Tokens] Total usage for non-streamed request: {response['usage']}")
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error"
            }
        }), 500


@app.route('/v1/balance', methods=['GET'])
@app.route('/v1/credits', methods=['GET'])
@app.route('/user/balance', methods=['GET'])
def get_balance():
    """查询账号余额"""
    global CONFIG
    
    # 验证访问密钥并获取 API Key
    auth_header = request.headers.get('Authorization', '')
    is_valid, api_key, error_msg = validate_access_key(auth_header)
    if not is_valid:
        return jsonify({"error": {"message": error_msg, "type": "auth_error"}}), 401
    
    balance_url = "https://api.deepseek.com/user/balance"
    
    try:
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        response = requests.get(balance_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": {
                    "message": f"Failed to fetch balance: {response.status_code}",
                    "type": "api_error"
                }
            }), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({
            "error": {
                "message": "Request to balance API timed out",
                "type": "timeout_error"
            }
        }), 504
    except Exception as e:
        return jsonify({
            "error": {
                "message": f"Failed to fetch balance: {str(e)}",
                "type": "server_error"
            }
        }), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型（支持多路由聚合）"""
    global CONFIG

    auth_header = request.headers.get('Authorization', '')
    user_key = auth_header[7:] if auth_header.startswith('Bearer ') else ''
    default_key = CONFIG.get('api_key') or user_key

    # 构建 models 源列表（默认 + model_routes）
    sources: Dict[str, str] = {}
    default_models_url = CONFIG.get('models_url', 'https://api.deepseek.com/v1/models')
    sources[default_models_url] = default_key

    routes = CONFIG.get("model_routes", [])
    if isinstance(routes, list):
        for route in routes:
            if not isinstance(route, dict):
                continue
            route_models_url = route.get("models_url") or default_models_url
            route_key = route.get("api_key") or default_key
            # 同 URL 优先保留有 key 的配置
            if route_models_url not in sources or (not sources[route_models_url] and route_key):
                sources[route_models_url] = route_key

    aggregated_data: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    first_error: Optional[Tuple[int, str]] = None

    for models_url, key in sources.items():
        payload, err = fetch_models_from_backend(models_url, key or "")
        if err:
            if first_error is None:
                first_error = err
            continue

        if not isinstance(payload, dict):
            continue

        model_items = payload.get("data", [])
        if not isinstance(model_items, list):
            continue

        for item in model_items:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                if model_id in seen_ids:
                    continue
                seen_ids.add(model_id)
            aggregated_data.append(item)

    if aggregated_data:
        return jsonify({
            "object": "list",
            "data": aggregated_data
        })

    if first_error:
        return jsonify({
            "error": {
                "message": first_error[1],
                "type": "api_error"
            }
        }), first_error[0]

    return jsonify({
        "error": {
            "message": "No API key available to fetch models",
            "type": "auth_error"
        }
    }), 401


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    global mcp_manager
    
    status = {"status": "ok"}
    
    if mcp_manager:
        status["mcp"] = {
            "available": True,
            "servers": mcp_manager.get_status(),
            "tools_count": len(mcp_manager.tools)
        }
    else:
        status["mcp"] = {"available": False}
    
    return jsonify(status)


# ==================== MCP 管理 API ====================

@app.route('/v1/mcp/status', methods=['GET'])
def mcp_status():
    """获取 MCP 状态"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    if not mcp_manager:
        return jsonify({"error": "MCP 管理器未初始化"}), 503
    
    return jsonify({
        "servers": mcp_manager.get_status(),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "server": tool.server_name
            }
            for tool in mcp_manager.tools.values()
        ]
    })


@app.route('/v1/mcp/tools', methods=['GET'])
def mcp_tools():
    """获取 MCP 工具列表（OpenAI 格式）"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"tools": []})
    
    return jsonify({
        "tools": mcp_manager.get_openai_tools()
    })


@app.route('/v1/mcp/servers', methods=['GET'])
def mcp_list_servers():
    """列出所有 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"servers": {}})
    
    return jsonify({
        "servers": mcp_manager.get_status()
    })


@app.route('/v1/mcp/servers', methods=['POST'])
def mcp_add_server():
    """添加 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    if not mcp_manager:
        return jsonify({"error": "MCP 管理器未初始化"}), 503
    
    data = request.get_json()
    name = data.get('name')
    command = data.get('command')
    args = data.get('args', [])
    description = data.get('description', '')
    enabled = data.get('enabled', True)
    env = data.get('env')
    
    if not name or not command:
        return jsonify({"error": "缺少必要参数: name, command"}), 400
    
    success = mcp_manager.add_server(name, command, args, description, enabled, env)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"服务器 {name} 已添加",
            "status": mcp_manager.get_status().get(name)
        })
    else:
        return jsonify({"error": f"添加服务器 {name} 失败"}), 500


@app.route('/v1/mcp/servers/<name>', methods=['DELETE'])
def mcp_remove_server(name: str):
    """移除 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    success = mcp_manager.remove_server(name)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"服务器 {name} 已移除"
        })
    else:
        return jsonify({"error": f"服务器 {name} 不存在"}), 404


@app.route('/v1/mcp/servers/<name>/start', methods=['POST'])
def mcp_start_server(name: str):
    """启动 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    success = mcp_manager.start_server(name)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"服务器 {name} 已启动",
            "status": mcp_manager.get_status().get(name)
        })
    else:
        return jsonify({"error": f"启动服务器 {name} 失败"}), 500


@app.route('/v1/mcp/servers/<name>/stop', methods=['POST'])
def mcp_stop_server(name: str):
    """停止 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    mcp_manager.stop_server(name)
    
    return jsonify({
        "success": True,
        "message": f"服务器 {name} 已停止"
    })


@app.route('/v1/mcp/reload', methods=['POST'])
def mcp_reload():
    """重新加载 MCP 配置"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    if mcp_manager:
        mcp_manager.reload_config()
        return jsonify({
            "success": True,
            "message": "MCP 配置已重新加载",
            "servers": mcp_manager.get_status()
        })
    else:
        return jsonify({"error": "MCP 管理器未初始化"}), 503


@app.route('/v1/mcp/servers/all', methods=['GET'])
def mcp_list_all_servers():
    """列出所有可用的 MCP 服务器（包括禁用的）"""
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    try:
        from mcp_servers import get_available_servers
        servers = get_available_servers()
        
        result = {}
        for name, info in servers.items():
            result[name] = {
                "name": name,
                "type": info.get("type", "stdio"),
                "description": info.get("description", ""),
                "enabled": info.get("enabled", False),
                "config": info.get("config", {}),
                "path": info.get("path", ""),
                "server_file": info.get("server_file"),
                "running": mcp_manager and name in mcp_manager.connections if mcp_manager else False,
                "tools_count": len([t for t in mcp_manager.tools.values() if t.server_name == name]) if mcp_manager else 0
            }
        
        return jsonify({"servers": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/v1/mcp/servers/<name>/enable', methods=['POST'])
def mcp_enable_server(name: str):
    """启用 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    try:
        from mcp_servers import enable_server
        success = enable_server(name)
        
        if success:
            # 重新加载配置
            if mcp_manager:
                mcp_manager.reload_config()
            
            return jsonify({
                "success": True,
                "message": f"服务器 {name} 已启用"
            })
        else:
            return jsonify({"error": f"服务器 {name} 不存在"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/v1/mcp/servers/<name>/disable', methods=['POST'])
def mcp_disable_server(name: str):
    """禁用 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    try:
        from mcp_servers import disable_server
        success = disable_server(name)
        
        if success:
            # 停止运行中的服务器
            if mcp_manager and name in mcp_manager.connections:
                mcp_manager.stop_server(name)
            
            return jsonify({
                "success": True,
                "message": f"服务器 {name} 已禁用"
            })
        else:
            return jsonify({"error": f"禁用服务器 {name} 失败"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/v1/mcp/servers/<name>/details', methods=['GET'])
def mcp_server_details(name: str):
    """获取 MCP 服务器详情"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    try:
        from mcp_servers import get_available_servers
        servers = get_available_servers()
        
        if name not in servers:
            return jsonify({"error": f"服务器 {name} 不存在"}), 404
        
        info = servers[name]
        config = info.get("config", {})
        
        # 获取工具列表
        tools = []
        if mcp_manager and name in mcp_manager.connections:
            for tool in mcp_manager.tools.values():
                if tool.server_name == name:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema
                    })
        
        result = {
            "name": name,
            "type": info.get("type", "stdio"),
            "description": info.get("description", ""),
            "enabled": info.get("enabled", False),
            "running": mcp_manager and name in mcp_manager.connections if mcp_manager else False,
            "path": info.get("path", ""),
            "server_file": info.get("server_file"),
            "config": config,
            "tools": tools
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 静态文件服务 ====================

@app.route('/')
def index():
    """返回首页"""
    return app.send_static_file('index.html')


@app.route('/admin')
def admin():
    """MCP 管理界面"""
    return app.send_static_file('mcp_admin.html')


@app.route('/tools')
def tools_page():
    """MCP 工具列表页面"""
    return app.send_static_file('tools.html')


@app.route('/status')
def status_page():
    """健康状态页面"""
    return app.send_static_file('health.html')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSeek OpenAI 兼容代理服务器')
    parser.add_argument('--config', type=str, default='config.jsonc', help='配置文件路径')
    parser.add_argument('--host', type=str, help='监听地址（覆盖配置文件）')
    parser.add_argument('--port', type=int, help='监听端口（覆盖配置文件）')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-mcp', action='store_true', help='禁用 MCP 功能')
    
    args = parser.parse_args()
    
    # 加载配置文件
    CONFIG = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    host = args.host or CONFIG.get('host', '127.0.0.1')
    port = args.port or CONFIG.get('port', 8002)
    debug = args.debug or CONFIG.get('debug', False)
    mcp_enabled = CONFIG.get('mcp_enabled', True) and not args.no_mcp
    
    print("=" * 60)
    print("DeepSeek OpenAI 兼容代理服务器")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"监听地址: http://{host}:{port}")
    print(f"API 端点: http://{host}:{port}/v1/chat/completions")
    print(f"后端 API: {CONFIG.get('chat_completions_url', 'N/A')}")
    
    # 显示访问控制状态
    access_keys = CONFIG.get('access_keys', [])
    if access_keys:
        print(f"访问控制: 已启用 ({len(access_keys)} 个密钥)")
    else:
        print("访问控制: 已禁用（开放访问）")
    
    if CONFIG.get('api_key'):
        print("转发 Key: 已配置")
    else:
        print("转发 Key: 未配置（使用用户 Key）")
    
    # 初始化 MCP（从 mcp_servers 目录自动发现服务）
    if MCP_AVAILABLE and mcp_enabled:
        try:
            mcp_manager = get_mcp_manager()
            print(f"MCP 配置: mcp_servers/ 目录")
            print(f"MCP 服务器: {len(mcp_manager.servers)} 个配置, {len(mcp_manager.connections)} 个运行中")
            print(f"MCP 工具: {len(mcp_manager.tools)} 个可用")
        except Exception as e:
            print(f"MCP 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            mcp_manager = None
    else:
        print("MCP: 已禁用")
    
    print("=" * 60)
    
    app.run(host=host, port=port, debug=debug)
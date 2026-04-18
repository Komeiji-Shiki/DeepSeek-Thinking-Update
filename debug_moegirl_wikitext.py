"""
调试萌娘百科 wikitext 格式
查看原始内容中"命名"标题的实际格式
"""

import sys
import time
from mcp_servers.mcp_client import MCPManager

def main():
    print("=" * 70)
    print("  萌娘百科 Wikitext 格式调试")
    print("=" * 70)
    
    # 创建 MCP 管理器
    print("\n📡 正在连接 MCP 服务器...")
    manager = MCPManager()
    time.sleep(2)
    
    # 获取原始 wikitext
    print("\n📖 获取初音未来页面的原始内容（不清理）...")
    
    # 找到 get_page 工具
    tools = manager.get_openai_tools()
    get_page_tool = None
    for tool in tools:
        if 'moegirl_get_page' in tool['function']['name'] and 'section' not in tool['function']['name']:
            get_page_tool = tool['function']['name']
            break
    
    if not get_page_tool:
        print("❌ 未找到页面获取工具")
        return
    
    # 获取原始内容（不清理）
    print(f"🔧 调用: {get_page_tool}")
    result = manager.call_tool(get_page_tool, {
        "title": "初音未来",
        "clean_content": False,  # 不清理，保留原始格式
        "max_length": 5000
    })
    
    if not result:
        print("❌ 获取失败")
        return
    
    print("\n" + "=" * 70)
    print("  原始 Wikitext 内容分析")
    print("=" * 70)
    
    # 分析前100行
    lines = result.split('\n')
    
    print(f"\n总行数: {len(lines)}")
    print("\n前150行内容：")
    print("-" * 70)
    
    for i, line in enumerate(lines[:150], 1):
        # 高亮显示可能是标题的行
        if '命名' in line or '音源' in line or '声库' in line:
            print(f"{i:4d} >>> {repr(line)}")  # 使用 repr 显示原始字符
        elif line.strip().startswith('=='):
            print(f"{i:4d} ||| {repr(line)}")
        elif i <= 50:  # 前50行全部显示
            print(f"{i:4d} | {line[:100]}")
    
    print("-" * 70)
    
    # 查找所有包含 = 的行
    print("\n" + "=" * 70)
    print("  所有包含等号(=)的行（可能是标题）")
    print("=" * 70)
    
    for i, line in enumerate(lines[:200], 1):
        stripped = line.strip()
        if stripped.startswith('=') and '命名' in line:
            print(f"行 {i:4d}: {repr(line)}")
            # 显示前后几行以了解上下文
            if i > 1:
                print(f"  前行: {repr(lines[i-2])}")
            if i < len(lines):
                print(f"  后行: {repr(lines[i])}")
    
    # 查找"命名"关键词
    print("\n" + "=" * 70)
    print("  查找 '命名' 关键词的所有出现位置")
    print("=" * 70)
    
    for i, line in enumerate(lines[:300], 1):
        if '命名' in line:
            print(f"行 {i:4d}: {repr(line)}")
    
    # 关闭连接
    print("\n🔌 关闭 MCP 连接...")
    manager.stop_all_servers()
    
    print("\n✅ 调试完成!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  调试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
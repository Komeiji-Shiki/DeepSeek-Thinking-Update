"""
调试失败的标题
"""

import sys
import time
from mcp_servers.mcp_client import MCPManager

manager = MCPManager()
time.sleep(3)

# 找 get_page 工具
tools = manager.get_openai_tools()
get_page_tool = None
for tool in tools:
    if 'moegirl_get_page' in tool['function']['name'] and 'section' not in tool['function']['name']:
        get_page_tool = tool['function']['name']
        break

# 调试这些页面
test_pages = [
    ("镜音铃·连", "命名"),
    ("KAITO", "基本资料"),
    ("MEIKO", "形象")
]

for page_title, section_name in test_pages:
    print("\n" + "=" * 70)
    print(f"调试: {page_title} -> {section_name}")
    print("=" * 70)
    
    result = manager.call_tool(get_page_tool, {
        "title": page_title,
        "clean_content": False,
        "max_length": 5000
    })
    
    if result:
        lines = result.split('\n')
        print(f"\n查找包含 '{section_name}' 的行:")
        print("-" * 70)
        
        found_count = 0
        for i, line in enumerate(lines[:200], 1):
            if section_name in line:
                print(f"行 {i:3d}: {repr(line[:100])}")
                # 显示前后各一行
                if i > 1 and i-2 < len(lines):
                    print(f"    前: {repr(lines[i-2][:80])}")
                if i < len(lines):
                    print(f"    后: {repr(lines[i][:80])}")
                print()
                found_count += 1
                if found_count >= 3:  # 最多显示3处
                    break
        
        if found_count == 0:
            print(f"❌ 未找到包含 '{section_name}' 的行")
            print("\n前50行内容:")
            for i, line in enumerate(lines[:50], 1):
                if line.strip().startswith('='):
                    print(f"{i:3d}: {repr(line)}")
    else:
        print("❌ 获取页面失败")
    
    time.sleep(0.5)

manager.stop_all_servers()
print("\n✅ 调试完成")
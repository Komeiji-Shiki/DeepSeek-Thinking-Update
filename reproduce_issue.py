import subprocess
import json
import sys
import os
import time

def run_mcp_server():
    # Path to the MCP server script
    server_script = os.path.join("mcp_servers", "google-search-mcp-master", "dist", "index.js")
    
    # Command to run the server
    cmd = ["node", server_script]
    
    # Environment variables
    env = os.environ.copy()
    env["HTTPS_PROXY"] = "http://127.0.0.1:7897"
    env["ALL_PROXY"] = "http://127.0.0.1:7897"
    
    print(f"Starting server: {' '.join(cmd)}")
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )
    
    return process

def send_request(process, request):
    json_req = json.dumps(request)
    print(f"Sending: {json_req}")
    process.stdin.write(json_req + "\n")
    process.stdin.flush()

def read_response(process):
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(f"Received: {line.strip()}")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

def main():
    process = run_mcp_server()
    
    try:
        # 1. Initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        send_request(process, init_req)
        read_response(process)
        
        # 2. Initialized notification
        notify_req = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        send_request(process, notify_req)
        
        # 3. List tools (optional, just to check)
        list_tools_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        send_request(process, list_tools_req)
        read_response(process)
        
        # 4. Call search tool
        search_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {
                    "query": "site:wikipedia.org artificial intelligence",
                    "limit": 3,
                    "timeout": 300000  # Increase timeout to 5 minutes
                }
            }
        }
        send_request(process, search_req)
        
        # Wait for response (might take a while)
        response = read_response(process)
        
        # Check for errors
        if "error" in response:
            print("Error received:", response["error"])
        elif "result" in response:
            content = response["result"]["content"][0]["text"]
            try:
                result_data = json.loads(content)
                print("Search Results:", json.dumps(result_data, indent=2, ensure_ascii=False))
            except:
                print("Raw Result:", content)
                
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        process.terminate()
        stderr_output = process.stderr.read()
        if stderr_output:
            print("STDERR:", stderr_output)

if __name__ == "__main__":
    main()
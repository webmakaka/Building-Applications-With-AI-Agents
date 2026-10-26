import requests
import json
from datetime import datetime

def query_all_logs():
    """查询所有日志"""
    end = int(datetime.now().timestamp())
    start = end - 360000  # 最近1小时
    
    url = "http://localhost:3100/loki/api/v1/query_range"
    params = {
        "query": '{app="tool.manage_inventory"}',
        "start": start,
        "end": end,
        "limit": 50,
        "direction": "BACKWARD"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("status") == "success":
        results = data.get("data", {}).get("result", [])
        for result in results:
            stream = result.get("stream", {})
            values = result.get("values", [])
            print(f"\n标签: {stream}")
            for timestamp_ns, message in values:
                timestamp_sec = int(timestamp_ns) / 1e9
                dt = datetime.fromtimestamp(timestamp_sec)
                print(f"  [{dt.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

query_all_logs()

import http.client
import json
from langchain.tools import Tool
# 引入能够读取yaml配置文件的工具
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# 定义自定义函数来查询 Serper.dev API
def search_serper(query: str) -> str:
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': config["api_keys"]["serpapi_key"],  # 替换为您的 API Key
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

# 使用 langchain 的 Tool 包装自定义函数
serper_tool = Tool(
    name="SerperSearch",
    func=search_serper,
    description="useful for when you need to answer questions about current events or the current state of the world"
)

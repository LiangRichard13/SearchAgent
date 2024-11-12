import json
import yaml
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 读取配置文件
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = config["api_keys"]["openai_key"]

# 定义 LLM
llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4o",
)

def memory_save(history):
    save_template = {"summary": "", "date": ""}
    # 获取当前本地日期和时间
    current_datetime = datetime.now()
    # 转换为可读字符串
    readable_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # 保存日期
    save_template["date"] = readable_datetime
    
    memory_save_prompt = f"""
    This conversation history:{history}\n\n
    The above is the history of this dialogue between user and AI assistant.
    Today's date:{readable_datetime},
    Please extract the key events and record the events and dates(if necessary) about user,
    In addition, please add the priority for event recording, 
    divided into "important", "normal" and "unimportant".
    Please output strictly according to the following format：
    {"summary":"Your extraction and recording","importance":"important/normal/unimportant"}
    """
    summary = llm.invoke(memory_save_prompt).content  # 使用 invoke 调用 LLM

    # 保存事件总结
    save_template["summary"] = summary
    try:
        if not os.path.exists('memory/memory.json'):
            # 创建初始的 JSON 结构
            memory_save = {"user portrait": "", "conversation memory": []}
        else:
            with open('memory/memory.json', 'r', encoding='utf-8') as file:
                memory_save = json.load(file)  # 加载 JSON 数据
        
        user_portrait = memory_save.get("user portrait", "")
        
        user_portrait_update_prompt = f"""
        This is the current user portrait:{user_portrait}\n\n
        This is a summary of the conversation just now:{summary}\n\n
        Please update the user portrait. 
        It is required that the portrait can reflect the user's hobbies, personality and other personal characteristics. 
        """
        user_update_portrait = llm.invoke(user_portrait_update_prompt).content  # 使用 invoke 调用 LLM
        
        memory_save["conversation memory"].append(save_template)
        memory_save["user portrait"] = user_update_portrait
        
        with open('memory/memory.json', 'w', encoding='utf-8') as file:
            json.dump(memory_save, file, ensure_ascii=False, indent=4)
        print("记忆已更新")
    except Exception as error:
        print(f"在记忆过程中出现了错误！Error: {str(error)}")

def memory_get(query) -> str:
    if os.path.exists('memory/memory.json'):
        with open('memory/memory.json', 'r', encoding='utf-8') as file:
            memory_save = json.load(file)
        return str(memory_save)
    return "Memory file not found."

long_term_memory_tool = Tool(
    name="long-term memory recall",
    func=memory_get,
    description="Tool for obtaining user portraits that describe user characteristics and a summary of important information and events about users."
)

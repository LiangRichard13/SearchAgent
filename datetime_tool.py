from datetime import datetime
from langchain.tools import Tool

def get_now_time(query:str)->str:
    # 获取当前本地日期和时间
    current_datetime = datetime.now()

    # 转换为可读字符串
    readable_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return f"Now the date and time are：{readable_datetime}"

datetime_tool = Tool(
    name="GetNowTime",
    func=get_now_time,
    description="Used to get the current date and time"
)

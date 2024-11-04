import logging
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents import AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
# 引入记忆组件
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

# 引入能够读取yaml配置文件的工具
import yaml
import os
# 引入自定义的搜索工具
from serper_tool import serper_tool
from rag.rag_tool import rag_tool
from datetime_tool import datetime_tool

# 引入日志记录
logging.basicConfig(
    filename='agent_output.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    encoding='utf-8'  # 指定日志文件的编码为 UTF-8
)

logger = logging.getLogger()

# 读取配置文件
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# 设置OPENIA_API_KEY环境变量
os.environ["OPENAI_API_KEY"] = config["api_keys"]["openai_key"]

# 定义 LLM
llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4o",
)

# 加载工具llm-math
tools = load_tools(
    ["llm-math"], 
    llm=llm,
    )

# 将自定义工具添加到工具列表中
tools.append(serper_tool)
tools.append(rag_tool)
tools.append(datetime_tool)
tools_information =str([{"name": tool.name, "description": tool.description} for tool in tools])


# 记忆组件
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 初始化gent
complex_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
        agent_kwargs={
        "extra_prompt_messages":[MessagesPlaceholder(variable_name="chat_history"),MessagesPlaceholder(variable_name="agent_scratchpad")],
    }
)

# 加载规划器和执行器
# planner = load_chat_planner(llm)
# executor = load_agent_executor(llm, tools, verbose=True)

# 创建Plan and Execute代理
# complex_agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

def determine_task_type(query):
    # 提供初步提示让大模型判断
    judgment_prompt = f"""
    Here is the user's request: "{query}" Please analyze whether this task requires a multi-step plan and execution.
    If it is a simple task that does not require any external tools, please return "simple." 
    If it is a complex task that requires the use of external tools, please return "complex." 
    The external tools available are:{tools_information}
    """
    result = llm.invoke(judgment_prompt).content
    return "complex" if "complex" in result else "simple"

def task_implement(query):
    judge=determine_task_type(query)
    if "complex" in judge:
        try:
            prompt=plan_before_implement(query)
            output=complex_agent(prompt)
            logger.info(f"Query: {query}")
            logger.info(f"Plan:{prompt}")
            logger.info(f"Output: {output}")
            print(output)
        except IndexError as e:
            print(str(e))
    else:
        output=llm.invoke(query).content
        memory.save_context(query,output)
        logger.info(f"Query: {query}")
        logger.info(f"Output: {output}")

def plan_before_implement(query):
    template=f"""
    Here is the user's request: "{query}",
    These are the tools you can use:{tools_information}
    Please make your best effort to plan the steps in detail to ensure the rigor of the output. 
    The output format is as follows: 
    Step 1: xxx, 
    Step 2: xxx, 
    ... 
    Step n: obtain the final answer.
    """
    result = llm.invoke(template).content
    prompt=f"""
    Here is the user's request: "{query}",
    These are the tools you can use:{tools_information},
    Here are the steps for your reference:{result}
    """
    return prompt


# while True:
#     # query = input("query:")

query="请问在2024巴黎奥运会是谁取得了中国队的首枚金牌？他的年龄在今天是多少岁？他年龄的平方又是多少？"
task_implement(query)  



from llama_index.core import load_index_from_storage,StorageContext,Settings
from llama_index.embeddings.openai import OpenAIEmbedding
# 配置查询工具
from llama_index.core.tools import QueryEngineTool
# from llama_index.core.tools import ToolMetadata
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from langchain.tools import Tool
import os
import yaml
with open("D:\Code Projects\AgentLearning\config.yaml", "r") as file:
    config = yaml.safe_load(file)

os.environ["OPENAI_API_KEY"] = config["api_keys"]["openai_key"]

embed_model = OpenAIEmbedding(
#指定了一个预训练的sentence-transformer模型的路径
    model="text-embedding-ada-002",  # 设置使用的嵌入模型
    api_key=config["api_keys"]["openai_key"]  # 替换为你的OpenAI API密钥
)
#将创建的嵌入模型赋值给全局设置的embed_model属性，
#这样在后续的索引构建过程中就会使用这个模型。
Settings.embed_model = embed_model

rag_index_path=config["path"]["rag_index_path"]
storage_context_vector = StorageContext.from_defaults(
        persist_dir=rag_index_path)
index_vector=load_index_from_storage(storage_context=storage_context_vector)

# 创建查询引擎
vector_query_engine = index_vector.as_query_engine()

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Tool for Accessing Past Political News Information"
    ),
)

from llama_index.core.selectors import LLMSingleSelector


query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(), #將LLMSingleSelector設定為selector
    query_engine_tools=[ #將前述的兩個tool設定為待選擇的query engine
        vector_tool,
    ],
    verbose=True
)

def news_rag(query: str) -> str:
    response=query_engine.query(query)
    print(str(response))


rag_tool = Tool(
    name="NewsRag",
    func=news_rag,
    description="Tool for Accessing News Information"
)
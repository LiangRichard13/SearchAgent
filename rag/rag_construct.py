from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings,SummaryIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import yaml

# 读取配置文件
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

#初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
embed_model = OpenAIEmbedding(
#指定了一个预训练的sentence-transformer模型的路径
    model="text-embedding-ada-002",  # 设置使用的嵌入模型
    api_key=config["api_keys"]["openai_key"]  # 替换为你的OpenAI API密钥
)
#将创建的嵌入模型赋值给全局设置的embed_model属性，
#这样在后续的索引构建过程中就会使用这个模型。
Settings.embed_model = embed_model

#从指定目录读取所有文档，并加载数据到内存中
documents = SimpleDirectoryReader("./rag_documents").load_data()
print("The length of documents_list:",len(documents))

#进行切分
# Documents->nodes
splitter = SentenceSplitter(chunk_size=1024)
# 根據上方splitter的設定把documents切割成nodes
nodes = splitter.get_nodes_from_documents(documents)


#创建一个VectorStoreIndex，并使用之前加载的文档来构建索引。
# 此索引将文档转换为向量，并存储这些向量以便于快速检索。
index_vector = VectorStoreIndex(nodes)
# index_summary=SummaryIndex(nodes)

index_vector.storage_context.persist(persist_dir="./rag_index_storage_vector")
# index_summary._storage_context.persist(persist_dir="./rag_index_storage_summary")

# 创建一个查询引擎，这个引擎可以接收查询并返回相关文档的响应。
# query_engine = index.as_query_engine()

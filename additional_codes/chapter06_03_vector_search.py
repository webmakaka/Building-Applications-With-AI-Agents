import os
from dotenv import load_dotenv
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()

import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore # 替代 vectordb
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END, MessagesState

# --- 1. 设置 LLM 和 向量库 ---
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings() # 需要 Embedding 模型来向量化文本

# 使用 LangChain 原生的内存向量库，替代 vectordb.Memory
vector_store = InMemoryVectorStore(embeddings)

# --- 2. 准备数据并存入向量库 ---
# 这里的逻辑等同于原来的 memory.save()
text1 = """
Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns and make decisions with minimal human intervention.
"""
doc1 = Document(
    page_content=text1,
    metadata={"title": "Introduction to Machine Learning", "url": "https://example.com/ml"}
)

text2 = """
Artificial intelligence (AI) is the simulation of human intelligence in machines
that are programmed to think like humans and mimic their actions.
The term may also be applied to any machine that exhibits traits associated with
a human mind such as learning and problem-solving.
"""
doc2 = Document(
    page_content=text2,
    metadata={"title": "Introduction to AI", "url": "https://example.com/ai"}
)

# 存入数据
vector_store.add_documents([doc1, doc2])

# --- 3. 定义工具/检索逻辑 ---
query = "What is the relationship between AI and machine learning?"
# 搜索 (Simulate search)
results = vector_store.similarity_search(query, k=3)

print("--- 初始检索测试 ---")
for doc in results:
    print(f"Found: {doc.metadata['title']}")

# --- 4. 定义 Graph 节点 ---

def call_model(state: MessagesState):
    # 这里演示如何把检索结果结合进 prompt (RAG)
    # 实际应用中，通常会先根据用户问题去 vector_store 检索，然后再 invoke LLM
    
    # 简单示例：直接回答用户，不强制在这个节点做检索，保持原逻辑结构
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# --- 5. 构建 Graph ---
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)
graph = builder.compile()

# --- 6. 运行 ---
input_message = HumanMessage(content="hi! I'm bob")
for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


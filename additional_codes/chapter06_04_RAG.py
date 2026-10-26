import os
from dotenv import load_dotenv
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()

import os
from typing import Annotated, List
from typing_extensions import TypedDict

# LangChain 核心组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# LangGraph 0.3.x 组件
from langgraph.graph import StateGraph, START, END, MessagesState

# ==========================================
# 1. 初始化环境与向量库 (Knowledge Base)
# ==========================================

# 确保环境变量设置了 OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = "sk-..."

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore(embeddings)

# 模拟存入数据 (Indexing)
docs = [
    Document(
        page_content="Machine learning is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
        metadata={"source": "ML_Intro"}
    ),
    Document(
        page_content="Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
        metadata={"source": "AI_Intro"}
    ),
    Document(
        page_content="Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers.",
        metadata={"source": "DL_Intro"}
    )
]
vector_store.add_documents(docs)
print("✅ 知识库初始化完成，文档已存入内存向量库。\n")

# ==========================================
# 2. 定义 State (状态管理)
# ==========================================

# 我们扩展 MessagesState，增加一个 'context' 字段来存储检索到的文本
class State(MessagesState):
    context: str

# ==========================================
# 3. 定义节点 (Nodes)
# ==========================================

def retrieve(state: State):
    """
    检索节点：获取用户最后一个问题，去向量库搜索相关内容。
    """
    # 1. 获取用户最新的消息
    latest_message = state["messages"][-1]
    query = latest_message.content
    
    # 2. 执行向量搜索 (Top 2)
    search_results = vector_store.similarity_search(query, k=2)
    
    # 3. 格式化检索结果为字符串
    context_text = "\n\n".join([doc.page_content for doc in search_results])
    
    print(f"--- 检索到的上下文 (Top 2) ---\n{context_text}\n---------------------------")
    
    # 4. 更新 State 中的 context
    return {"context": context_text}

def generate(state: State):
    """
    生成节点：利用检索到的 context 和历史消息，让 LLM 生成回答。
    """
    context = state["context"]
    messages = state["messages"]
    
    # 构造 System Prompt，强制 LLM 基于 Context 回答
    system_prompt = (
        "You are a helpful assistant. "
        "Use the following pieces of context to answer the user's question. "
        "If you don't know the answer, just say that you don't know. \n\n"
        f"Context:\n{context}"
    )
    
    # 将 System Prompt 放在消息列表的最前面
    # 注意：LangGraph 的 MessagesState 会自动处理消息追加，但这里我们是临时构建本次调用的 prompt
    prompt_messages = [SystemMessage(content=system_prompt)] + messages
    
    # 调用模型
    response = llm.invoke(prompt_messages)
    
    # 返回生成的 AI 消息，这会被追加到 state["messages"] 中
    return {"messages": [response]}

# ==========================================
# 4. 构建 Graph (Workflow)
# ==========================================

builder = StateGraph(State)

# 添加节点
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

# 定义边 (流程控制)
# START -> 检索 -> 生成 -> END
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

# 编译图
graph = builder.compile()

# ==========================================
# 5. 运行测试
# ==========================================

# 测试问题
user_query = "What is the relationship between AI and Machine Learning?"
input_message = HumanMessage(content=user_query)

print(f"🤖 User: {user_query}")

# 运行图
# stream_mode="values" 会打印出每一步的状态更新
for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
    # 这里我们只关心最后生成的 AI 回答，通常它是消息列表的最后一个
    last_msg = chunk["messages"][-1]
    if isinstance(last_msg, AIMessage):
        print(f"\n💡 AI Answer (RAG):")
        print(last_msg.content)

# 你可以打印图结构图（如果安装了 mermaid 渲染支持）
# print(graph.get_graph().draw_mermaid())

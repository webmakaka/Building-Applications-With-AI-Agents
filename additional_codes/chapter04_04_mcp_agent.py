# ! pip install "mcp[cli]" langchain-mcp-adapters
import time
import os, threading, asyncio
from queue import Queue
from concurrent.futures import Future
from langchain_mcp_adapters. client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint. memory import MemorySaver
from langchain_openai import ChatOpenAI

# 演示性质用OpenAI，需要访问OpenAI。也可以用DeepSeek等国产大模型。
os.environ["OPENAI_API_KEY"] = 'Your Key'
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 全局请求队列与会话上下文存储
request_queue: Queue[tuple[str, str, Future]] = Queue()
sessions: dict[str, dict] = {}

# LLM用全局实例变量，便于切换不同的LLM
# 官方对 OpenAI 客户端明确保证"所有实例方法本身线程安全"
llm = ChatOpenAI(model="gpt-4o", temperature=0)
system_prompt = "你是一个有用的助手，能够根据需要自动决定调用哪个工具回答问题。"

# 连接多个MCP Server的配置
mcp_config = {
    "math": {
        "command": "python",
        "args": ["/home/ubuntu/python/mcp/math_server.py"],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8300/sse",
        "transport": "sse",
    }
}

async def _dispatcher_loop():
    """
    后台单循环调度器：取出请求，根据 session_id 初始化或复用 Agent 上下文，执行 ainvoke 并通过 Future 回写结果。
    """
    global sessions
    # 创建 MCP 客户端
    client = MultiServerMCPClient(mcp_config)
    # 获取工具列表
    tools = await client.get_tools()
    print(f"[INFO] 获取到 {len(tools)} 个工具: {[tool. name for tool in tools]}")

    while True:
        session_id, question, fut = await asyncio.to_thread(request_queue.get)
        # 主线程向队列中压入一个(None,None,None)来结束事件循环退出
        if session_id is None:
            break
        # 若无该会话，初始化上下文
        if session_id not in sessions:
            # 每个会话创建自己的Agent及MemorySaver实例
            memory = MemorySaver()
            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=system_prompt,
                checkpointer=memory,
                debug=False
            )
            sessions[session_id] = {"agent": agent, "memory": memory}
            print("创建Session ", session_id, "。")
        ctx = sessions[session_id]
        # 异步调用 Agent
        try:
            resp = await ctx["agent"].ainvoke(
                {"messages": question},
                config={"configurable": {"thread_id": session_id}}
            )
            fut.set_result(resp)
        except Exception as e:
            fut.set_exception(e)
    # 后台线程结束，退出后台事件循环
    asyncio.get_running_loop().stop()

def start_dispatcher():
    """启动后台事件循环线程，仅调用一次"""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=lambda: loop.run_forever(), daemon=True)
    t.start()
    asyncio.run_coroutine_threadsafe(_dispatcher_loop(), loop)
    print("后台事件循环线程已启动。")

def ask_agent(session_id: str, question: str) -> str:
    """主线程同步接口"""
    fut = Future()
    request_queue.put((session_id, question, fut))
    return fut.result()["messages"]

def stop_dispatcher():
    """优雅关闭"""
    request_queue.put((None, None, None))
    print("后台事件循环线程已退出。")
    
# 清除对话历史最简便的方法是销毁实例并在新对话中建立新的实例
def clear_session(session_id: str):
    """清除用户对话"""
    global sessions
    if session_id in sessions:
        sessions. pop(session_id)
        print("Session ", session_id, "已清空。")
    
# ───────主线程交互示例 & 优雅退出 ────────
if __name__ == "__main__":
    # 启动后台事件循环线程
    t3 = time.time()
    start_dispatcher()
    
    # 用户161多轮连续对话
    session_id = "161"
    query = "what's (3 + 5) x 12?"
    ans = ask_agent(session_id, query)
    print(query)
    print(ans[-1].content)
    query = "what's the result 96 divided by 6?"
    ans = ask_agent(session_id, query)
    print(query)
    print(ans[-1].content)
    query = "what is the weather in nyc?"
    ans = ask_agent(session_id, query)
    print(query)
    print(ans[-1].content)
    t4 = time.time()
    print("t4-t3: ",t4-t3)
    # 测试清除用户session
    clear_session("161")
    # 退出后台事件循环，关闭Agent
    stop_dispatcher()

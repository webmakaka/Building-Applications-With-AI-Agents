import os
os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END


# -- 1) Define our single business tool
@tool
def cancel_order(order_id: str):
    """Cancel an order that hasn't shipped."""
    # (Here you'd call your real backend API)
    return f"Order {order_id} has been cancelled."


# -- 2) The agent "brain": invoke LLM, run tool, then invoke LLM again
def call_model(state):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "UNKNOWN"})

    # System prompt tells the model exactly what to do
    prompt = f"""You are an ecommerce support agent.
    ORDER ID: {order['order_id']}
    If the customer asks to cancel, call cancel_order(order_id)
    and then send a confirmation.
    Otherwise respond normally."""

    full_msgs = [SystemMessage(content=prompt)] + msgs
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 1st LLM pass: decides whether to call our tool
    first = llm(full_msgs)
    outputs = [first]

    # Old tool call format
    if hasattr(first, "tool_calls") and first.tool_calls:
        # run the cancel_order tool
        tc = first.tool_calls[0]
        tool_output = cancel_order(**tc["args"])

        outputs.append(
            ToolMessage(
                content=tool_output,
                tool_call_id=tc["id"]
            )
        )

        # 2nd LLM pass: generate the final confirmation text
        second = llm(full_msgs + outputs)
        outputs.append(second)

    return {"messages": outputs}


# -- 3) Wire it all up in a StateGraph
def construct_graph():
    schema = {"order": None, "messages": []}
    g = StateGraph(schema)

    g.add_node("assistant", call_model)

    # required in old versions
    g.add_edge("assistant", END)

    g.set_entry_point("assistant")
    return g.compile()


if __name__ == "__main__":
    graph = construct_graph()

    example_order = {"order_id": "A12345"}
    convo = [HumanMessage(content="Please cancel my order A12345.")]

    result = graph.invoke({"order": example_order, "messages": convo})

    for m in result["messages"]:
        print(m.type, ":", m.content)

# -------------------------------------------------------------------------------------------------
# 注：这是为从国内访问OpenAI增加的VPN代理设置，以及设置OpenAI key
import os
os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
# 设置VPN URL以访问OpenAI
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END


# -- 1) 定义我们的第一个业务工具
@tool
def cancel_order(order_id: str):
    """取消没有发货的订单"""
    # (这里你可以调用真正的后端业务API)
    return f"订单 {order_id} 已被撤销。"


# -- 2) 智能体的大脑: 调用LLM, 运行工具, 然后再次调用LLM
def call_model(state):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "UNKNOWN"})

    # 系统提示词，告诉大模型怎么干
    prompt = f"""你是个电子商务支持智能体。
    ORDER ID: {order['order_id']}
    如果客户要求取消订单，调用 cancel_order(order_id)
    然后发送一条确认信息。
    否则照常回复。"""

    full_msgs = [SystemMessage(content=prompt)] + msgs
    llm = ChatOpenAI(model="gpt-5", temperature=1)
    
    # 第1次LLM调用: 决定是否调用工具
    first = llm(full_msgs)
    outputs = [first]

    # 旧版LangChain调用格式
    if hasattr(first, "tool_calls") and first.tool_calls:
        # 运行 cancel_order 工具
        tc = first.tool_calls[0]
        tool_output = cancel_order(**tc["args"])

        outputs.append(
            ToolMessage(
                content=tool_output,
                tool_call_id=tc["id"]
            )
        )

        # 第2次LLM调用: 生成最终的确认信息
        second = llm(full_msgs + outputs)
        outputs.append(second)

    return {"messages": outputs}


# -- 3) 通过一个LangGraph StateGraph组装起所有部件
def construct_graph():
    schema = {"order": None, "messages": []}
    g = StateGraph(schema)

    g.add_node("assistant", call_model)

    # LangGraph需要一条指向END的边
    g.add_edge("assistant", END)

    g.set_entry_point("assistant")
    return g.compile()


if __name__ == "__main__":
    graph = construct_graph()

    example_order = {"order_id": "A12345"}
    convo = [HumanMessage(content="请取消我的订单A12345。")]

    result = graph.invoke({"order": example_order, "messages": convo})

    for m in result["messages"]:
        print(m.type, ":", m.content)

    # 最小评估检查
    example_order = {"order_id": "B73973"}
    convo = [HumanMessage(content='''请撤销订单 #B73973.
        我在其它地方找到了更便宜的选择。''')]
    result = graph.invoke({"order": example_order, "messages": convo})

    print(result)
    assert any("cancel_order" in str(m.content) for m in result["messages"]),\
        "没有调用撤销订单工具"
    assert any("撤销" in m.content.lower() for m in result["messages"]),\
        "缺少确认撤销消息"

    print("✅ 智能体通过了最小评估检查。")

# -----------------------------------------------------------------------------------------------
import os
os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"

from typing import TypedDict, List, Optional
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.messages import ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END
import json

# -- 定义状态 schema
class AgentState(TypedDict):
    order: Optional[dict]
    messages: List[BaseMessage]


# -- 1) 定义业务工具
@tool
def cancel_order(order_id: str):
    """取消没有发货的订单"""
    return f"订单 {order_id} 已被撤销。"


# -- 2) 智能体大脑函数
def call_model(state: AgentState) -> dict:
    msgs = state["messages"]
    order = state. get("order", {"order_id": "UNKNOWN"})

    prompt = f"""你是个电子商务支持智能体。
            ORDER ID: {order['order_id']}
            如果客户要求取消订单，调用 cancel_order(order_id)
            然后发送一条确认信息。
            否则照常回复。"""

    # 构建完整消息列表（包括系统提示）
    full_msgs = [SystemMessage(content=prompt)] + msgs
    
    # 将工具绑定到 LLM
    llm = ChatOpenAI(model="gpt-5", temperature=1)
    llm_with_tools = llm.bind_tools([cancel_order])
    
    # 第1次 LLM 调用 - 决定是否调用工具
    first = llm_with_tools.invoke(full_msgs)
    outputs = [first]

    # 检查是否有工具调用
    if hasattr(first, "tool_calls") and first.tool_calls:
        tc = first.tool_calls[0]
        # 运行工具
        tool_output = cancel_order. invoke({"order_id": tc["args"]["order_id"]})

        # 添加工具消息
        outputs.append(
            ToolMessage(
                content=tool_output,
                tool_call_id=tc["id"]
            )
        )

        # 第2次 LLM 调用 - 根据工具结果生成回应
        second = llm_with_tools.invoke(full_msgs + outputs)
        outputs. append(second)

    # ✅ 只返回 outputs，不包括 SystemMessage 和原始 HumanMessage
    return {"messages": outputs}


# -- 3) 构建图
def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_model)
    g.add_edge("assistant", END)
    g.set_entry_point("assistant")
    return g. compile()


if __name__ == "__main__":
    graph = construct_graph()

    example_order = {"order_id": "A12345"}
    convo = [HumanMessage(content="请取消我的订单A12345。")]

    result = graph.invoke({"order": example_order, "messages": convo})

    print("=== 最终消息序列 ===")
    for m in result["messages"]:
        print(f"{m.type}: {m.content}\n")

    # 最小评估检查
    graph = construct_graph()
    example_order = {"order_id": "B73973"}
    convo = [HumanMessage(content='''请撤销订单 #B73973.
        我在其它地方找到了更便宜的选择。''')]
    result = graph.invoke({"order": example_order, "messages": convo})

    print(result)
    assert any("cancel_order" in str(json.dumps(m.tool_calls)) for m in result["messages"]),\
        "没有调用撤销订单工具"
    assert any("撤销" in m.content.lower() for m in result["messages"]),\
        "缺少确认撤销消息"

    print("✅ 智能体通过了最小评估检查。")

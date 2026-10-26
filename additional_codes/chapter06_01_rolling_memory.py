import os
from dotenv import load_dotenv
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()

from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ---- 定义 state ----
class State(TypedDict):
    messages: List[BaseMessage]


# ---- LLM ----
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---- 节点函数：根据 state["messages"] 调用模型 ----
def call_model(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


# ---- 定义 Graph ----
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# ---------- 不带 memory 的图 ----------
graph = builder.compile()

print("\n=== No Memory ===")

input_msg = HumanMessage(content="hi! I'm bob")
for event in graph.stream({"messages": [input_msg]}):
    for node, val in event.items():
        if "messages" in val:
            print(val["messages"][-1].content)

input_msg = HumanMessage(content="what's my name?")
for event in graph.stream({"messages": [input_msg]}):
    for node, val in event.items():
        if "messages" in val:
            print(val["messages"][-1].content)


# ---------- 带 memory 的图 ----------
print("\n=== With Memory ===")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

input_msg = HumanMessage(content="hi! I'm bob")
graph.invoke({"messages": [input_msg]}, config=config)
messages = graph.get_state(config).values["messages"]
print(messages[-1].content)

input_msg = HumanMessage(content="what's my name?")
graph.invoke({"messages": messages+[input_msg]}, config=config)
messages = graph.get_state(config).values["messages"]
print(messages[-1].content)


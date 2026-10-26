from __future__ import annotations
"""
customer_support_agent.py
Updated for LangGraph 0.3.x
"""

# ==================== SETING UP ENVIRONMENTS ====================
# Proxy and Key settings  -----------------------------------------------------
import os
from dotenv import load_dotenv
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
# CRITICAL: Bypass the proxy for local Tempo traffic
os.environ['no_proxy'] = "localhost,127.0.0.1"
os.environ['NO_PROXY'] = "localhost,127.0.0.1"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
load_dotenv()

import os
import json
from typing import Annotated, Sequence, TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Setup Traceloop with local Grafana stack: Loki, Tempo, Prometheus ------------
from traceloop.sdk import Traceloop
from src.common.observability.loki_logger import log_to_loki

# Environment Configuration, alternatives for stack without Prometheus ---------
# 1. Force Traceloop to local Tempo OTLP receiver, no http prefix for gRPC
os.environ["TRACELOOP_BASE_URL"] = "localhost:4317"
# 2. Clear this to prevent it from defaulting to Traceloop Cloud
os.environ["TRACELOOP_API_KEY"] = ""
os.environ["TRACELOOP_METRICS_ENABLED"] = "true"   # false
# 3. Setup OTEL to localhost with gRPC
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "localhost:4317"
os.environ["OTEL_EXPORTER_OTLP_INSECURE"] = "true"
os.environ["OTEL_METRICS_EXPORTER"] = "oltp"   # none
# 4. Ray environment configurations without GPU
# Down grade to ray==2.50.0
# https://github.com/ray-project/ray/issues/59665
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_ENABLE_OTEL_METRICS"] = "1"    # 0

# --- Tools Definition ---

@tool
def send_customer_message(order_id: str, text: str) -> str:
    """Send a plain response to the customer."""
    print(f"[TOOL] send_customer_message → {text}")
    log_to_loki("tool.send_customer_message", f"order_id={order_id}, text={text}")
    return "sent"

@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund for the given order."""
    print(f"[TOOL] issue_refund(order_id={order_id}, amount={amount})")
    log_to_loki("tool.issue_refund", f"order_id={order_id}, amount={amount}")
    return "refund_queued"

@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""
    print(f"[TOOL] cancel_order(order_id={order_id})")
    log_to_loki("tool.cancel_order", f"order_id={order_id}")
    return "cancelled"

# The original type of shipping_address is a dictionary, as in the sample data
# but there's no example in the systemprompt, so 
@tool
def update_address_for_order(order_id: str, shipping_address: dict) -> str:  #shipping_address: dict
    """Change the shipping address for a pending order."""
    print(f"[TOOL] update_address_for_order(order_id={order_id}, address={shipping_address})")
    log_to_loki("tool.update_address_for_order", f"order_id={order_id}, address={shipping_address}")
    return "address_updated"

TOOLS = [send_customer_message, issue_refund, cancel_order, update_address_for_order]

# --- Agent Configuration ---

Traceloop.init(disable_batch=True, app_name="customer_support_agent")

llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.0, 
    streaming=True
).bind_tools(TOOLS)

class AgentState(TypedDict):
    # add_messages allows the graph to append new messages to the history automatically
    order: Optional[dict]
    messages: Annotated[list[BaseMessage], add_messages]

# --- Nodes ---

def call_model(state: AgentState):
    """The assistant node that decides which action to take."""
    order = state.get("order") or {"order_id": "UNKNOWN", "status": "unknown", "total": 0.0}
    order_json = json.dumps(order, ensure_ascii=False)

    # Remove the following line in system prompt, 
    # because expected_function_call in sample data doesn't includes it.
    # "2) Call send_customer_message to explain what you did.\n"
    system_prompt = (
        "You are a helpful e-commerce support agent.\n"
        "When a user has a problem, you MUST:\n"
        "1) Call a business tool (issue_refund, cancel_order, etc.) if appropriate.\n"
        "2) If you call the update_address_for_order tool, you should give the shipping_address"
        "  in a python dictionary format like the following:"
        "{\"name\":\"Customer2\",\"street1\":\"175 Elm St\",\"street2\":\"\",\"city\":\"Metropolis\",\"state\":\"CA\",\"zip\":\"95303\",\"country\":\"US\"}"
        f"ORDER CONTEXT: {order_json}"
    )

    # Prepend system message to the conversation
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    
    # We return the response as a list because 'messages' is Annotated with add_messages
    return {"messages": [response]}

# --- Graph Construction ---

def construct_graph():
    workflow = StateGraph(AgentState)

    # Add the primary assistant node
    workflow.add_node("assistant", call_model)
    
    # Add the standard ToolNode to handle execution of TOOLS
    tool_node = ToolNode(TOOLS)
    workflow.add_node("tools", tool_node)

    # Define the flow
    workflow.add_edge(START, "assistant")
    
    # tools_condition checks if the LLM produced tool_calls. 
    # If yes, go to "tools"; if no, go to END.
    workflow.add_conditional_edges(
        "assistant",
        tools_condition,
    )

    # After tools are executed, go back to assistant to formulate the final response
    workflow.add_edge("tools", "assistant")

    return workflow.compile()

graph = construct_graph()

if __name__ == "__main__":
    example_order = {"order_id": "A12345", "status": "Delivered", "total": 19.99}
    initial_input = {
        "order": example_order,
        "messages": [HumanMessage(content="My mug arrived broken. Refund?")]
    }
    
    # Use stream or invoke to run the graph
    # for event in graph.stream(initial_input, stream_mode="values"):
    #     if "messages" in event:
    #         last_msg = event["messages"][-1]
    #         print(f"[{type(last_msg).__name__}]: {last_msg.content}")

    result = graph.invoke(initial_input)
    for m in result["messages"]:
        print(f"{m.type}: {m.content}")



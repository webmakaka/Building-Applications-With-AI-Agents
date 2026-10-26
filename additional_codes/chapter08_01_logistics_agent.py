from __future__ import annotations
"""
supply_chain_logistics_agent.py
LangGraph workflow for a Supply Chain & Logistics Management agent,
handling inventory management, shipping operations, supplier relations, and warehouse optimization.
Updated for LangGraph 0.3.x API compatibility.
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
from typing import TypedDict, List, Dict, Optional, Literal


from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from langchain.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

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

# ==================== TOOL DEFINITIONS ====================
# Updated with Optional[str] to satisfy Pydantic v2 validation
# Shared tool for all specialists
@tool
def send_logistics_response(operation_id: Optional[str] = None, message: Optional[str] = None) -> str:
    """Send logistics updates, recommendations, or status reports to stakeholders."""
    print(f"[TOOL] send_logistics_response → {message}")
    log_to_loki("tool.send_logistics_response", f"operation_id={operation_id}, message={message}")
    return "logistics_response_sent"

# Inventory & Warehouse Specialist Tools
@tool
def manage_inventory(sku: Optional[str] = None, **kwargs) -> str:
    """Manage inventory levels, stock replenishment, audits, and optimization strategies."""
    print(f"[TOOL] manage_inventory(sku={sku}, kwargs={kwargs})")
    log_to_loki("tool.manage_inventory", f"sku={sku}")
    return "inventory_management_initiated"

@tool
def optimize_warehouse(operation_type: Optional[str] = None, **kwargs) -> str:
    """Optimize warehouse operations, layout, capacity, and storage efficiency."""
    print(f"[TOOL] optimize_warehouse(operation_type={operation_type}, kwargs={kwargs})")
    log_to_loki("tool.optimize_warehouse", f"operation_type={operation_type}")
    return "warehouse_optimization_initiated"

@tool
def forecast_demand(season: Optional[str] = None, **kwargs) -> str:
    """Analyze demand patterns, seasonal trends, and create forecasting models."""
    print(f"[TOOL] forecast_demand(season={season}, kwargs={kwargs})")
    log_to_loki("tool.forecast_demand", f"season={season}")
    return "demand_forecast_generated"

@tool
def manage_quality(supplier: Optional[str] = None, **kwargs) -> str:
    """Manage quality control, defect tracking, and supplier quality standards."""
    print(f"[TOOL] manage_quality(supplier={supplier}, kwargs={kwargs})")
    log_to_loki("tool.manage_quality", f"supplier={supplier}")
    return "quality_management_initiated"

@tool
def scale_operations(scaling_type: Optional[str] = None, **kwargs) -> str:
    """Scale operations for peak seasons, capacity planning, and workforce management."""
    print(f"[TOOL] scale_operations(scaling_type={scaling_type}, kwargs={kwargs})")
    log_to_loki("tool.scale_operations", f"scaling_type={scaling_type}")
    return "operations_scaled"

@tool
def optimize_costs(cost_type: Optional[str] = None, **kwargs) -> str:
    """Analyze and optimize transportation, storage, and operational costs."""
    print(f"[TOOL] optimize_costs(cost_type={cost_type}, kwargs={kwargs})")
    log_to_loki("tool.optimize_costs", f"cost_type={cost_type}")
    return "cost_optimization_initiated"

INVENTORY_TOOLS = [manage_inventory, optimize_warehouse, forecast_demand, manage_quality, scale_operations, optimize_costs, send_logistics_response]

# Transportation & Logistics Specialist Tools
@tool
def track_shipments(origin: Optional[str] = None, **kwargs) -> str:
    """Track shipment status, delays, and coordinate delivery logistics."""
    print(f"[TOOL] track_shipments(origin={origin}, kwargs={kwargs})")
    log_to_loki("tool.track_shipments", f"origin={origin}")
    return "shipment_tracking_updated"

@tool
def arrange_shipping(shipping_type: Optional[str] = None, **kwargs) -> str:
    """Arrange shipping methods, expedited delivery, and multi-modal transportation."""
    print(f"[TOOL] arrange_shipping(shipping_type={shipping_type}, kwargs={kwargs})")
    log_to_loki("tool.arrange_shipping", f"shipping_type={shipping_type}")
    return "shipping_arranged"

@tool
def coordinate_operations(operation_type: Optional[str] = None, **kwargs) -> str:
    """Coordinate complex operations like cross-docking, consolidation, and transfers."""
    print(f"[TOOL] coordinate_operations(operation_type={operation_type}, kwargs={kwargs})")
    log_to_loki("tool.coordinate_operations", f"operation_type={operation_type}")
    return "operations_coordinated"

@tool
def manage_special_handling(product_type: Optional[str] = None, **kwargs) -> str:
    """Handle special requirements for hazmat, cold chain, and sensitive products."""
    print(f"[TOOL] manage_special_handling(product_type={product_type}, kwargs={kwargs})")
    log_to_loki("tool.manage_special_handling", f"product_type={product_type}")
    return "special_handling_managed"

@tool
def process_returns(returned_quantity: Optional[str] = None, **kwargs) -> str:
    """Process returns, reverse logistics, and product disposition."""
    print(f"[TOOL] process_returns(returned_quantity={returned_quantity}, kwargs={kwargs})")
    log_to_loki("tool.process_returns", f"returned_quantity={returned_quantity}")
    return "returns_processed"

@tool
def optimize_delivery(delivery_type: Optional[str] = None, **kwargs) -> str:
    """Optimize delivery routes, last-mile logistics, and sustainability initiatives."""
    print(f"[TOOL] optimize_delivery(delivery_type={delivery_type}, kwargs={kwargs})")
    log_to_loki("tool.optimize_delivery", f"delivery_type={delivery_type}")
    return "delivery_optimization_complete"

@tool
def manage_disruption(disruption_type: Optional[str] = None, **kwargs) -> str:
    """Manage supply chain disruptions, contingency planning, and risk mitigation."""
    print(f"[TOOL] manage_disruption(disruption_type={disruption_type}, kwargs={kwargs})")
    log_to_loki("tool.manage_disruption", f"disruption_type={disruption_type}")
    return "disruption_managed"

TRANSPORTATION_TOOLS = [track_shipments, arrange_shipping, coordinate_operations, manage_special_handling, process_returns, optimize_delivery, manage_disruption, send_logistics_response]

# Supplier & Compliance Specialist Tools
@tool
def evaluate_suppliers(supplier_name: Optional[str] = None, **kwargs) -> str:
    """Evaluate supplier performance, conduct audits, and manage supplier relationships."""
    print(f"[TOOL] evaluate_suppliers(supplier_name={supplier_name}, kwargs={kwargs})")
    log_to_loki("tool.evaluate_suppliers", f"supplier_name={supplier_name}")
    return "supplier_evaluation_complete"

@tool
def handle_compliance(compliance_type: Optional[str] = None, **kwargs) -> str:
    """Manage regulatory compliance, customs, documentation, and certifications."""
    print(f"[TOOL] handle_compliance(compliance_type={compliance_type}, kwargs={kwargs})")
    log_to_loki("tool.handle_compliance", f"compliance_type={compliance_type}")
    return "compliance_handled"

SUPPLIER_TOOLS = [evaluate_suppliers, handle_compliance, send_logistics_response]
TOOLS = INVENTORY_TOOLS+TRANSPORTATION_TOOLS+SUPPLIER_TOOLS

# Initialize Traceloop, for sendnig Actor trace to local Tempo
Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent")
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.0, 
    streaming=False,
    verbose=True
).bind_tools(TOOLS)

class AgentState(TypedDict):
    operation: Optional[dict]  # Supply chain operation information
    messages: List[BaseMessage]  # 保持简单的List类型，不使用add_messages

def call_model(state: AgentState):
    """保持原有逻辑：单次调用，内部处理工具调用和第二次模型调用"""
    history = state["messages"]
    
    # Handle missing or incomplete operation data gracefully
    operation = state.get("operation", {})
    if not operation:
        operation = {"operation_id": "UNKNOWN", "type": "general", "priority": "medium", "status": "active"}
    
    operation_json = json.dumps(operation, ensure_ascii=False)
    system_prompt = (
        "You are an experienced Supply Chain & Logistics Management professional.\n"
        "Your expertise covers:\n"
        "- Inventory management and demand forecasting\n"
        "- Transportation and shipping optimization\n"
        "- Supplier relationship management and evaluation\n"
        "- Warehouse operations and capacity planning\n"
        "- Quality control and compliance management\n"
        "- Cost optimization and operational efficiency\n"
        "- Risk management and disruption response\n"
        "- Sustainability and green logistics initiatives\n"
        "\n"
        "When managing supply chain operations:\n"
        "  1) Analyze the logistics challenge or opportunity\n"
        "  2) Call the appropriate supply chain management tool\n"
        "  3) Follow up with send_logistics_response to provide recommendations\n"
        "  4) Consider cost, efficiency, quality, and sustainability impacts\n"
        "  5) Prioritize customer satisfaction and business continuity\n"
        "\n"
        "Always balance cost optimization with service quality and risk mitigation.\n\n"
        f"OPERATION: {operation_json}"
    )

    full = [SystemMessage(content=system_prompt)] + history

    first = llm.invoke(full)
    messages = [first]

    # 保持原有的工具调用逻辑
    if hasattr(first, "tool_calls") and first.tool_calls:
        for tc in first.tool_calls:
            print(f"Tool call: {tc['name']} with args: {tc['args']}")
            # 找到对应的工具
            fn = next(t for t in TOOLS if t.name == tc['name'])
            out = fn.invoke(tc["args"])
            messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"], name=tc["name"]))

        # 第二次模型调用（保持原有逻辑）
        second = llm.invoke(full + messages)
        messages.append(second)

    # 返回所有生成的消息（包括原始历史消息）
    return {"messages": history + messages}

def construct_graph():
    """构建与原来相同的简单图结构：单个节点，无循环"""
    g = StateGraph(AgentState)
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    g.add_edge("assistant", END)  # 直接连接到END，无循环
    return g.compile()

graph = construct_graph()

if __name__ == "__main__":
    example = {"operation_id": "OP-12345", "type": "inventory_management", "priority": "high", "location": "Warehouse A"}
    convo = [HumanMessage(content="We're running critically low on SKU-12345. Current stock is 50 units but we have 200 units on backorder. What's our reorder strategy?")]
    
    # 初始化状态
    initial_state = AgentState(operation=example, messages=convo)
    
    # 调用图（保持原有调用方式）
    result = graph.invoke(initial_state)
    
    # 打印结果
    for m in result["messages"]:
        if hasattr(m, 'tool_calls'):
            print(f"{m.type}: [Tool calls: {len(m.tool_calls)}]")
            for tool_call in m.tool_calls:
                print(f'name:{tool_call["name"]}, args:{tool_call["args"]}')
        elif hasattr(m, 'content'):
            print(f"{m.type}: {m.content}")


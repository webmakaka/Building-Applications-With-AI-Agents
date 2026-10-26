from __future__ import annotations
"""
supply_chain_logistics_agent.py
LangGraph workflow for a multi-agent Supply Chain & Logistics Management system.
Compatible with LangGraph 0.3.x - Simplified version
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
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START

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
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, streaming=False)

# Create LLMs with bound tools
inventory_llm = llm.bind_tools(INVENTORY_TOOLS)
transportation_llm = llm.bind_tools(TRANSPORTATION_TOOLS)
supplier_llm = llm.bind_tools(SUPPLIER_TOOLS)

# ==================== AGENT SETUP ====================

# ==================== STATE DEFINITION ====================
class AgentState(TypedDict):
    """State for the supply chain agent workflow"""
    operation: Optional[dict]
    messages: List[BaseMessage]

# ==================== NODE FUNCTIONS ====================
# Supervisor (Manager) Node: Routes to the appropriate specialist
def supervisor_node(state: AgentState) -> AgentState:
    """Routes the task to the appropriate specialist agent"""
    messages = state["messages"]
    operation = state.get("operation", {})
    operation_json = json.dumps(operation, ensure_ascii=False)
    
    supervisor_prompt = (
        "You are a supervisor coordinating a team of supply chain specialists.\n"
        "Team members:\n"
        "- inventory: Handles inventory levels, forecasting, quality, warehouse optimization, scaling, and costs.\n"
        "- transportation: Handles shipping tracking, arrangements, operations coordination, special handling, returns, delivery optimization, and disruptions.\n"
        "- supplier: Handles supplier evaluation and compliance.\n"
        "\n"
        "Based on the user query, select ONE team member to handle it.\n"
        "Output ONLY the selected member's name (inventory, transportation, or supplier), nothing else.\n\n"
        f"OPERATION: {operation_json}"
    )

    full_messages = [SystemMessage(content=supervisor_prompt)] + messages
    response = llm.invoke(full_messages)
    
    return {"messages": messages + [response]}

# Specialist Node Template (保持原始逻辑)
def specialist_node(state: AgentState, specialist_llm, system_prompt: str) -> AgentState:
    """Generic specialist node that calls tools and returns results"""
    messages = state["messages"]
    operation = state.get("operation", {})
    if not operation:
        operation = {"operation_id": "UNKNOWN", "type": "general", "priority": "medium", "status": "active"}
    operation_json = json.dumps(operation, ensure_ascii=False)
    
    full_prompt = system_prompt + f"\n\nOPERATION: {operation_json}"
    full_messages = [SystemMessage(content=full_prompt)] + messages

    # First AI response (可能包含工具调用)
    first_response = specialist_llm.invoke(full_messages)
    messages = messages + [first_response]

    # 如果包含工具调用，执行工具
    if hasattr(first_response, 'tool_calls') and first_response.tool_calls:
        for tool_call in first_response.tool_calls:
            # 查找对应的工具
            all_tools = INVENTORY_TOOLS + TRANSPORTATION_TOOLS + SUPPLIER_TOOLS
            tool_fn = next(t for t in all_tools if t.name == tool_call['name'])
            tool_result = tool_fn.invoke(tool_call["args"])
            
            # 添加工具结果消息
            messages.append(ToolMessage(
                content=str(tool_result), 
                tool_call_id=tool_call["id"]
            ))
        
        # 第二次AI响应（处理工具结果）
        second_response = specialist_llm.invoke(full_messages + messages[1:])
        messages.append(second_response)

    return {"messages": messages}

# Inventory Specialist Node
def inventory_node(state: AgentState) -> AgentState:
    inventory_prompt = (
        "You are an inventory and warehouse management specialist.\n"
        "When managing:\n"
        "  1) Analyze the inventory/warehouse challenge\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider cost, efficiency, and scalability."
    )
    return specialist_node(state, inventory_llm, inventory_prompt)

# Transportation Specialist Node
def transportation_node(state: AgentState) -> AgentState:
    transportation_prompt = (
        "You are a transportation and logistics specialist.\n"
        "When managing:\n"
        "  1) Analyze the shipping/delivery challenge\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider efficiency, sustainability, and risk mitigation."
    )
    return specialist_node(state, transportation_llm, transportation_prompt)

# Supplier Specialist Node
def supplier_node(state: AgentState) -> AgentState:
    supplier_prompt = (
        "You are a supplier relations and compliance specialist.\n"
        "When managing:\n"
        "  1) Analyze the supplier/compliance issue\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider performance, regulations, and relationships."
    )
    return specialist_node(state, supplier_llm, supplier_prompt)

# Routing function for conditional edges (保持原始逻辑)
def route_to_specialist(state: AgentState) -> Literal["inventory", "transportation", "supplier", END]:
    last_message = state["messages"][-1]
    agent_name = last_message.content.strip().lower()
    
    if agent_name == "inventory":
        return "inventory"
    elif agent_name == "transportation":
        return "transportation"
    elif agent_name == "supplier":
        return "supplier"
    else:
        # Fallback if no match
        return END

# ==================== GRAPH CONSTRUCTION ====================
def construct_graph():
    """Build the LangGraph workflow - Simplified version matching original structure"""
    g = StateGraph(AgentState)
    
    # Add nodes (保持原始节点)
    g.add_node("supervisor", supervisor_node)
    g.add_node("inventory", inventory_node)
    g.add_node("transportation", transportation_node)
    g.add_node("supplier", supplier_node)
    
    # Set entry point
    g.add_edge(START, "supervisor")
    
    # Add conditional edges from supervisor (保持原始路由逻辑)
    g.add_conditional_edges(
        "supervisor",
        route_to_specialist,
        {
            "inventory": "inventory",
            "transportation": "transportation", 
            "supplier": "supplier",
            END: END
        }
    )
    
    # Add edges from specialists to END (直接结束，无循环)
    g.add_edge("inventory", END)
    g.add_edge("transportation", END)
    g.add_edge("supplier", END)
    
    return g.compile()

# ==================== MAIN EXECUTION ====================
graph = construct_graph()

if __name__ == "__main__":
    # 保持原始示例不变
    example = {
        "operation_id": "OP-12345", 
        "type": "inventory_management", 
        "priority": "high", 
        "location": "Warehouse A"
    }
    
    convo = [
        HumanMessage(content="We're running critically low on SKU-12345. Current stock is 50 units but we have 200 units on backorder. What's our reorder strategy?")
    ]
    
    # 初始化状态
    initial_state = AgentState(
        operation=example,
        messages=convo
    )
    
    # 执行图
    result = graph.invoke(initial_state)
    
    # 打印结果
    for m in result["messages"]:
        if hasattr(m, 'tool_calls') and len(m.tool_calls)>0:
            print(f"{m.type}: [Tool calls: {len(m.tool_calls)}]")
            for tool_call in m.tool_calls:
                print(f'name:{tool_call["name"]}, args:{tool_call["args"]}')
        elif hasattr(m, 'content'):
            print(f"{m.type}: {m.content}")


from __future__ import annotations
"""
supply_chain_logistics_agent_ray_per_session.py
LangGraph workflow for a multi-agent Supply Chain & Logistics Management system, revised to use Ray actors for decoupling with per-session isolation.
Handles inventory management, shipping operations, supplier relations, and warehouse optimization through specialized agents as Ray actors.
Each session (identified by operation_id) gets its own actor instances per specialist type, ensuring isolated state and sequential execution per session.
The supervisor determines the specialist and invokes the session-specific actor remotely via a SessionManager.
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
import time
import operator
from typing import TypedDict, List, Optional, Literal, Annotated, Sequence

import ray
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

# Initialize Traceloop, for sendnig Actor trace to local Tempo
Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent_ray")

# ==================== STATE DEFINITION ====================
class AgentState(TypedDict):
    """State matching the structure in chapter8_3.py"""
    operation: Optional[dict]
    messages: Annotated[Sequence[BaseMessage], operator.add]

# ==================== RAY ACTOR LOGIC ====================
@ray.remote
class SpecialistActor:
    """Ray Actor for per-session isolation"""
    def __init__(self, name: str, system_prompt: str, tool_list: list):
        self.name = name
        self.prompt = system_prompt
        # Re-initialize the bound LLM inside the actor to avoid serialization issues
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0).bind_tools(tool_list)
        self.tools = {t.name: t for t in tool_list}
        self.internal_state = {}  # Isolated per-session state, e.g., for tracking within the session

    def process_task(self, operation: Optional[dict], messages: Sequence[BaseMessage]):
        if not operation:
            operation = {"operation_id": "UNKNOWN", "status": "active"}
        
        operation_json = json.dumps(operation, ensure_ascii=False)
        full_prompt = self.prompt + f"\n\nOPERATION: {operation_json}"
        
        # Prepare context
        full_messages = [SystemMessage(content=full_prompt)] + list(messages)
        
        # First AI Call
        response = self.llm.invoke(full_messages)
        output_messages = [response]

        # Tool execution logic mirrored from specialist_node in chapter8_3.py
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_fn = self.tools.get(tool_call['name'])
                if tool_fn:
                    # Invoke tool and wrap in ToolMessage
                    tool_result = tool_fn.invoke(tool_call["args"])
                    output_messages.append(ToolMessage(
                        content=str(tool_result), 
                        tool_call_id=tool_call["id"]
                    ))
            
            # Second AI Call to process tool outputs
            final_response = self.llm.invoke(full_messages + output_messages)
            output_messages.append(final_response)

        # Update internal state (example: track processed steps within session)
        step_key = str(len(self.internal_state) + 1)  # Or use a more specific key
        self.internal_state[step_key] = {"status": "processed", "timestamp": time.time()}

        return output_messages

    def get_state(self):
        return self.internal_state  # Return entire session state


@ray.remote
class SessionManager:
    """Tracks per-session specialist actors"""
    def __init__(self):
        self.sessions = {}

    def get_actor(self, session_id: str, agent_name: str, prompt: str, tools: list):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        if agent_name not in self.sessions[session_id]:
            actor = SpecialistActor.remote(agent_name, prompt, tools)
            self.sessions[session_id][agent_name] = actor
        return self.sessions[session_id][agent_name]

    def get_session_state(self, session_id: str, agent_name: str):
        if session_id in self.sessions and agent_name in self.sessions[session_id]:
            actor = self.sessions[session_id][agent_name]
            return actor.get_state.remote()  # Returns future
        return None

# ==================== LANGGRAPH NODES ====================

def supervisor_node(state: AgentState):
    """Standard supervisor routing node"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    operation = state.get("operation", {})
    
    prompt = (
        "You are a supervisor coordinating supply chain specialists.\n"
        "Select ONE: inventory, transportation, or supplier.\n"
        "Output ONLY the name.\n\n"
        f"OPERATION: {json.dumps(operation)}"
    )

    response = llm.invoke([SystemMessage(content=prompt)] + list(state["messages"]))
    return {"messages": [response]}

def call_ray_agent(state: AgentState, agent_name: str, prompt: str, tools: list):
    """Bridge node that invokes the remote Ray Actor"""
    session_id = state.get("operation", {}).get("operation_id", "GLOBAL_SESSION")
    
    # Access the manager via named actor
    manager = ray.get_actor("session_manager")
    
    # Retrieve the handle for the session-specific actor
    actor_handle = ray.get(manager.get_actor.remote(session_id, agent_name, prompt, tools))
    
    # Trigger remote processing
    new_messages = ray.get(actor_handle.process_task.remote(state["operation"], state["messages"]))
    
    return {"messages": new_messages}

# Specialist wrapper nodes
def inventory_node(state: AgentState):
    return call_ray_agent(state, "inventory", "You are an inventory specialist.", INVENTORY_TOOLS)

def transportation_node(state: AgentState):
    return call_ray_agent(state, "transportation", "You are a logistics specialist.", TRANSPORTATION_TOOLS)

def supplier_node(state: AgentState):
    return call_ray_agent(state, "supplier", "You are a supplier specialist.", SUPPLIER_TOOLS)

def route_to_specialist(state: AgentState) -> Literal["inventory", "transportation", "supplier", END]:
    last_msg = state["messages"][-1].content.strip().lower()
    if last_msg in ["inventory", "transportation", "supplier"]:
        return last_msg
    return END

# ==================== GRAPH CONSTRUCTION ====================

def construct_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("inventory", inventory_node)
    workflow.add_node("transportation", transportation_node)
    workflow.add_node("supplier", supplier_node)
    
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", route_to_specialist)
    
    workflow.add_edge("inventory", END)
    workflow.add_edge("transportation", END)
    workflow.add_edge("supplier", END)
    
    return workflow.compile()

# ==================== EXECUTION ====================

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, logging_level="ERROR") # Silences most RpcErrors
    manager = SessionManager.options(name="session_manager").remote()
    
    graph_app = construct_graph()
    
    test_input = {
        "operation": {"operation_id": "OP-12345", "type": "inventory"},
        "messages": [HumanMessage(content="Check status for SKU-123 and let me know.")]
    }
    
    print("\n--- Starting Supply Chain Workflow ---")
    for output in graph_app.stream(test_input):
        for node, data in output.items():
            print(f"\n[Node: {node}]")
            for msg in data.get("messages", []):
                # Distinguish between types of messages
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  AI calling tool: {tc['name']}({tc['args']})")
                elif isinstance(msg, ToolMessage):
                    print(f"  Tool Result: {msg.content}")
                else:
                    role = "Supervisor" if node == "supervisor" else "Specialist"
                    print(f"  {role}: {msg.content}")


    # Optional: Query session-specific actor state
    state_ref = manager.get_session_state.remote("OP-12345", "inventory")
    if state_ref:
        state = ray.get(ray.get(state_ref))  # Resolve nested future
        print("Session actor state:", state)

    ray.shutdown()


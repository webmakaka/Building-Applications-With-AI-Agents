from __future__ import annotations
"""
supply_chain_logistics_agent_temporal.py
LangGraph workflow for a multi-agent Supply Chain & Logistics Management system, revised to use Temporal for durable orchestration.
Handles inventory management, shipping operations, supplier relations, and warehouse optimization through specialized agents orchestrated via Temporal workflows.
The workflow sequences agent steps with retries, persistent state, and failure recovery, ideal for long-running supply chain processes.
"""

import os
from dotenv import load_dotenv

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
from datetime import timedelta
from typing import Annotated, Sequence, TypedDict, Optional, Dict, Any, List, Literal

# --- Temporal Imports ---
from temporalio import workflow, activity
from temporalio.common import RetryPolicy
from temporalio.client import Client
from temporalio.worker import Worker

# --- LangChain 0.3.x Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool

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
Traceloop.init(disable_batch=True, app_name="supply_chain_workflow")

# ==================== AGENT CONFIGURATION ====================

ALL_TOOLS_MAP = {t.name: t for t in INVENTORY_TOOLS + TRANSPORTATION_TOOLS + SUPPLIER_TOOLS}

# Initialize LLMs at the module level so they are available to workers
llm = ChatOpenAI(model="gpt-4o", temperature=0)

inventory_llm = llm.bind_tools(INVENTORY_TOOLS)
transportation_llm = llm.bind_tools(TRANSPORTATION_TOOLS)
supplier_llm = llm.bind_tools(SUPPLIER_TOOLS)

LLM_MAP = {
    "inventory": inventory_llm,
    "transportation": transportation_llm,
    "supplier": supplier_llm
}

# Define prompts (as in original)
PROMPTS = {
    "inventory": "You are an inventory specialist. Analyze stock levels and use tools to manage replenishment.",
    "transportation": "You are a logistics specialist. Track shipments and optimize delivery routes.",
    "supplier": "You are a supplier specialist. Evaluate performance and handle compliance."
}

# ==================== TEMPORAL ACTIVITIES ====================

@activity.defn
async def supervisor_activity(data: Dict[str, Any]) -> str:
    """Routes the task to the appropriate specialist."""
    messages = [HumanMessage(content=m["content"]) if m["type"] == "human" else AIMessage(content=m["content"]) 
                for m in data["messages"]]
    operation_json = json.dumps(data["operation"])
    
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
    
    response = await llm.ainvoke([SystemMessage(content=supervisor_prompt)] + messages)
    return response.content.strip().lower()

async def _run_specialist_logic(agent_name: str, operation: Dict[str, Any], messages_dict: List[Dict]) -> List[Dict]:
    """Shared internal logic for specialist processing matching chapter8_3 patterns."""
    specialist_llm = LLM_MAP[agent_name]
    system_prompt = PROMPTS[agent_name] + f"\nContext: {json.dumps(operation)}"
    
    # Reconstruct messages from dicts
    messages = []
    for m in messages_dict:
        if m["type"] == "human": messages.append(HumanMessage(content=m["content"]))
        elif m["type"] == "ai": messages.append(AIMessage(content=m["content"]))

    full_messages = [SystemMessage(content=system_prompt)] + messages
    
    # First LLM Call
    response = await specialist_llm.ainvoke(full_messages)
    output_messages = [response]
    
    # Handle Tool Calls (LangChain 0.3.x style)
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_fn = ALL_TOOLS_MAP.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                output_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        
        # Second LLM Call with tool results
        final_response = await specialist_llm.ainvoke(full_messages + output_messages)
        output_messages.append(final_response)

    # Convert back to serializable dicts for Temporal return
    return [{"content": m.content, "type": m.type, "tool_calls": getattr(m, 'tool_calls', [])} for m in output_messages]

# Define Three Distinct Specialist Activities
@activity.defn(name="inventory_activity")
async def inventory_activity(data: Dict[str, Any]) -> List[Dict]:
    return await _run_specialist_logic("inventory", data["operation"], data["messages"])

@activity.defn(name="transportation_activity")
async def transportation_activity(data: Dict[str, Any]) -> List[Dict]:
    return await _run_specialist_logic("transportation", data["operation"], data["messages"])

@activity.defn(name="supplier_activity")
async def supplier_activity(data: Dict[str, Any]) -> List[Dict]:
    return await _run_specialist_logic("supplier", data["operation"], data["messages"])

# ==================== TEMPORAL WORKFLOW ====================

@workflow.defn
class SupplyChainWorkflow:
    @workflow.run
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        state = {
            "operation": input_data["operation"],
            "messages": input_data["initial_messages"]
        }
        
        # 1. Supervisor Routing
        agent_name = await workflow.execute_activity(
            supervisor_activity,
            state,
            start_to_close_timeout=timedelta(minutes=1)
        )
        
        # 2. Specialist Execution (Routes to the specific activity function)
        activity_map = {
            "inventory": inventory_activity,
            "transportation": transportation_activity,
            "supplier": supplier_activity
        }
        
        if agent_name in activity_map:
            new_msgs = await workflow.execute_activity(
                activity_map[agent_name],
                state,
                start_to_close_timeout=timedelta(minutes=2)
            )
            state["messages"].extend(new_msgs)
            
        return state

# ==================== WORKER & MAIN ====================
from temporalio.worker import UnsandboxedWorkflowRunner

async def main():
    # 1. Connect to the local server you started with the CLI
    client = await Client.connect("localhost:7233")
    
    activities = [
        supervisor_activity, 
        inventory_activity, 
        transportation_activity, 
        supplier_activity
    ]
    
    # 2. Start the Worker using the Unsandboxed runner
    # This prevents the "AttributeError: module '__main__' has no attribute '__file__'"
    async with Worker(
        client, 
        task_queue="supply-chain-queue", 
        workflows=[SupplyChainWorkflow], 
        activities=activities,
        workflow_runner=UnsandboxedWorkflowRunner() 
    ):
        print("--- Supply Chain Worker Started (Version 1.21.1) ---")
        
        # 3. Execute the workflow immediately for the demo
        example_operation = {
            "operation_id": "OP-12345", 
            "type": "inventory_management", 
            "priority": "high", 
            "location": "Warehouse A"
        }
        example_messages = [{
            "content": "We're running low on SKU-12345. What is our strategy?", 
            "type": "human"
        }]

        print("Starting Workflow execution...")
        result = await client.execute_workflow(
            SupplyChainWorkflow.run,
            {"operation": example_operation, "initial_messages": example_messages},
            id="supply-chain-demo-run",
            task_queue="supply-chain-queue"
        )
        
        print("\n--- Workflow Result ---")
        # Final message content from the specialist
        print(result["messages"][-1])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


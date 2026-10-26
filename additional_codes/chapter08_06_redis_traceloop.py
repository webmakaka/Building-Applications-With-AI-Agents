from __future__ import annotations
"""
supply_chain_logistics_agent_redis. py
LangGraph workflow for a multi-agent Supply Chain & Logistics Management system, revised to use Redis Streams for decoupling. 
Handles inventory management, shipping operations, supplier relations, and warehouse optimization through specialized agents coordinated asynchronously via Redis Streams.
The supervisor publishes tasks to a shared stream, specialists consume and process relevant tasks, and publish responses to another stream.
Compatible with LangGraph 0.3.x
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


import json
import time
import uuid
from typing import TypedDict, List, Optional, Literal, Annotated, Sequence
import threading
import operator

import redis
from langchain_openai. chat_models import ChatOpenAI
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

# Redis connection (assume Redis running on localhost:6379)
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
TASK_STREAM = 'supply-chain-tasks'
RESPONSE_STREAM = 'supply-chain-responses'

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
Traceloop.init(disable_batch=True, app_name="supply_chain_logistics_agent_redis")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0, verbose=True)

# Create a global exit signal
stop_event = threading.Event()

# Bind tools to specialized LLMs
inventory_llm = llm.bind_tools(INVENTORY_TOOLS)
transportation_llm = llm. bind_tools(TRANSPORTATION_TOOLS)
supplier_llm = llm.bind_tools(SUPPLIER_TOOLS)

class AgentState(TypedDict):
    operation: Optional[dict]  # Supply chain operation information
    messages:  Annotated[Sequence[BaseMessage], operator.add]

# Helper to serialize messages for Redis
def serialize_messages(messages: Sequence[BaseMessage]) -> list[dict]:
    return [m.model_dump() for m in messages]

# Helper to deserialize messages from Redis
def deserialize_messages(serialized: list[dict]) -> Sequence[BaseMessage]:
    return [HumanMessage(**m) if m['type'] == 'human' else AIMessage(**m) if m['type'] == 'ai' else ToolMessage(**m) if m['type'] == 'tool' else SystemMessage(**m) for m in serialized]

# Supervisor:  Determines specialist and publishes task to Redis stream
def supervisor_publish(operation: dict, messages: Sequence[BaseMessage]) -> str:
    operation_json = json.dumps(operation, ensure_ascii=False)
    
    supervisor_prompt = (
        "You are a supervisor coordinating a team of supply chain specialists.\n"
        "Team members:\n"
        "- inventory:  Handles inventory levels, forecasting, quality, warehouse optimization, scaling, and costs.\n"
        "- transportation:  Handles shipping tracking, arrangements, operations coordination, special handling, returns, delivery optimization, and disruptions.\n"
        "- supplier: Handles supplier evaluation and compliance.\n"
        "\n"
        "Based on the user query, select ONE team member to handle it.\n"
        "Output ONLY the selected member's name (inventory, transportation, or supplier), nothing else.\n\n"
        f"OPERATION: {operation_json}"
    )

    full = [SystemMessage(content=supervisor_prompt)] + list(messages)
    response = llm.invoke(full)
    
    agent_name = response.content. strip().lower()
    
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    task_id = str(uuid.uuid4())
    task_message = {
        'task_id': task_id,
        'agent':  agent_name,
        'operation': operation,
        'messages':  serialize_messages(messages)
    }
    r.xadd(TASK_STREAM, {'data': json.dumps(task_message, ensure_ascii=False)})
    
    return task_id


# Specialist Node Template
def specialist_node(state: AgentState, specialist_llm, system_prompt: str) -> AgentState:
    """Generic specialist node that calls tools and returns results"""
    messages = state["messages"]
    operation = state.get("operation", {})
    if not operation:
        operation = {"operation_id": "UNKNOWN", "type": "general", "priority": "medium", "status": "active"}
    operation_json = json.dumps(operation, ensure_ascii=False)
    
    full_prompt = system_prompt + f"\n\nOPERATION: {operation_json}"
    full_messages = [SystemMessage(content=full_prompt)] + messages

    # First AI response
    first_response = specialist_llm.invoke(full_messages)
    messages = messages + [first_response]

    # If there's tool call
    if hasattr(first_response, 'tool_calls') and first_response.tool_calls:
        for tool_call in first_response.tool_calls:
            # Find the tool
            all_tools = INVENTORY_TOOLS + TRANSPORTATION_TOOLS + SUPPLIER_TOOLS
            tool_fn = next(t for t in all_tools if t.name == tool_call['name'])
            tool_result = tool_fn.invoke(tool_call["args"])
            
            # Add a ToolMessage
            messages.append(ToolMessage(
                content=str(tool_result), 
                tool_call_id=tool_call["id"]
            ))
        
        # Second AI response with result of tool call
        second_response = specialist_llm.invoke(full_messages + messages[1:])
        messages.append(second_response)

    return {"messages": messages}


# Inventory Specialist Consumer Loop
def inventory_consumer():
    """Inventory Specialist Consumer Loop"""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    last_id = '$'  # Only new message
    inventory_prompt = (
        "You are an inventory and warehouse management specialist.\n"
        "When managing:\n"
        "  1) Analyze the inventory/warehouse challenge\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider cost, efficiency, and scalability."
    )
    
    while not stop_event.is_set(): # 2. Check the signal  # True
        try:
            messages = r.xread({TASK_STREAM: last_id}, count=1, block=5000)
            if messages:
                stream, entries = messages[0]
                for entry_id, entry_data in entries:
                    task = json.loads(entry_data['data'])
                    if task['agent'] == 'inventory':
                        state = {
                            'operation': task['operation'],
                            'messages': list(deserialize_messages(task['messages']))
                        }
                        result = specialist_node(state, inventory_llm, inventory_prompt)
                        response_message = {
                            'task_id':  task['task_id'],
                            'from': 'inventory',
                            'result': {
                                'messages': serialize_messages(result['messages'])
                            }
                        }
                        r.xadd(RESPONSE_STREAM, {'data': json.dumps(response_message, ensure_ascii=False)})
                    last_id = entry_id
        except Exception as e:
            print(f"[ERROR] Inventory consumer error: {e}")
            time.sleep(1)

# Transportation Specialist Consumer Loop
def transportation_consumer():
    """Transportation Specialist Consumer Loop"""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    last_id = '$'  # Only new message
    transportation_prompt = (
        "You are a transportation and logistics specialist.\n"
        "When managing:\n"
        "  1) Analyze the shipping/delivery challenge\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider efficiency, sustainability, and risk mitigation."
    )
    
    while not stop_event.is_set():  # True
        try:
            messages = r.xread({TASK_STREAM:  last_id}, count=1, block=5000)
            if messages:
                stream, entries = messages[0]
                for entry_id, entry_data in entries:
                    task = json. loads(entry_data['data'])
                    if task['agent'] == 'transportation':
                        state = {
                            'operation': task['operation'],
                            'messages': list(deserialize_messages(task['messages']))
                        }
                        result = specialist_node(state, transportation_llm, transportation_prompt)
                        response_message = {
                            'task_id': task['task_id'],
                            'from': 'transportation',
                            'result': {
                                'messages': serialize_messages(result['messages'])
                            }
                        }
                        r. xadd(RESPONSE_STREAM, {'data': json.dumps(response_message, ensure_ascii=False)})
                    last_id = entry_id
        except Exception as e:
            print(f"[ERROR] Transportation consumer error: {e}")
            time.sleep(1)

# Supplier Specialist Consumer Loop
def supplier_consumer():
    """Supplier Specialist Consumer Loop"""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    last_id = '$'  # Only new message
    supplier_prompt = (
        "You are a supplier relations and compliance specialist.\n"
        "When managing:\n"
        "  1) Analyze the supplier/compliance issue\n"
        "  2) Call the appropriate tool\n"
        "  3) Follow up with send_logistics_response\n"
        "Consider performance, regulations, and relationships."
    )
    
    while not stop_event.is_set():  # True
        try: 
            messages = r.xread({TASK_STREAM: last_id}, count=1, block=5000)
            if messages: 
                stream, entries = messages[0]
                for entry_id, entry_data in entries: 
                    task = json.loads(entry_data['data'])
                    if task['agent'] == 'supplier':
                        state = {
                            'operation': task['operation'],
                            'messages': list(deserialize_messages(task['messages']))
                        }
                        result = specialist_node(state, supplier_llm, supplier_prompt)
                        response_message = {
                            'task_id': task['task_id'],
                            'from':  'supplier',
                            'result': {
                                'messages': serialize_messages(result['messages'])
                            }
                        }
                        r.xadd(RESPONSE_STREAM, {'data': json.dumps(response_message, ensure_ascii=False)})
                    last_id = entry_id
        except Exception as e:
            print(f"[ERROR] Supplier consumer error: {e}")
            time.sleep(1)

# Function to wait for response by task_id
def wait_for_response(task_id: str, timeout: int = 60) -> dict:
    """Wait for response from Redis Stream"""
    r = redis. Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    last_id = '$'  # Only new message
    start_time = time.time()
    
    while time.time() - start_time < timeout: 
        try:
            messages = r.xread({RESPONSE_STREAM: last_id}, count=1, block=5000)
            if messages:
                stream, entries = messages[0]
                for entry_id, entry_data in entries:
                    response = json.loads(entry_data['data'])
                    if response['task_id'] == task_id:
                        return response
                    last_id = entry_id
        except Exception as e:
            print(f"[ERROR] Error waiting for response: {e}")
            time.sleep(1)
    
    raise TimeoutError(f"No response received for task {task_id} within {timeout} seconds")

if __name__ == "__main__": 
    t1 = time.time()
    # Start specialist consumers in separate thread
    # Use threads instead of processes for Reticulate compatibility
    inventory_thread = threading.Thread(target=inventory_consumer, daemon=True)
    transportation_thread = threading.Thread(target=transportation_consumer, daemon=True)
    supplier_thread = threading.Thread(target=supplier_consumer, daemon=True)

    inventory_thread.start()
    transportation_thread.start()
    supplier_thread.start()

    # Example invocation
    example_operation = {"operation_id": "OP-12345", "type": "inventory_management", "priority": "high", "location": "Warehouse A"}
    example_messages = [HumanMessage(content="We're running critically low on SKU-12345. Current stock is 50 units but we have 200 units on backorder. What's our reorder strategy?")]

    try:
        task_id = supervisor_publish(example_operation, example_messages)
        print(f"Task published with ID: {task_id}")

        response = wait_for_response(task_id)
        print("\n" + "="*80)
        print("Received response:")
        print("="*80)
        for m in deserialize_messages(response['result']['messages']):
            print(m.type,":")
            print(m)

    except TimeoutError as e:
        print(f"[ERROR] {e}")
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # 3. Signal all threads to stop
        stop_event.set()        
        inventory_thread.join()
        transportation_thread.join()
        supplier_thread.join()
        
        print("Processes terminated")

    t2 = time.time()
    print("\nTotal time: ",t2-t1)



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


# 这里开始是本例自己的代码 -----------------------------------------------------
# 工具集合，引用上例中定义的所有工具 
all_tools = INVENTORY_TOOLS+TRANSPORTATION_TOOLS+SUPPLIER_TOOLS
tool_map = {tool.name: tool for tool in all_tools}

# ==================== LLM 配置 ====================
# 使用不同的temperature：actor用0.2增加多样性，critic用0.0保持稳定性
llm_actor = ChatOpenAI(model="gpt-4o", temperature=0.2)
llm_critic = ChatOpenAI(model="gpt-4o", temperature=0.0)

llm_actor_with_tools = llm_actor.bind_tools(all_tools)
llm_critic_with_tools = llm_critic.bind_tools(all_tools)

# ==================== 状态定义 ====================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    candidates: Optional[List[Dict]]

# ==================== Actor 节点 ====================
def actor_node(state: AgentState) -> AgentState:
    """Generate 3 diverse candidate plans using tool calls"""
    history = state["messages"]
    query = history[-1].content

    # 改进的Actor提示词 - 强调多样性
    actor_prompt = """Generate a complete supply chain execution plan for the given situation.
    
    Use the available tools to create a coherent plan with multiple steps.
    Create a plan with 2-4 tool calls in logical sequence.
 
    IMPORTANT: Generate DIFFERENT plans for each request. Vary:
    1. Reorder quantity (consider current stock + backorder + safety buffer)
    2. Shipping strategy (express, standard, priority, etc.)
    3. Logistic cost and warehouse efficency
    4. Tool call sequences (order and combination of tools)
   
    For diversity, consider different strategies:
    - Aggressive: Large reorder with express shipping
    - Conservative: Split orders with different priorities
    - Balanced: Medium reorder with priority shipping
    - Optimized: Based on demand forecasting (if applicable)
    """
    
    candidates = []
    plan_focuses = [
        "Focus on fastest replenishment (express shipping, large quantity)",
        "Focus on cost optimization (balanced shipping, moderate quantity)",
        "Focus on risk mitigation (split shipments, multiple priorities)"
    ]
    
    # 生成3个候选计划
    for i in range(3):
        # 为每个计划添加独特的上下文
        plan_context = f"""Generate plan {i+1}/3 with a focus on: {plan_focuses[i]}
        
        Make this plan DISTINCT from others by varying:
        - Reorder quantity (suggest: {150 + i*50} to {300 - i*50} units)
        - Shipping priority (options: urgent, high, medium, standard)
        - Logistic cost and warehouse efficiency strategy
        
        Ensure the plan is practical for the following situation: 
        {query}"""
        
        response = llm_actor_with_tools.invoke([
            SystemMessage(content=actor_prompt),
            # *history,
            HumanMessage(content=plan_context)
        ])
        
        # 如果生成的执行计划中有工具调用序列
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # 将工具调用转换为候选计划格式
            candidate = {
                "plan_id": i,
                "description": f"Plan {i+1}: {plan_focuses[i]}",
                "tool_calls": []
            }
            
            for tool_call in response.tool_calls:
                candidate["tool_calls"].append({
                    "name": tool_call["name"],
                    "args": tool_call["args"]
                })
            
            candidates.append(candidate)
        else:
            # 如果没有工具调用，创建基于focus的默认计划
            default_quantities = [250, 200, 300]  # 不同的数量
            default_shipping = ["express", "priority", "standard"]  # 不同的运输方式
            default_priority = ["urgent", "high", "medium"]  # 不同的优先级
            
            candidate = {
                "plan_id": i,
                "description": f"Plan {i+1}: {plan_focuses[i]}",
                "tool_calls": [
                    {
                        "name": "manage_inventory",
                        "args": {"sku": "SKU-12345", "action": "reorder", "quantity": default_quantities[i]}
                    },
                    {
                        "name": "arrange_shipping",
                        "args": {"shipping_type": default_shipping[i], "priority": default_priority[i]}
                    },
                    {
                        "name": "send_logistics_response",
                        "args": {"message": f"Plan {i+1}: {plan_focuses[i]}. Reordered {default_quantities[i]} units with {default_shipping[i]} shipping."}
                    }
                ]
            }
            candidates.append(candidate)
    
    # 创建纯文本摘要
    summary_parts = [f"Generated {len(candidates)} diverse candidate plans:"]
    for i, candidate in enumerate(candidates):
        summary_parts.append(f"{i+1}.\n {candidate}")
    
    summary_msg = AIMessage(content="\n".join(summary_parts))
    
    state["candidates"] = candidates
    state["messages"] = history + [summary_msg]
    return state

# ==================== Critic 节点 ====================
def critic_node(state: AgentState) -> AgentState:
    """Evaluate candidates and execute the best one"""
    candidates = state["candidates"]
    history = state["messages"]
    
    # 评估执行计划
    critic_prompt = f"""Evaluate these 3 supply chain plans and select the BEST one.
    
    Each plan has different strategies and parameters. Consider:
    1. Appropriateness of reorder quantity (current stock: 50, backorder: 200)
    2. Cost-effectiveness of shipping strategy
    3. Risk mitigation in the approach
    4. Communication effectiveness
    
    PLANS:
    {candidates}
    
    Return a JSON object with:
    - "selected": index (0-2) of the best plan
    - "score": score from 1-10 (8+ to execute)
    - "reason": brief explanation of why this plan is best
    - "improvement_suggestions": optional suggestions for improvement
    
    Select the plan that best balances speed, cost, and risk for the situation."""
    
    response = llm_critic_with_tools.invoke([
        SystemMessage(content=critic_prompt),
        *history
    ])
    
    # 解析评估结果
    try:
        import re
        
        # 从响应中提取JSON
        content = response.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        
        if json_match:
            eval_result = json.loads(json_match.group())
        else:
            eval_result = {"selected": 0, "score": 5, "reason": "Could not parse evaluation"}
    except:
        eval_result = {"selected": 0, "score": 5, "reason": "Evaluation failed"}
    
    selected_idx = eval_result.get("selected", 0)
    score = eval_result.get("score", 0)
    
    # 执行评分大于8的计划
    if score >= 8 and selected_idx < len(candidates):
        # 执行选中的计划
        selected_plan = candidates[selected_idx]
        tool_messages = []
        
        print(f"\n[INFO] Executing Plan {selected_idx+1} (Score: {score})")
        print(f"Reason: {eval_result.get('reason', 'No reason provided')}")
        
        for i, tool_call in enumerate(selected_plan["tool_calls"]):
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\n  Step {i+1}: {tool_name}")
            print(f"    Args: {tool_args}")
            
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=f"call_{selected_idx}_{i}"
                    )
                )
                print(f"    Result: {result}")
        
        # 创建执行结果摘要
        result_summary = f"Executed Plan {selected_idx+1} (Score: {score}/10)\n"
        result_summary += f"Strategy: {selected_plan['description']}\n"
        result_summary += f"Reason: {eval_result.get('reason', 'No reason provided')}\n"
        result_summary += f"Tools executed: {len(tool_messages)}"
        
        result_msg = AIMessage(content=result_summary)
        
        state["messages"] = history + [response, result_msg] + tool_messages
    else:
        # 请求重新生成
        feedback = eval_result.get("improvement_suggestions", f"No plan scored 8+ (best score: {score}). Please regenerate with more diverse strategies.")
        feedback_msg = AIMessage(content=f"Regenerate plans: {feedback}")
        state["messages"] = history + [response, feedback_msg]
    
    return state

# ==================== 路由逻辑 ====================
def should_continue(state: AgentState) -> str:
    """Determine next step based on last message"""
    last_msg = state["messages"][-1]
    
    # 检查最后一条消息是否要求重新生成（包含单词regenerate）
    if isinstance(last_msg, AIMessage):
        content_lower = last_msg.content.lower()
        if "regenerate" in content_lower:
            # 检查重新生成次数，最多3次
            regen_count = sum(1 for msg in state["messages"] 
                            if isinstance(msg, AIMessage) and 
                            ("regenerate" in msg.content.lower()))
            # 重新生成计划的AIMessage中有2个regenerate，3*2=6,            
            return "actor" if regen_count < 6 else END 
    
    return END


# ==================== 构建图 ====================
def build_graph():
    g = StateGraph(AgentState)
    
    g.add_node("actor", actor_node)
    g.add_node("critic", critic_node)
    
    g.add_edge(START, "actor")
    g.add_edge("actor", "critic")
    
    g.add_conditional_edges(
        "critic",
        should_continue,
        {"actor": "actor", END: END}
    )
    
    return g.compile()

# ==================== 主程序 ====================
if __name__ == "__main__":
    graph = build_graph()
    
    # 测试输入
    initial_state = AgentState(
        messages=[HumanMessage(content="We're low on SKU-12345. Current stock: 50 units, backorder: 200 units. Need reorder strategy.")],
        candidates=None
    )
    
    result = graph.invoke(initial_state, config={"recursion_limit": 5})
    
    # 输出结果
    print("\n" + "="*60)
    print("Actor-Critic Agent Demo - Diverse Plans")
    print("="*60)
    
    for i, msg in enumerate(result["messages"]):
        msg_type = msg.type
        if isinstance(msg, AIMessage):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"\n{i}. AI [Tool Calls]:")
                for tc in msg.tool_calls:
                    print(f"   - {tc['name']}({tc['args']})")
            else:
                print(f"\n{i}. AI: {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"\n{i}. Tool Result: {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"\n{i}. Human: {msg.content}")
    
    print("\n" + "="*60)
    print("CANDIDATE PLANS SUMMARY:")
    print("="*60)
    
    if result.get('candidates'):
        for i, candidate in enumerate(result['candidates']):
            print(f"\nCandidate {i+1}: {candidate['description']}")
            for j, tool_call in enumerate(candidate.get('tool_calls', [])):
                args_str = ", ".join([f"{k}={v}" for k, v in tool_call['args'].items()])
                print(f"  {j+1}. {tool_call['name']}({args_str})")
    
    print("\n" + "="*60)

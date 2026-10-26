from __future__ import annotations
"""
soc_analyst_agent.py
LangGraph workflow for a Security Operations Center (SOC) Analyst agent,
handling threat investigation, incident response, log analysis, and security monitoring.
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

import os
import json
import operator
from typing import Annotated, Sequence, TypedDict, Optional

# LangChain 0.3.x Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# LangGraph 0.3.x Imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from traceloop.sdk import Traceloop
from src.common.observability.loki_logger import log_to_loki

# --- Tool Definitions ---

@tool
def lookup_threat_intel(indicator: str, type: str) -> str:
    """Look up threat intelligence for IP addresses, file hashes, URLs, and domains."""
    print(f"[TOOL] lookup_threat_intel(indicator={indicator}, type={type})")
    log_to_loki("tool.lookup_threat_intel", f"indicator={indicator}, type={type}")
    return "Threat Intel: IP is flagged as a known TOR exit node with high malicious score."

@tool
def query_logs(query: str, log_index: str) -> str:
    """Search and analyze security logs across authentication, endpoint, network, firewall, and DNS systems."""
    print(f"[TOOL] query_logs(query={query}, log_index={log_index})")
    log_to_loki("tool.query_logs", f"query={query}, log_index={log_index}")
    return "Log Query: Found 3 failed login attempts followed by one successful login from this IP."

@tool
def triage_incident(incident_id: str, decision: str, reason: str) -> str:
    """Classify security incidents as true positive, false positive, or escalate."""
    print(f"[TOOL] triage_incident(id={incident_id}, decision={decision})")
    log_to_loki("tool.triage_incident", f"incident_id={incident_id}, decision={decision}")
    return f"Incident {incident_id} triaged as {decision}."

@tool
def isolate_host(host_id: str, reason: str) -> str:
    """Isolate compromised hosts to prevent lateral movement."""
    print(f"[TOOL] isolate_host(host_id={host_id}, reason={reason})")
    log_to_loki("tool.isolate_host", f"host_id={host_id}, reason={reason}")
    return f"Host {host_id} has been isolated from the network."

@tool
def send_analyst_response(incident_id: str, message: str) -> str:
    """Send security analysis, incident updates, or recommendations to stakeholders."""
    print(f"[TOOL] send_analyst_response → {message}")
    log_to_loki("tool.send_analyst_response", f"incident_id={incident_id}, message={message}")
    return "Response sent to management."

TOOLS = [lookup_threat_intel, query_logs, triage_incident, isolate_host, send_analyst_response]

# --- Agent Logic ---

class AgentState(TypedDict):
    # Use Annotated with operator.add to manage message history append logic
    incident: Optional[dict]
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Initialize Model
Traceloop.init(disable_batch=True, app_name="soc_analyst_agent")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0).bind_tools(TOOLS)

def call_model(state: AgentState, config: RunnableConfig):
    """The node that decides what to do next."""
    # Handle missing or incomplete incident data gracefully
    incident = state.get("incident", {})
    if not incident:
        incident = {"incident_id": "UNKNOWN", "severity": "medium", "status": "investigating", "analyst": "SOC_EVAL"}
    
    incident_json = json.dumps(incident, ensure_ascii=False)
    
    system_prompt = (
        "You are an experienced Security Operations Center (SOC) analyst specializing in cybersecurity incident response.\n"
        "Your expertise covers:\n"
        "- Threat intelligence analysis and IOC research\n"
        "- Security log analysis and correlation across multiple systems\n"
        "- Incident triage and classification (true positive/false positive)\n"
        "- Malware analysis and threat hunting\n"
        "- Network security monitoring and anomaly detection\n"
        "- Incident containment and response coordination\n"
        "- SIEM/SOAR platform operations\n"
        "\n"
        "Your investigation methodology:\n"
        "  1) Analyze security alerts and gather initial indicators\n"
        "  2) Use lookup_threat_intel to research IPs, hashes, URLs, and domains\n"
        "  3) Use query_logs to search relevant log sources for evidence\n"
        "  4) Use triage_incident to classify findings as true/false positives\n"
        "  5) Use isolate_host when containment is needed to prevent spread\n"
        "  6) Follow up with send_analyst_response to document findings\n"
        "\n"
        "Always prioritize rapid threat containment and accurate incident classification.\n\n"
        f"INCIDENT: {incident_json}"
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages, config)
    
    # We return a list of messages which will be added to the state
    return {"messages": [response]}


# --- Graph Construction ---

def construct_graph():
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(TOOLS))

    # Define flow
    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools", # Map the string "tools" to your node named "tools"
            END: END    # Map the end condition to the END constant
        }
    )
    
    # After tools are executed, go back to agent to see if more work is needed
    workflow.add_edge("tools", "agent")

    return workflow.compile()

graph = construct_graph()

# --- Execution ---

if __name__ == "__main__":
    example_incident = {
        "incident_id": "INC-12345", 
        "severity": "high", 
        "type": "suspicious_login",
        "analyst": "J.Smith"
    }
    
    initial_input = {
        "incident": example_incident,
        "messages": [
            HumanMessage(content="Alert: Suspicious login from IP 203.0.113.45. Investigate and contain if necessary.")
        ]
    }

    # Stream the output to see the steps
    for event in graph.stream(initial_input, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            print(f"\n[{type(last_msg).__name__}]: {last_msg.content}")


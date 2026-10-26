import os
from dotenv import load_dotenv
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# 1. State schema
class SupportState(TypedDict):
    user_message: str
    user_id: str
    issue_type: str | None          # billing / technical
    billing_type: str | None        # invoice / refund
    technical_type: str | None      # login / performance
    billing_result: str | None
    technical_result: str | None
    final_response: str | None

# 2. LLM init
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 3. Node implementations
# Issue router
def categorize_issue(state: SupportState) -> dict:
    msg = state["user_message"]
    prompt = f"""
        Classify this message as 'billing' or 'technical',
        just return the class of it, not anything else.
        Message:
        {msg}
        """
    resp = llm.invoke(prompt).content.lower().strip()
    return {"issue_type": resp}

# Billing router
def billing_issue_router(state: SupportState) -> dict:
    msg = state["user_message"].lower()
    b_type = "invoice" if "invoice" in msg else "refund"
    return {"billing_type": b_type}

def handle_invoice(state: SupportState) -> dict:
    return {"billing_result": f"Invoice details for {state['user_id']}"}

def handle_refund(state: SupportState) -> dict:
    return {"billing_result": "Refund process initiated"}

# Technical router
def technical_issue_router(state: SupportState) -> dict:
    msg = state["user_message"].lower()
    t_type = "login" if "login" in msg else "performance"
    return {"technical_type": t_type}

def handle_login(state: SupportState) -> dict:
    return {"technical_result": "Password reset link sent"}

def handle_performance(state: SupportState) -> dict:
    return {"technical_result": "Performance metrics analyzed"}

# Final summarization
def summarize_response(state: SupportState) -> dict:
    detail = state.get("billing_result") or state.get("technical_result") or ""
    summary = llm.invoke(f"Write a concise customer reply based on: {detail}").content.strip()
    return {"final_response": summary}

# 4. Build graph
graph = StateGraph(SupportState)
# --- Add nodes ---
graph.add_node("categorize_issue", categorize_issue)
graph.add_node("billing_issue_router", billing_issue_router)
graph.add_node("technical_issue_router", technical_issue_router)
graph.add_node("handle_invoice", handle_invoice)
graph.add_node("handle_refund", handle_refund)
graph.add_node("handle_login", handle_login)
graph.add_node("handle_performance", handle_performance)
graph.add_node("summarize_response", summarize_response)

# Start → categorize_issue
graph.add_edge(START, "categorize_issue")

# Top-level routing
graph.add_conditional_edges(
    "categorize_issue",
    lambda s: s["issue_type"],
    {
        "billing": "billing_issue_router",
        "technical": "technical_issue_router",
    }
)

# Billing router
graph.add_conditional_edges(
    "billing_issue_router",
    lambda s: s["billing_type"],
    {
        "invoice": "handle_invoice",
        "refund": "handle_refund",
    }
)

# Technical router
graph.add_conditional_edges(
    "technical_issue_router",
    lambda s: s["technical_type"],
    {
        "login": "handle_login",
        "performance": "handle_performance",
    }
)

# Merge all handlers → summarize_response → END
for node in ["handle_invoice", "handle_refund", "handle_login", "handle_performance"]:
    graph.add_edge(node, "summarize_response")

graph.add_edge("summarize_response", END)

# 5. Compile & run
app = graph.compile()
# Initial state
initial_state: SupportState = {
    "user_message": "Hi, I need help with my invoice and possibly a refund.",
    "user_id": "U1234",
    "issue_type": None,
    "billing_type": None,
    "technical_type": None,
    "billing_result": None,
    "technical_result": None,
    "final_response": None,
}

result = app.invoke(initial_state)
print(result["final_response"])

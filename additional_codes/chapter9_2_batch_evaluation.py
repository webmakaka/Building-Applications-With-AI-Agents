from __future__ import annotations
"""
batch_evaluation.py
Handles multi-turn conversations and surfaces diagnostics, including parameter accuracy.
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

import builtins
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator
from typing import Dict, List, Annotated, Sequence, Optional
builtins.Annotated = Annotated
builtins.BaseMessage = BaseMessage
builtins.operator = operator
builtins.Sequence = Sequence
builtins.List = List
builtins.Optional = Optional
builtins.add_messages = add_messages
import argparse
import importlib.util
import json
import pathlib
import statistics as stats
from src.common.observability.loki_logger import log_to_loki
from src.common.evaluation.metrics import phrase_recall, tool_metrics, param_accuracy, task_success
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# laod conversation context from sample data
def to_lc_message(turn: dict):
    role = (turn.get("role") or "").lower()
    txt = turn.get("content", "")
    if role in {"customer", "user", "human"}:
        return HumanMessage(content=txt)
    if role in {"assistant", "agent", "ai"}:
        return AIMessage(content=txt)
    if role == "system":
        return SystemMessage(content=txt)
    if role == "tool":
        return ToolMessage(content=txt, tool_call_id=turn.get("tool_call_id", "unknown"))
    return HumanMessage(content=txt)

def load_graph(path: str):
    spec = importlib.util.spec_from_file_location("user_graph", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)                 # type: ignore
    if hasattr(mod, "graph"):
        return mod.graph
    if hasattr(mod, "construct_graph"):
        return mod.construct_graph()  # type: ignore
    raise AttributeError(f"{path} exposes neither `graph` nor `construct_graph()`")

def parse_weights(pairs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for pair in pairs:
        if "=" in pair:
            k, v = pair.split("=", 1)
            try:
                out[k.strip().lower()] = float(v)
            except ValueError:
                pass
    return out

def evaluate_single_instance(raw: str, graph) -> Optional[Dict[str, float]]:
    """
    Parse one JSONL line, run the graph, compute metrics, or return None if skipped.
    """
    if not raw.strip():
        return None
    try:
        ex = json.loads(raw)
        print(f"[DEBUG] Parsed JSON keys: {list(ex.keys())}")

        if "input" in ex and "expected_function_call" in ex:
            messages = [to_lc_message(t) for t in ex["input"]]
            expected_call = ex["expected_function_call"]
            exp_final = {
                "tool_calls": [{
                    "tool": expected_call["name"],
                    "params": expected_call["arguments"]
                }],
                "customer_msg_contains": []
            }

            initial_state = {"messages": messages}
            initial_state["order"] = ex["order"]
        else:
            print(f"[SKIPPED] Unrecognized format: {list(ex.keys())}")
            return None

        result = graph.invoke(initial_state)

        final_reply = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                calls = msg.additional_kwargs.get("tool_calls", [])
                if not calls:
                    final_reply = msg.content or ""
                    break

        pred_tool_names: List[str] = []
        pred_call_objs: List[dict] = []
        for m in result["messages"]:
            if not isinstance(m, AIMessage):
                continue
            for tc in m.additional_kwargs.get("tool_calls", []):
                if "function" in tc:
                    name = tc["function"]["name"]
                    params = json.loads(tc["function"]["arguments"])
                else:
                    name = tc.get("name")
                    params = tc.get("args", {})
                pred_tool_names.append(name)
                pred_call_objs.append({"tool": name, "params": params})

        tm = tool_metrics(pred_tool_names, exp_final.get("tool_calls", []))
        
        return {
            "phrase_recall": phrase_recall(final_reply, exp_final.get("customer_msg_contains", [])),
            "tool_recall": tm["tool_recall"],
            "tool_precision": tm["tool_precision"],
            "param_accuracy": param_accuracy(pred_call_objs, exp_final.get("tool_calls", [])),
            "task_success": task_success(final_reply, pred_tool_names, exp_final),
        }

    except Exception as e:
        print(f"[SKIPPED] example failed with error: {e!r}")
        import traceback
        traceback.print_exc()
        return None

def run_evaluation(
    graph, args, weights: Dict[str, float],
    metrics: Dict[str, List[float]], verbose: bool = False
):
    i = 0
    for raw in pathlib.Path(args.dataset).read_text().splitlines():
        i = i+1
        # if i>4:
        #     break
        print(f"\nEvaluating sample {i}...\n")
        result = evaluate_single_instance(raw, graph)
        if result is None:
            try:
                log_to_loki("batch_eval", "[SKIPPED] evaluation failed or malformed entry.")
            except:
                print("[SKIPPED] evaluation failed or malformed entry.")
            continue
        else:
            try:
                log_to_loki("batch_eval", f"[METRICS] {json.dumps(result)}")
            except:
                print(f"[METRICS] {json.dumps(result)}")
        # Append to global metrics
        for k, v in result.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)

    print("\n=== Aggregate scores ===")
    if any(len(vals) > 0 for vals in metrics.values()):
        for k, vals in metrics.items():
            if len(vals) > 0:
                print(f"{k:15s}: {stats.mean(vals):.3f} (n={len(vals)})")
            else:
                print(f"{k:15s}: No data")

        active_metrics = {k: vals for k, vals in metrics.items() if len(vals) > 0}
        if active_metrics:
            active = {m: weights.get(m, 1.0) for m in active_metrics}
            total = sum(active.values())
            overall = sum(stats.mean(active_metrics[m]) * w for m, w in active.items()) / total
            print(f"\nWeighted overall score: {overall:.3f}")
    else:
        print("No successful evaluations completed.")
    try:
        if any(len(vals) > 0 for vals in metrics.values()):
            log_lines = [f"{k}: {stats.mean(vals):.3f}" for k, vals in metrics.items() if len(vals) > 0]
            summary = "Evaluation Summary - " + " | ".join(log_lines)
            log_to_loki("batch_eval", summary)
    except:
        pass  # Don't crash if logging fails

def main():
    graph_py = "chapter9_1_customer_support_agent.py"
    dataset = "/home/ubuntu/Building_Applications_with_AI_Agents/src/common/evaluation/scenarios/ecommerce_customer_support_evaluation_set.jsonl"
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_py", type=str, default=graph_py)
    ap.add_argument("--dataset", type=str, default=dataset)
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--verbose", default=False)
    args = ap.parse_args()

    graph = load_graph(args.graph_py)
    weights = parse_weights(args.weights) or {"task_success": 1.0}

    metrics: Dict[str, List[float]] = {
        "task_success": [],
        "phrase_recall": [],
        "tool_recall": [],
        "tool_precision": [],
        "param_accuracy": [],
    }

    run_evaluation(graph, args, weights, metrics, verbose=args.verbose)


if __name__ == "__main__":
    '''
Example usage
python -m src.common.evaluation.batch_evaluation \
  --dataset src/common/evaluation/scenarios/ecommerce_customer_support_evaluation_set.json \
  --graph_py src/frameworks/langgraph_agents/ecommerce_customer_support/customer_support_agent.py
    '''
  
    main()


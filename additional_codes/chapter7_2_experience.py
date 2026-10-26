import os
from dotenv import load_dotenv
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()


from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

# =========================
# Initialize LLM
# =========================
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# =========================
# Node function (LangGraph 0.3.x compatible)
# =========================
def call_model(state: MessagesState):
    """
    LangGraph 0.3.x:
    - state["messages"] is a list
    - return value must be a partial state
    """
    response: AIMessage = llm.invoke(state["messages"])
    return {"messages": [response]}


# =========================
# InsightAgent
# =========================
class InsightAgent:
    def __init__(self):
        self.insights = []
        self.promoted_insights = []
        self.demoted_insights = []
        self.reflections = []

        # Build graphs once
        self._insight_graph = self._build_graph()
        self._reflection_graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(MessagesState)
        builder.add_node("llm", call_model)
        builder.add_edge(START, "llm")
        builder.add_edge("llm", END)
        return builder.compile()

    # =========================
    # Insight generation
    # =========================
    def generate_insight(self, observation: str):
        messages = [
            HumanMessage(
                content=f"Generate an insightful analysis based on the following observation:\n{observation}"
            )
        ]

        result = self._insight_graph.invoke({"messages": messages})

        insight = result["messages"][-1].content
        self.insights.append(insight)

        print(f"Generated: {insight}")
        return insight

    # =========================
    # Insight lifecycle ops
    # =========================
    def promote_insight(self, insight: str):
        if insight in self.insights:
            self.insights.remove(insight)
            self.promoted_insights.append(insight)
            print(f"Promoted: {insight}")
        else:
            print(f"Insight '{insight}' not found in insights.")

    def demote_insight(self, insight: str):
        if insight in self.promoted_insights:
            self.promoted_insights.remove(insight)
            self.demoted_insights.append(insight)
            print(f"Demoted: {insight}")
        else:
            print(f"Insight '{insight}' not found in promoted insights.")

    def edit_insight(self, old_insight: str, new_insight: str):
        for bucket in (self.insights, self.promoted_insights, self.demoted_insights):
            if old_insight in bucket:
                idx = bucket.index(old_insight)
                bucket[idx] = new_insight
                print(f"Edited: '{old_insight}' → '{new_insight}'")
                return

        print(f"Insight '{old_insight}' not found.")

    def show_insights(self):
        print("\nCurrent Insights:")
        print(f"Insights: {self.insights}")
        print(f"Promoted Insights: {self.promoted_insights}")
        print(f"Demoted Insights: {self.demoted_insights}")

    # =========================
    # Reflection
    # =========================
    # def reflect(self, reflexion_prompt: str):
    #     messages = [HumanMessage(content=reflexion_prompt)]
    # 
    #     result = self._reflection_graph.invoke({"messages": messages})
    #     reflection = result["messages"][-1].content
    # 
    #     self.reflections.append(reflection)
    #     print(f"Reflection: {reflection}")
    #     return reflection


    # =========================
    # Reflection
    # =========================
    def reflect(self, reflexion_prompt: str):
        """
        Reflect based on the current state of insights.
        This includes generated, promoted, demoted, and edited insights.
        """
    
        insight_context = f"""
    Current Insight State:
    
    1. Active Insights:
    {chr(10).join(f"- {i}" for i in self.insights) or "- None"}
    
    2. Promoted Insights:
    {chr(10).join(f"- {i}" for i in self.promoted_insights) or "- None"}
    
    3. Demoted Insights:
    {chr(10).join(f"- {i}" for i in self.demoted_insights) or "- None"}
    """
    
        messages = [
            HumanMessage(
                content=(
                    insight_context
                    + "\n\n"
                    + reflexion_prompt
                )
            )
        ]
    
        result = self._reflection_graph.invoke({"messages": messages})
        reflection = result["messages"][-1].content
    
        self.reflections.append(reflection)
        print(f"Reflection: {reflection}")
        return reflection



# =========================
# Example usage
# =========================
if __name__ == "__main__":
    agent = InsightAgent()

    insight1 = agent.generate_insight(
        "The sales have increased by 20% in the last quarter."
    )

    insight2 = agent.generate_insight(
        "Customer complaints have decreased significantly."
    )

    agent.promote_insight(insight1)
    agent.demote_insight(insight1)

    agent.edit_insight(
        insight2,
        "Refined insight: Customer satisfaction is improving due to reduced complaints."
    )

    agent.show_insights()

    agent.reflect(
        "Reflect on the current insights and provide suggestions for improvement on business."
    )

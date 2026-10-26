import os
from dotenv import load_dotenv
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage

# --- 环境变量配置保持不变 ---
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()
# --- 环境变量配置结束 ---

# 1. 定义图的状态
class GraphState(TypedDict):
    """
    表示图状态的 TypedDict。
    """
    messages: Annotated[List[BaseMessage], lambda x, y: x + y] # 合并消息列表

llm = ChatOpenAI(model="gpt-4o")

reflections = []

# 2. 节点函数签名
def call_model(state: GraphState) -> dict:
    """
    调用语言模型生成反射计划。
    """
    print("---CALL MODEL/REFLEXION---")
    response = llm.invoke(state["messages"])
    return {"messages": [response]} 

# 3. 反射提示保持不变
reflexion_prompt = """You will be given the history of a past experience in which you were 
placed in an environment and given a task to complete. You were unsuccessful in 
completing the task. Do not summarize your environment, but rather think about 
the strategy and path you took to attempt to complete the task. 
Devise a concise, new plan of action that accounts for your mistake with reference 
to specific actions that you should have taken. For example, if you tried A and B but forgot C, 
then devise a plan to achieve C with environment-specific actions. You will need this 
later when you are solving the same task. Give your plan after "Plan". 

Instruction:
i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars
[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Action: think[The first two items are not what I want. The third item is the dairy free and apple variety pack of chips I want.]
Observation: OK.

Action: click[B008D2X8C4]
Observation:
[Back to Search]
[< Prev]
flavor name [dill and sour cream][garlic & parmesan][light sea salt][margherita pizza][thai chili lime][variety pack]
size [0.8 ounce (pack of 24)][4 ounce (pack of 12)]
Price: $100.0
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[The item has options 'variety pack' and '0.8 ounce (pack of 24)' and seems good to buy.]
Observation: OK.

Action: click[variety pack]
Observation: You have clicked variety pack.

Action: click[0.8 ounce (pack of 24)]
Observation: You have clicked 0.8 ounce (pack of 24).

Action: click[Buy Now]

STATUS: FAIL

Plan:
"""


# 4. 初始化 StateGraph
builder = StateGraph(GraphState) 

# 5. 添加节点
builder.add_node("reflexion", call_model)

# 6. 使用 START 常量建立边 (按照您的要求)
# 导入 START 并将它作为源节点添加到边中。
builder.add_edge(START, "reflexion") 
builder.add_edge("reflexion", END)   

# 7. 编译图
graph = builder.compile()

# 8. 运行图
result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content=reflexion_prompt
            )
        ]
    }
)

reflections.append(result)

# 9. 输出结果
print("---FINAL STATE---")
print(result["messages"][-1].content)


#-------------------------------------------------------------------------------
# def get_completion(prompt: str) -> str:
#     # Wraps our `call_model` helper for one‐off text completions 
#     result = llm.invoke([{"role":"user","content":prompt}]) 
#     return result[0].content
# 
# def _generate_reflection_query(trial_log: str, recent_reflections: List[str]): 
#     history = "\n\n".join(recent_reflections)
#     return f'''{history}
#         {trial_log}
#         Based on the above, what plan would you follow next? Plan:'''
# 
# def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]): 
#     """Updates the given env_config with the appropriate reflections.""" 
#     with open(trial_log_path, 'r') as f:
#         full_log: str = f.read()
#     env_logs: List[str] = full_log.split('#####\n\n#####')
#     assert len(env_logs) == len(env_configs), print(f'bad: {env_logs}') 
#     for i, env in enumerate(env_configs):
#         # if unsolved, get reflection and update env config 
#         if not env['is_success'] and not env['skip']:
#             if len(env['memory']) > 3:
#                 memory: List[str] = env['memory'][-3:] 
#             else:
#                 memory: List[str] = env['memory']
#             reflection_query = _generate_reflection_query(env_logs[i], memory) 
#             reflection = get_completion(reflection_query) 
#             env_configs[i]['memory'] += [reflection]
# 
# update_memory(trial_log_path, env_configs)
# 

# ==============================================================================
# 以下是 Memory 更新逻辑（重点修复与补全部分）
# ==============================================================================

trial_log_path = "trial_log_example.txt"

with open(trial_log_path, "w", encoding="utf-8") as f:
    f.write(reflexion_prompt)


env_configs = [
    {
        "env_id": "shopping_env_1",
        "is_success": False,
        "skip": False,
        "memory": []   # 初始为空
    }
]


from typing import List, Dict, Any

def update_memory(
    trial_log_path: str,
    env_configs: List[Dict[str, Any]],
    agent_reflections: List[str]
) -> None:
    """
    使用 LangGraph Agent 已生成的 reflection 更新 memory
    """

    with open(trial_log_path, "r", encoding="utf-8") as f:
        full_log = f.read()

    env_logs = full_log.split("#####\n\n#####")

    assert len(env_logs) == len(env_configs) == len(agent_reflections)

    for i, env in enumerate(env_configs):
        if env.get("is_success") or env.get("skip"):
            continue

        env.setdefault("memory", []).append(agent_reflections[i])


# LangGraph Agent 的反思输出
agent_reflections = [result["messages"][-1].content]

# 用 Agent 输出更新 memory
update_memory(
    trial_log_path=trial_log_path,
    env_configs=env_configs,
    agent_reflections=agent_reflections
)

# 查看结果
print("\n--- UPDATED MEMORY ---")
for m in env_configs[0]["memory"]:
    print(m)

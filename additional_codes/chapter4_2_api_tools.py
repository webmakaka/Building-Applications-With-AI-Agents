# ------------------------------------------------------------------------------------
# !pip install wikipedia
import os
os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"

from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage
import json

# --- Tool setup ---
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# --- LLM setup ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools([tool])

# --- User query ---
messages = [
    HumanMessage("What was the most impressive thing about Buzz Aldrin?")
]

# --- Step 1: model produces a tool call ---
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

# --- Step 2: handle tool calls ---
for tool_call in ai_msg.additional_kwargs.get("tool_calls", []):
    # Extract name, args, id
    tool_call_id = tool_call["id"]
    name = tool_call["function"]["name"]
    args_str = tool_call["function"]["arguments"]
    args = json.loads(args_str)

    # Execute tool
    result = tool.invoke(list(args.values())[0])

    # Create ToolMessage (critical fix)
    tool_msg = ToolMessage(
        content=str(result),
        tool_call_id=tool_call_id,
        name=name
    )

    print("Tool:", name)
    print("Args:", args)
    print("Result:", result)
    print()

    messages.append(tool_msg)

# --- Step 3: final LLM response ---
final_response = llm_with_tools.invoke(messages)
print("\nFinal answer:\n", final_response.content)


#-----------------------------------------------------------------------------------------
# 使用Ashare读取国内的股市行情，单文件Python库，不需要安装，直接下载使用，要补装pandas库
# https://github.com/mpquant/Ashare/blob/main/Ashare.py
# !pip inistall pandas
import os
os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"

import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

# 引入 Ashare
from Ashare import get_price  # 假设你已把 Ashare.py 放在同目录，或已正确安装

@tool
def get_stock_price(ticker: str) -> float:
    """
    Get the latest close price for the given A‑share ticker.
    ticker: e.g. '600519' 或 '600519.XSHG' 或 'sh600519'
    """
    # 调用 Ashare.get_price 获取最近一期日线数据，count=1
    df = get_price(ticker+".XSHG", frequency='1d', count=1)
    print(df)
    # 假设 dataFrame 有列 'close'
    price = df.iloc[0]['close']
    return float(price)

# Initialize LLM + tools
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools([get_stock_price])

messages = [HumanMessage("What is the stock price of 600519 (贵州茅台)?")]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

print("LLM initial:", ai_msg)

for tool_call in ai_msg.additional_kwargs.get("tool_calls", []):
    tool_call_id = tool_call["id"]
    name = tool_call["function"]["name"]
    args_str = tool_call["function"]["arguments"]
    args = json.loads(args_str)

    result = get_stock_price.invoke(args)

    tool_msg = ToolMessage(
        content=str(result),
        tool_call_id=tool_call_id,
        name=name
    )

    print("Tool:", name, "Args:", args, "Result:", result)
    messages.append(tool_msg)

final_response = llm_with_tools.invoke(messages)
print("Final answer:", final_response.content)

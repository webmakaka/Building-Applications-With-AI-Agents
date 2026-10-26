# !pip install faiss-cpu
import os
from dotenv import load_dotenv
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
load_dotenv()

import requests
import logging
import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np

# -------------------------------
# 1. Setup and Configuration
# -------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WOLFRAM_ALPHA_APP_ID = os.getenv("WOLFRAM_ALPHA_APP_ID")
ZAP_ID = os.getenv("ZAP_ID")  
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL")


# Validate that all necessary API keys are present
if not all([OPENAI_API_KEY, WOLFRAM_ALPHA_APP_ID, ZAP_ID, SLACK_BOT_TOKEN]):
    logging.error("One or more API keys are missing. Please set them as environment variables.")
    exit(1)

# -------------------------------
# 2. Define Tools
# -------------------------------

@tool
def query_wolfram_alpha(expression: str) -> str:
    """
    Query Wolfram Alpha to compute mathematical expressions or retrieve information.

    Args:
        expression (str): The mathematical expression or query to evaluate.

    Returns:
        str: The result of the computation or the retrieved information.

    Raises:
        ValueError: If the API request fails or returns an error.
    """
    api_url = f"https://api.wolframalpha.com/v1/result?i={requests.utils.quote(expression)}&appid={WOLFRAM_ALPHA_APP_ID}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.text
        else:
            raise ValueError(f"Wolfram Alpha API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to query Wolfram Alpha: {e}")

@tool
def trigger_zapier_webhook(zap_id: str, payload: dict) -> str:
    """
    Trigger a Zapier webhook to execute a predefined Zap.

    Args:
        zap_id (str): The unique identifier for the Zap to be triggered.
        payload (dict): The data to send to the Zapier webhook.

    Returns:
        str: Confirmation message upon successful triggering of the Zap.

    Raises:
        ValueError: If the API request fails or returns an error.
    """
    zapier_webhook_url = f"https://hooks.zapier.com/hooks/catch/{zap_id}/"
    try:
        response = requests.post(zapier_webhook_url, json=payload)
        if response.status_code == 200:
            return f"Zapier webhook '{zap_id}' successfully triggered."
        else:
            raise ValueError(f"Zapier API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to trigger Zapier webhook '{zap_id}': {e}")

@tool
def send_slack_message(channel: str, message: str) -> str:
    """
    Send a message to a specified Slack channel.

    Args:
        channel (str): The Slack channel ID or name where the message will be sent.
        message (str): The content of the message to send.

    Returns:
        str: Confirmation message upon successful sending of the Slack message.

    Raises:
        ValueError: If the API request fails or returns an error.
    """
    api_url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "channel": channel,
        "text": message
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()
        if response.status_code == 200 and response_data.get("ok"):
            return f"Message successfully sent to Slack channel '{channel}'."
        else:
            error_msg = response_data.get("error", "Unknown error")
            raise ValueError(f"Slack API Error: {error_msg}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to send message to Slack channel '{channel}': {e}")


# Initialize the LLM with GPT-4o and bind the tools 
llm = ChatOpenAI(model_name="gpt-4o") 
llm_with_tools = llm.bind_tools([query_wolfram_alpha,trigger_zapier_webhook,send_slack_message])
messages = [HumanMessage("What is the answer of (1+2)*3?")]
ai_msg = llm_with_tools.invoke(messages) 
messages.append(ai_msg)


# 处理工具调用
tool_dict = {"query_wolfram_alpha": query_wolfram_alpha, 
    "trigger_zapier_webhook": trigger_zapier_webhook, 
    "send_slack_message": send_slack_message}
tool_calls = ai_msg.additional_kwargs['tool_calls']
for tool_call in tool_calls:
    # 提取工具调用的函数名和参数
    function_name = tool_call['function']['name']
    function_args = tool_call['function']['arguments']
    selected_tool = tool_dict[function_name]
    # 解析参数（假设参数是 JSON 格式）
    parsed_args = json.loads(function_args)
    # 调用工具并传入解析后的参数
    result = selected_tool.invoke(parsed_args)
    print(f'{function_name} {function_args} {result}')
    # 创建 ToolMessage
    tool_msg = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"]
    )
    messages.append(tool_msg)
# 得到LLM的最后答复
final_response = llm_with_tools.invoke(messages) 
print(final_response.content)






result = query_wolfram_alpha.invoke(
    {"expression":"(1+2)*3"}
)

print(result)



result = trigger_zapier_webhook.invoke(
    {"zap_id":"25633083/uftgukz",
    "payload":{"name": "Jean"}}
)

print(result)


result = send_slack_message.invoke(
    {"channel":"C0A24HZ9U2J",
    "message":"Hello again from the tool!"}
)

print(result)

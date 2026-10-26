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



# ------------------------------------------------------------------------------
# Define tool groups with descriptions
tool_groups = {
    "Computation": {
        "description": "Tools related to mathematical computations and data analysis.",
        "tools": []
    },
    "Automation": {
        "description": "Tools that automate workflows and integrate different services.",
        "tools": []
    },
    "Communication": {
        "description": "Tools that facilitate communication and messaging.",
        "tools": []
    }
}

# Assign tools to their respective groups
tool_groups["Computation"]["tools"].append(query_wolfram_alpha)
tool_groups["Automation"]["tools"].append(trigger_zapier_webhook)
tool_groups["Communication"]["tools"].append(send_slack_message)

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 原始版本----------------------------------------------------------------------
# -------------------------------
# LLM-Based Hierarchical Skill Selection
# -------------------------------
def select_group_llm(query: str) -> str:
    """
    Use the LLM to determine the most appropriate skill group based on the user's query.
    
    Args:
        query (str): The user's input query.
        
    Returns:
        str: The name of the selected group.
    """
    prompt = f"Select the most appropriate skill group for the following query: '{query}'.\nOptions are: Computation, Automation, Communication."
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

def select_tool_llm(query: str, group_name: str) -> str:
    """
    Use the LLM to determine the most appropriate tool within a group based on the user's query.
    
    Args:
        query (str): The user's input query.
        group_name (str): The name of the selected skill group.
        
    Returns:
        str: The name of the selected tool function.
    """
    prompt = f"Based on the query: '{query}', select the most appropriate tool from the group '{group_name}'."
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

# ------------------------------------------------------------------------------
# Example user query
user_query = "Solve this equation: 2x + 3 = 7"

# Step 1: Select the most relevant skill group using LLM
selected_group_name = select_group_llm(user_query)
if not selected_group_name:
    print("No relevant skill group found for your query.")
else:
    logging.info(f"Selected Group: {selected_group_name}")
    print(f"Selected Skill Group: {selected_group_name}")

    # Step 2: Select the most relevant tool within the group using LLM
    selected_tool_name = select_tool_llm(user_query, selected_group_name)
    selected_tool = globals().get(selected_tool_name, None)
    
    if not selected_tool:
        print("No relevant tool found within the selected group.")
    else:
        logging.info(f"Selected Tool: {selected_tool.__name__}")
        print(f"Selected Tool: {selected_tool.__name__}")
        
        # Prepare arguments based on the tool
        args = {}
        if selected_tool == query_wolfram_alpha:
            # Assume the entire query is the expression
            args["expression"] = user_query
        elif selected_tool == trigger_zapier_webhook:
            # Use placeholders for demo
            args["zap_id"] = "123456"
            args["payload"] = {"message": user_query}
        elif selected_tool == send_slack_message:
            # Use placeholders for demo
            args["channel"] = "#general"
            args["message"] = user_query
        else:
            print("Selected tool is not recognized.")
        
        # Invoke the selected tool
        try:
            tool_result = selected_tool.invoke(args)
            print(f"Tool '{selected_tool.__name__}' Result: {tool_result}")
        except ValueError as e:
            print(f"Error: {e}")



# 改进版本----------------------------------------------------------------------
def select_group_llm(query: str) -> str:
    """
    Use the LLM to determine the most appropriate skill group based on the user's query.
    Dynamically reads group names from tool_groups.
    """
    group_names = list(tool_groups.keys())
    group_list_str = ", ".join(group_names)

    prompt = f"""
        You are a routing module. Your task is to select a tool group.

        User query: '{query}'

        Available tool groups:
        {json.dumps({k:v["description"] for k,v in tool_groups.items()}, indent=2)}

        Respond ONLY with one exact group name from this list:
        {group_list_str}
        """
    response = llm([HumanMessage(content=prompt)])
    selected = response.content.strip()

    # Ensure LLM output is valid
    if selected not in group_names:
        logging.warning(f"LLM returned invalid group '{selected}', fallback to heuristic.")
        # fallback (simple heuristic)
        if "slack" in query.lower():
            return "Communication"
        if "zap" in query.lower() or "automation" in query.lower():
            return "Automation"
        return "Computation"

    return selected


def select_tool_llm(query: str, group_name: str) -> AIMessage:
    """LLM selects tool from the group using built-in tool calling."""
    tools = tool_groups[group_name]["tools"]
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(query)
    return response

# Example user query
user_query = "Solve this equation: 2x + 3 = 7"
# Step 1: select group
selected_group = select_group_llm(user_query)
print("Selected group:", selected_group)

tool_msg = select_tool_llm(user_query, selected_group)

tool_call = tool_msg.tool_calls[0]
tool_name = tool_call["name"]
tool_args = tool_call["args"]

# 找到工具对象
tool_obj = next(t for t in tool_groups[selected_group]["tools"] if t.name == tool_name)

# 执行工具
result = tool_obj.invoke(tool_args)
print("Tool result:", result)


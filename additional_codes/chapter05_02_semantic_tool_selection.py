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


# -------------------------------
# 3. Embed Tool Descriptions
# -------------------------------

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# Define tool descriptions
tool_descriptions = {
    "query_wolfram_alpha": "Use Wolfram Alpha to compute mathematical expressions or retrieve information.",
    "trigger_zapier_webhook": "Trigger a Zapier webhook to execute predefined automated workflows.",
    "send_slack_message": "Send messages to specific Slack channels to communicate with team members."
}

# Create embeddings for each tool description
tool_embeddings = []
tool_names = []
for tool_name, description in tool_descriptions.items():
    embedding = embeddings.embed_query(description)
    tool_embeddings.append(embedding)
    tool_names.append(tool_name)

# -------------------------------
# 4. Initialize Vector Store
# -------------------------------

# Initialize FAISS vector store
dimension = len(tool_embeddings[0])
# Convert list to FAISS-compatible format
tool_embeddings_np = np.array(tool_embeddings).astype('float32')
index = faiss.IndexFlatL2(dimension)
# Normalize embeddings for cosine similarity
faiss.normalize_L2(tool_embeddings_np)  
index.add(tool_embeddings_np)

# Map index to tool functions
index_to_tool = {
    0: query_wolfram_alpha,
    1: trigger_zapier_webhook,
    2: send_slack_message
}

# -------------------------------
# 5. Select Appropriate Tool
# -------------------------------

def select_tool(query: str, top_k: int = 1) -> list:
    """
    Select the most relevant tool(s) based on the user's query using vector-based retrieval.

    Args:
        query (str): The user's input query.
        top_k (int): Number of top tools to retrieve.

    Returns:
        list: List of selected tool functions.
    """
    query_embedding = embeddings.embed_query(query)
    query_embedding_np = np.array(query_embedding, dtype='float32').reshape(1, -1)

    faiss.normalize_L2(query_embedding_np)
    D, I = index.search(query_embedding_np, top_k)
    selected_tools = [index_to_tool[idx] for idx in I[0] if idx in index_to_tool]
    return selected_tools

# 调用LLM确定工具调用的参数
# top_k=1 -----------------------------------------------------------------------
def determine_parameters(query: str, tool, tool_llm) -> dict:
    """
    Let the LLM extract parameters using a standard tool call format.
    Now the LLM is bound to the specific vector-selected tool.
    """
    tool_name = tool.name

    messages = [
        SystemMessage(content=f"""
            You are a tool parameter extraction assistant.
            You MUST call the tool '{tool_name}' with correct JSON arguments.
            Return ONLY a standard LangChain tool_call. Do NOT answer normally.
        """),
        HumanMessage(content=f"User query: {query}\nExtract arguments for tool: {tool_name}")
    ]

    # Use the LLM that already has tool binding
    response = tool_llm.invoke(messages)
    print("LLM raw response:", response)

    # Parse tool call
    tool_calls = response.additional_kwargs.get("tool_calls", None)
    if not tool_calls:
        raise ValueError("LLM did not return a tool call. Check prompts.")

    # Example call is:
    #   {"id": "...", "function": {"name": "query_wolfram_alpha", "arguments": "{...}" }}
    call = tool_calls[0]
    args_json = call["function"]["arguments"]  # raw JSON string
    args = json.loads(args_json)

    return args


user_query = "Solve this equation: 2x + 3 = 7"

# 1. Vector search → pick tool
selected_tools = select_tool(user_query, top_k=1)
selected_tool = selected_tools[0]
tool_name = selected_tool.name

print("Selected tool:", tool_name)

# 2. Bind only this tool to the LLM → LLM 不能调用其他工具
llm = ChatOpenAI(model_name="gpt-4o") 
tool_llm = llm.bind_tools([selected_tool])

# 3. Extract parameters with standard tool call
args = determine_parameters(user_query, selected_tool, tool_llm)

# 4. Invoke the selected tool normally
tool_result = selected_tool.invoke(args)

print("Tool result:", tool_result)

# top_k>1 ----------------------------------------------------------------------
def determine_tool_call(query: str, candidate_tools: list, llm):
    """
    Given the top_k vector-retrieved tools, let LLM choose the correct tool
    and generate a proper tool_call with JSON arguments.
    """
    tool_llm = llm.bind_tools(candidate_tools)

    messages = [
        SystemMessage(content="""
            You are a Tool Router.
            You MUST select the best tool from the provided tool list and call it
            using a standard LangChain tool_call with JSON arguments.
            Return ONLY a tool call. Do NOT output anything else.
        """),
        HumanMessage(content=f"User query: {query}")
    ]

    response = tool_llm.invoke(messages)
    print("LLM tool_call:", response)

    tool_calls = response.additional_kwargs.get("tool_calls")
    if not tool_calls:
        raise ValueError("LLM did not return a tool call.")

    call = tool_calls[0]
    tool_name = call["function"]["name"]
    args = json.loads(call["function"]["arguments"])

    return tool_name, args


# Example query
user_query = "Solve the integral of sin(x) from 0 to pi"

# 1. Vector search for candidate tools
candidate_tools = select_tool(user_query, top_k=2)
print("Candidate tools:", [t.name for t in candidate_tools])

# 2. Ask LLM: among these two tools, which one should be used?
llm = ChatOpenAI(model_name="gpt-4o") 
selected_tool_name, args = determine_tool_call(user_query, candidate_tools, llm)
print("Selected tool:", selected_tool_name)
print("Arguments:", args)

# 3. Execute selected tool
selected_tool = globals()[selected_tool_name]
result = selected_tool.invoke(args)

print("Tool result:", result)

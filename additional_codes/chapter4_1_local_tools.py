import os
os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4o")

from langchain_core.messages import HumanMessage
messages = [HumanMessage("What’s the answer of 1 plus 2?")]

from langchain_core.tools import tool
@tool
def add_numbers(x: int, y: int) -> int:
    """Adds two numbers and returns the sum."""
    return x + y

llm_with_tools = llm.bind_tools([add_numbers])
ai_msg = llm_with_tools.invoke(messages)


# 处理工具调用
tool_calls = ai_msg.additional_kwargs['tool_calls']
for tool_call in tool_calls:
    # 提取工具调用的函数名和参数
    function_name = tool_call['function']['name']
    function_args = tool_call['function']['arguments']
    
    # 解析参数（假设参数是 JSON 格式）
    import json
    parsed_args = json.loads(function_args)
    print(parsed_args)
    
    # 确保工具名称匹配
    if function_name == 'add_numbers':
        # 调用 add_numbers 工具并传入解析后的参数
        # tool_response = add_numbers(**parsed_args)
        tool_response = add_numbers.invoke(parsed_args)
        print(f"Tool response: {tool_response}")

#---------------------------------------------------------------------------------
import os
os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json

# Define tools using concise function definitions 
@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the 'y'."""
    return x**y
    
@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

tools = [multiply, exponentiate, add]
# Initialize the LLM with GPT-4o and bind the tools 
llm = ChatOpenAI(model_name="gpt-4o", temperature=0) 
llm_with_tools = llm.bind_tools(tools)

query = "What is 393 * 12.25? Also, what is 11 + 49?" 
messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages) 
messages.append(ai_msg)

print(ai_msg)

# 处理工具调用
tool_dict = {"add": add, "multiply": multiply, "exponentiate": exponentiate}
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


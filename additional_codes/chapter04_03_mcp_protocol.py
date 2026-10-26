class AgentState(TypedDict):
    messages: Sequence[Any] # A list of BaseMessage/HumanMessage/...

mcp_client = MultiServerMCPClient(
    {
        "math": {
            "command": "python3",
            "args": ["src/common/mcp/MCP_math_server.py"], 
            "transport": "stdio", # Subprocess → STDIO JSON-RPC
        },
        "weather": {
            # Assumes a separate MCP server is already running on port 8000 
            "url": "http://localhost:8000/mcp",
             "transport": "streamable_http",
             # HTTP→JSON-RPC over WebSocket/stream
        },
    }
)

async def get_mcp_tools() -> list[Tool]: 
    return await mcp_client.get_tools()

async def call_mcp_tools(state: AgentState) -> dict[str, Any]: 
    messages = state["messages"]

    last_msg = messages[-1].content.lower()
    # Fetch and cache MCP tools on the first call global MCP_TOOLS
    if "MCP_TOOLS" not in globals():
        MCP_TOOLS = await mcp_client.get_tools()
    # Simple heuristic: if any digit-operator token appears, choose "math" 
    if any(token in last_msg for token in ["+", "-", "*", "/", "(", ")"]):
        tool_name = "math"
    elif "weather" in last_msg: 
        tool_name = "weather"
    else:
        # No match → respond directly 
        return {
            "messages": [
                {
                    "role": "assistant",
                    "Sorry, I can only answer math" + " or weather queries."
                }
            ]
        }
    
    tool_obj = next(t for t in MCP_TOOLS if t.name == tool_name)
    
    user_input = messages[-1].content
    mcp_result: str = await tool_obj.arun(user_input)
    
    return {
        "messages": [
            {"role": "assistant", "content": mcp_result}
        ]
    }

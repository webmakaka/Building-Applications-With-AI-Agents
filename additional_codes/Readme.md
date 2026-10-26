# Кодом любезно поделился @icejean

https://github.com/michaelalbada/BuildingApplicationsWithAIAgents/issues/8

<br/>

requirements.txt

```
# Core language & typing
typing_extensions>=4.10

# LLM client & tokenizer
openai>=1.25
tiktoken>=0.5.1

# LangChain + LangGraph (stable set)
langchain==0.3.27
langchain-core==0.3.80 
langchain-community==0.3.31
langchain-openai==0.2.3
langgraph==0.3.34

# MCP integration
langchain-mcp-adapters==0.1.14
mcp[cli]>=1.9.2

# Observability / tracing
langsmith==0.4.58

# Additional packages
fastapi
uvicorn
aiohttp
python-dotenv
websockets
socksio
wikipedia
pandas
faiss-cpu

# For Jupyter Lab
ipykernel
```


<br/>

pyproject.toml


```
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "agents"
version = "0.0.1"
description = "Local install for BuildingApplicationsWithAIAgents"
authors = [{ name = "Jean" }]
requires-python = ">=3.8"

[tool.setuptools]
package-dir = { "" = "src" }

[tool.setuptools.packages.find]
where = ["src"]
```

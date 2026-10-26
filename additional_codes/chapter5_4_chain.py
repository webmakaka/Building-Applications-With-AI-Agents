from langchain_core.runnables import RunnableLambda 
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate 
# Wrap a function or model call as a Runnable
llm = RunnableLambda.from_callable(ChatOpenAI(model_name="gpt-4",temperature=0).generate) 
prompt = RunnableLambda.from_callable(lambda text:\
    PromptTemplate.from_template(text).format_prompt({"input": text}).to_messages())
# Traditional chain equivalent:
# chain = LLMChain(prompt=prompt, llm=llm) # LCEL chain using pipes:
chain = prompt | llm # Invoke the chain
result = chain.invoke("What is the capital of France?")

# ------------------------------------------------------------------------------
import os
from dotenv import load_dotenv
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. 定义 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 定义 Prompt
prompt = PromptTemplate.from_template(
    "Question: {question}\nAnswer:"
)

# 3. 用 LCEL 组合成 chain
chain = prompt | llm

# 4. 执行
result = chain.invoke({"question": "What is the capital of France?"})

print(result.content)

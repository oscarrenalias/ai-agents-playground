"""
Multiagent example
"""

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# initialize llm
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

greetings_agent = ReActAgent(
    name="calculator",
    description="Performs basic greetings",
    system_prompt="You are a greeting assistant. Greet the user in any formal way that you like before reporting a result.",
    llm=llm
)

# Create and run the workflow
agent = AgentWorkflow(
    agents=[calculator_agent, greetings_agent], root_agent="calculator"
)

# Run the system
async def main():
    response = await agent.run(user_msg="Can you add 5 and 3?")
    print(response)
    response = await agent.run(user_msg="My name is John")
    print(response)

import asyncio
asyncio.run(main())
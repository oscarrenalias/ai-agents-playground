"""
Tools are properly configured
Step limit is set appropriately
Agent name and description are provided
Basic configuration is complete
"""

# Create a tool-calling agent
from smolagents import (HfApiModel, ToolCallingAgent, tool)

model=HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

@tool
def say_good_morning(name: str) -> str:
    """
    This tool will generate a good morning greeting to anyone who calls it.

    Args:
        name: the name of the person
    """
    return f"Good morning, {name}!"

agent = ToolCallingAgent(
    tools=[say_good_morning],
    name="Tool calling agent quiz 4",
    description="This agent calls tools",
    max_steps=5,
    model=model
)
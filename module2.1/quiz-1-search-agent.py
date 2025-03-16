# Create a CodeAgent with DuckDuckGo search capability
from smolagents import (
  CodeAgent, 
  DuckDuckGoSearchTool,
  HfApiModel
)

model=HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()], # Add search tool here
    model=model          # Add model here
)
""""
Module 2 Quiz: Multi-Agent Structure"

Web agent has correct tools configured
Manager agent properly references web agent
Appropriate max_steps value is set
Required imports are authorized

"""

from smolagents import (ToolCallingAgent, CodeAgent, HfApiModel, DuckDuckGoSearchTool)

model=HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create web agent and manager agent structure
web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],           # Add required tools
    model=model,         # Add model
    max_steps=5,        # Adjust steps
    name="web agent",            # Add name
    description="you perform web searches"      # Add description
)

manager_agent = CodeAgent(  
    managed_agents=[web_agent],         # Add managed agents
    tools=[],        # Add tools
    model=model,        # Add model
    max_steps=5,       # Adjust steps
    name="manager agent",       # Add name
    description="you manage the web agent",     # Add description
    additional_authorized_imports=["time", "numpy", "pandas"]  # Add authorized imports
)
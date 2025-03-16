from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

# initialize llm
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# initialize agent
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)

async def main():
    # stateless
    response = await agent.run("What is 2 times 2?")
    print(response)

    # remembering state
    ctx = Context(agent)
    response = await agent.run("My name is Bob.", ctx=ctx)
    print(response)
    response = await agent.run("What was my name again?", ctx=ctx)
    print(response)

# Run the main function
import asyncio
asyncio.run(main())
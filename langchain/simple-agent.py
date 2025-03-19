# Import relevant functionality
from langchain_huggingface import (ChatHuggingFace, HuggingFaceEndpoint)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# authenticate to HF
from huggingface_hub import login
import os
login(token=os.getenv("HF_TOKEN"))

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

model = ChatHuggingFace(llm=llm, verbose=True)

# Create the agent
memory = MemorySaver()
agent_executor = create_react_agent(model, [], checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]

for step in agent_executor.stream(
    {"messages": messages },
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
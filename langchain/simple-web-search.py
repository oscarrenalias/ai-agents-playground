# Import relevant functionality
from langchain_huggingface import (ChatHuggingFace, HuggingFaceEndpoint)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.prompts.chat import ChatPromptTemplate

# authenticate to HF
from huggingface_hub import login
import os
login(token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"))

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
web_search_agent = create_react_agent(model, [], checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "1"}}

# define a tool
def weather_tool(input: str) -> str:
    """Returns the weather in a given location."""
    if "Finland" in input:
        return f"The weather in {input} is always snowy, did you know?"
    else:
        return f"The weather in {input} is sunny."

# Create the agent
memory = MemorySaver()
tools = [weather_tool]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in Spanish."),
        ("human", "{{input}}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{{agent_scratchpad}}"),
    ]
)

query_weather = "What is the weather in Finland?"
query_general = "How are you doing today?"

query = query_general
web_search_agent = create_react_agent(model, tools, prompt=prompt)
#messages = web_search_agent.invoke(
#)
#    {"messages": [("human", query)]}
#print(
#    {
#        "output": messages["messages"][-1].content,
#    }
#        "input": query,
#    )

for step in web_search_agent.stream(
    input=prompt,
    config=config,
    stream_mode="values"
):
    step["messages"][-1].pretty_print()
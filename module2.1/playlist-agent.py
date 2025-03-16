from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    LiteLLMModel,
)

# enable this to run with local ollama
#model = LiteLLMModel(
#    model_id="ollama_chat/gemma3:1b",
#    api_base="http://localhost:11434",
#    num_ctx=8192
#)
# Enable this to run with HF serverless calls

from huggingface_hub import login
login()

from smolagents import CodeAgent, tool, HfApiModel

# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

# Alfred, the butler, preparing the menu for the party
menu_agent = CodeAgent(tools=[suggest_menu], model=HfApiModel())

# Preparing the menu for the party
menu_agent.run("Prepare a formal menu for the party.")

music_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()], 
    #model=model,
    model=HfApiModel(),
    additional_authorized_imports=["requests"]
)

music_agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
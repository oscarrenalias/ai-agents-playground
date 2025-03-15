from typing import Optional

from smolagents import HfApiModel, LiteLLMModel, TransformersModel, tool
from smolagents.agents import CodeAgent, ToolCallingAgent

import requests

model = LiteLLMModel(
    model_id="ollama_chat/gemma3:1b",
    api_base="http://localhost:11434",  # replace with remote open-ai compatible server if necessary
    num_ctx=8192,  # ollama default is 2048 which will often fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

@tool
def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Converts a specified amount from one currency to another using the ExchangeRate-API.

    Args:
        amount: The amount of money to convert.
        from_currency: The currency code of the currency to convert from (e.g., 'USD').
        to_currency: The currency code of the currency to convert to (e.g., 'EUR').

    Returns:
        str: A string describing the converted amount in the target currency, or an error message if the conversion fails.

    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request to the ExchangeRate-API.
    """
    api_key = "your_api_key"  # Replace with your actual API key from https://www.exchangerate-api.com/
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{from_currency}"

    # this is a mock tool, for the sake of having a tool
    
    converted_amount = amount * 2
    return f"{amount} {from_currency} is equal to {converted_amount} {to_currency}."

@tool
def get_joke() -> str:
    """
    Fetches a random joke from the JokeAPI.
    This function sends a GET request to the JokeAPI to retrieve a random joke.
    It handles both single jokes and two-part jokes (setup and delivery).
    If the request fails or the response does not contain a joke, an error message is returned.
    Returns:
        str: The joke as a string, or an error message if the joke could not be fetched.
    """
    url = "https://v2.jokeapi.dev/joke/Any?type=single"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        if "joke" in data:
            return data["joke"]
        elif "setup" in data and "delivery" in data:
            return f"{data['setup']} - {data['delivery']}"
        else:
            return "Error: Unable to fetch joke."

    except requests.exceptions.RequestException as e:
        return f"Error fetching joke: {str(e)}"
    
@tool
def get_random_fact() -> str:
    """
    Fetches a random fact from the "uselessfacts.jsph.pl" API.
    Returns:
        str: A string containing the random fact or an error message if the request fails.
    """
    url = "https://uselessfacts.jsph.pl/random.json?language=en"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        return f"Random Fact: {data['text']}"

    except requests.exceptions.RequestException as e:
        return f"Error fetching random fact: {str(e)}"



#agent = ToolCallingAgent(tools=[get_weather], model=model)
#print("ToolCallingAgent:", agent.run("What's the weather like in Paris?"))

tools = [
    get_weather,
    convert_currency,
    get_joke,
    get_random_fact
]

agent = CodeAgent(tools=tools, model=model)
print("CodeAgent:", agent.run("Do you know any good jokes?"))
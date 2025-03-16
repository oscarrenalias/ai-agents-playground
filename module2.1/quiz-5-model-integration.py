"""
Quiz 5: Model Integration

Correct model imports are included
Model is properly initialized
Model ID is correctly specified
Alternative model option is provided
"""

from smolagents import (HfApiModel, LiteLLMModel)

model=HfApiModel(
    model_id="gpt-3.5-turbo"
)

# Alternative model via LiteLLM
other_model = LiteLLMModel("anthropic/claude-3-sonnet")
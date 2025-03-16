"""
Most basic example of LLamaIndex interacting with HuggingFace API
"""

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=os.getenv("HF_TOKEN")
)

llm.complete("Hello, how are you?")
# I am good, how can I help you today?
from typing import Literal
from typing_extensions import TypedDict, Dict, Any, Optional, List
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
# HuggingFace imports
from langchain_huggingface import (ChatHuggingFace, HuggingFaceEndpoint)
from huggingface_hub import login
import os
from langchain_core.messages import HumanMessage, AIMessage
import logging

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s"
)

# log into HF and configure the model
login(token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"))
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)
model = ChatHuggingFace(llm=llm, verbose=True)

#
# Set this to true if using the LangFuse handler. If set to True,
# set environment variables LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY
#
USE_LANGFUSE = True

# Initialize and return the Langfuse handler
def get_langfuse_handler():
    from langfuse.callback import CallbackHandler
    langfuse_handler = CallbackHandler(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host="https://cloud.langfuse.com" # EU region
    )
    return(langfuse_handler)

class EmailState(TypedDict):
    # The email being processed
    email: Dict[str, Any]  # Contains subject, sender, body, etc.
    
    # Analysis and decisions
    is_spam: Optional[bool]
    
    # Response generation
    draft_response: Optional[str]
    
    # Processing metadata
    messages: List[Dict[str, Any]]

    # define __str__ method for EmailState
    def __str__(self):
        return f"EmailState(email={self['email']}, is_spam={self['is_spam']}, draft_response={self['draft_response']}, messages={self['messages']})"

#
# open and read the email
#
def read_email(state: EmailState):
    logging.info("Analyzing email: %s", state)
    # no state change, just logging
    return {}

# 
# Checks if a given message is spam or not
#
def check_for_spam(state: EmailState) -> EmailState:
    # Ask the model to classify if the message is or isn't spam
    logging.info("Checking for spam: %s", state)

    # build a prompt for the model to determine if the mail is spam
    prompt = f"""
        You will be provided a sender, a subject and a boy. Your task is to determine if the email could be considered spam or not.

        Sender: {state["email"]["sender"]}
        Subject: {state["email"]["subject"]}
        Body: {state["email"]["body"]}

        If the email is spam, you must explain why and include "IS SPAM" in the response. If it is not spam, please make sure to state "NOT SPAM".
    """
    logging.info("Checking for spam with prompt: %s", prompt)

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    logging.info("Response received: %s", response.content)

    # determine if it is spam or not based on the response
    state["is_spam"] = "IS SPAM" in response.content.__str__()
    logging.info("is_spam set to: %s", state["is_spam"])

    # also, add the response to the state
    state["messages"].append({
        "role": "system",
        "content": response.content
    })

    state["messages"].append({
        "role": "system",
        "content": f"Email classified as {'spam' if state['is_spam'] else 'not spam'}."
    })
    return state


#
# Node that handles spam emails
#
def handle_spam(state: EmailState) -> EmailState:
    logging.info("Handling spam email: %s", state)
    state["messages"].append({
        "role": "system",
        "content": "Email marked as spam."
    })
    return state

# 
# Creates drafts from emails that are not spam
#
def create_draft(state: EmailState) -> EmailState:
    # Placeholder for generating draft responses
    logging.info("Creating draft response: %s", state)
    
    # build a prompt for the model to create a draft response
    prompt = f"""
        You will be provided a sender, a subject and a boy. Your task is to create a suitable response. Please end all
        messages with "Kind regards". Please address the sender by name at the beginning if you can determine the sender's name, 
        but do not make up a name if you can't figure one out, just start with "Hello".

        Sender: {state["email"]["sender"]}
        Subject: {state["email"]["subject"]}
        Body: {state["email"]["body"]}
    """
    logging.info("Requesting a draft to be generated with: %s", prompt)

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    logging.info("Draft message generate with the following content: %s", response.content)
    
    state["draft_response"] = response.content
    state["messages"].append({
        "role": "system",
        "content": "Draft response created."
    })
    state["messages"].append({
        "role": "system",
        "content": response.content
    })
    return state

#
# Provides a draft response to the user for validation and sending
#
def validate_draft(state: EmailState) -> EmailState:
    # interactive action that shows the draft email and waits for validation
    logging.info("Validating draft response: %s", state)
    state["messages"].append({
        "role": "system",
        "content": "Response draft validated successfully."
    })
    return state

#
# send the draft response after it's been validated
#
def send_response(state: EmailState) -> EmailState:
    # Placeholder for sending the response
    logging.info("Sending response: %s", state["draft_response"])
    state["messages"].append({
        "role": "system",
        "content": "Response sent successfully."
    })
    return state

#  
# Decision function to determine whether to classify spam or handle spam
#
def is_spam(state: EmailState) -> Literal["create_draft", "handle_spam"]:
    logging.info("Checking if email is spam: %s", state)
    return "create_draft" if not state["is_spam"] else "handle_spam"

# create graph and add nodes
builder = StateGraph(EmailState)
builder.add_node("read_email", read_email)
builder.add_node("check_for_spam", check_for_spam)
builder.add_node("handle_spam", handle_spam)
builder.add_node("create_draft", create_draft)
builder.add_node("validate_draft", validate_draft)
builder.add_node("send_response", send_response)

# connect nodes and build graph
builder.add_edge(START, "read_email")
builder.add_edge("read_email", "check_for_spam")
builder.add_conditional_edges("check_for_spam", is_spam)
builder.add_edge("handle_spam", END)
builder.add_edge("create_draft", "validate_draft")
builder.add_edge("validate_draft", "send_response")
builder.add_edge("send_response", END)

# create the graph and invoke
graph = builder.compile()

# test messages
test_message_not_spam = {
    "email": {
        "subject": "Important Message",
        "sender": "Doe, John <foo@bar.com>",
        "body": "This is a test email."
    },
    "is_spam": None,
    "draft_response": None,
    "messages": []
}

test_message_spam = {
    "email": {
        "subject": "Buy V1agr4 now!",
        "sender": "yaihhyl35@gmail.com",
        "body": "We have v1agr4 in large quantities."
    },
    "is_spam": None,
    "draft_response": None,
    "messages": []
}

# determine whether to use the LangFuse handler or not
if(USE_LANGFUSE):
    graph.invoke(test_message_not_spam, config={"callbacks": [get_langfuse_handler()]})
else:
    graph.invoke(test_message_not_spam)
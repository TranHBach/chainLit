import json
import ast
import os
from openai import AsyncOpenAI
from typing import Dict, Optional
from operator import itemgetter

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict

import chainlit as cl

api_key = os.environ.get("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

MAX_ITER = 5

import pickle

# Load the tools list from the file
with open('tools.pkl', 'rb') as f:
    tools = pickle.load(f)

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.base.llms.types import MessageRole

agent = OpenAIAgent.from_tools(tools, verbose=True)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.set_starters
async def set_starters():
    global agent
    agent = OpenAIAgent.from_tools(tools, verbose=True)
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/idea.svg",
            ),

        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            icon="/public/learn.svg",
            ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            icon="/public/terminal.svg",
            ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            icon="/public/write.svg",
            )
        ]


@cl.on_chat_start
def start_chat():
    # cl.user_session.set(
    #     "message_history",
    #     [{"role": "system", "content": "You are a helpful UBER assistant."}],
    # )
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    global agent
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])
    # context = """
    # These are your history chat with the client. 
    # You do not need to respond to this prompt but in the future, if the user ask questions, you can and should refer to the history chat to provide the best response.
    # """
    previous_messages = []
    for m in thread["steps"]:
        if (m["type"] == "user_message"):
            previous_messages.append(ChatMessage(role=MessageRole.USER, content=m["output"]))
        elif (m["type"] == "assistant_message"):
            previous_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=m["output"]))
    agent = OpenAIAgent.from_tools(tools=tools, chat_history=previous_messages, verbose=True)
    cl.user_session.set("memory", memory)


@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  return default_user

@cl.step(type="tool")
async def call_tool(message):
    response = agent.chat(message)
    return str(response)

@cl.on_message
async def run_conversation(message: cl.Message):
    # message_history = cl.user_session.get("message_history")
    # message_history.append({"name": "user", "role": "user", "content": message.content})
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    answer = await call_tool(message.content)
    res = cl.Message(content=answer, author="Answer")

    await res.send()
    print(memory.chat_memory)
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)

    # await cl.Message(content=message, author="Answer").send()


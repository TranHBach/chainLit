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


@cl.on_chat_start
def start_chat():
    # cl.user_session.set(
    #     "message_history",
    #     [{"role": "system", "content": "You are a helpful UBER assistant."}],
    # )
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

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

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)

    # await cl.Message(content=message, author="Answer").send()


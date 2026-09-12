from langchain.chat_models import init_chat_model
from base.config import config as conf
from langchain_core.messages import BaseMessage
from typing import Optional
import ollama

class InitModel:

    @staticmethod
    def chat_model():
        return init_chat_model(
            api_key=conf.llm.api_key,
            model=conf.llm.chat_model,
            base_url=conf.llm.base_url,
            model_provider="openai",
            temperature=conf.llm.temperature
        )

    @staticmethod
    def init_ollama_model(
            stream:Optional[bool],
            messages,
            think:Optional[bool],
            temperature:float=conf.llm.temperature,
            model = conf.ollama.model,
    ):
        return ollama.chat(
            model=model,
            think=think,
            messages=messages,
            stream=stream,
            options={"temperature":temperature},
        )

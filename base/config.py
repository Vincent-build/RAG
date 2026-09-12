from omegaconf import OmegaConf
from pathlib import Path
from dataclasses import dataclass
from typing import Union


@dataclass
class MySQL:
    host:str
    port:int
    user:str
    password:str
    database:str
    table_name:str

@dataclass
class Redis:
    host:str
    port:str
    password:str
    db:int
    index_name:str

@dataclass
class Milvus:
    host:str
    port:str
    database_name:str
    collection_name:str

@dataclass
class LLM:
    embed_model:str
    chat_model:str
    api_key:str
    base_url:str
    temperature:float

@dataclass
class Retrieval:
    parent_chunk_size:int
    child_chunk_size:int
    chunk_overlap:int
    retrieval_k:int
    candidate_m:int

@dataclass
class Logger:
    log_file:str

@dataclass
class Ollama:
    model:str
    embedding_model:str

@dataclass
class App:
    valid_sources:list[str]
    customer_service_phone:str

@dataclass
class SysPath:
    bge_m3:str
    bge_rerank_large:str
    databases:Union[str,Path]
    bge_large_zh:str


@dataclass
class AppConfig:
    mysql:MySQL
    redis:Redis
    milvus:Milvus
    llm:LLM
    retrieval:Retrieval
    logger:Logger
    ollama:Ollama
    app:App
    syspath:SysPath

conf_path = Path(__file__).parents[0] / "config.yaml"

schema = OmegaConf.structured(AppConfig)

context = OmegaConf.load(conf_path)

merge_conf = OmegaConf.merge(schema,context)

config:AppConfig = OmegaConf.to_object(merge_conf)

if __name__ == "__main__":
    model = config.llm.chat_model
    log = config.logger.log_file
    print(model)
    print(log)




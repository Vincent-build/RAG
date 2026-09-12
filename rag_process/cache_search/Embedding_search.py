"""
使用问题匹配Redis缓存中的问答对，击中则直接取回，若未击中则直接使用RAG
"""
from langchain_redis import RedisVectorStore,RedisConfig
from FlagEmbedding import FlagModel

from base.config import config as conf
from operate_database_management.mysql_operated import OperateMySQL
from operate_database_management.redis_operated import OperateRedis
from base.logger import logger


class FlagEmbeddingWrapper:
    def __init__(self,model):
        self.model = model

    def embed_query(self,texts):
        return self.model.encode_queries([texts]).tolist()

    def embed_documents(self,documents):
        return self.model.encode_corpus(documents).tolist()


class EmbeddingSearch:
    def __init__(self,mysql_client,redis_client,bge_large_zh_path=conf.syspath.bge_large_zh):
        self.index_name = conf.redis.index_name
        self.redis_config = None
        self.bge_large_zh_path = bge_large_zh_path
        self._init_embed_model()
        self._init_vector_store()
        self.mysql_operated = OperateMySQL(mysql_client)
        self.redis_operated = OperateRedis(redis_client)
        self._check_vector_store_content_num()




    def _init_embed_model(self):
        try:
            self.bge_large_zh = FlagModel(self.bge_large_zh_path,normalize_embeddings=True,use_fp16=False)
            self.bge_large_zh_model = FlagEmbeddingWrapper(self.bge_large_zh)
            self.model = self.bge_large_zh_model

            logger.info(f"初始化bge_large模型完成")
        except Exception as e:
            logger.error(f"EmbeddingSearch 初始化嵌入模型失败:{e}")


    def _init_vector_store(self):
        try:
            redis_url = f"redis://{conf.redis.host}:{conf.redis.port}"
            self.redis_config = RedisConfig(redis_url=redis_url,index_name=self.index_name,vector_datatype="Float32",embedding_dimensions=1024,distance_metric="COSINE")
            self.vector = RedisVectorStore(embeddings=self.model,config=self.redis_config)
            logger.info(f"初始化向量引擎完成")
        except Exception as e:
            logger.error(f"EmbeddingSearch 初始化向量引擎失败:{e}")


    def _check_vector_store_content_num(self):
        try:
            logger.info(f"开始校验缓存数据库同步状态")
            original_key = "qa_original_queries"
            doc_number = 0
            info = self.redis_operated.check_embeddings_num(self.index_name)
            if info:
                if info.get("num_docs",""):
                    doc_number = int(info["num_docs"])
            else:
                logger.warning(f"未查到向量数据库的索引数量")
                doc_number = 0
            list_data = self.redis_operated.get_list_data(original_key)
            if not list_data:
                list_data = []
            if len(list_data) != doc_number:
                logger.warning(f"向量数据数量与原问题存储数量不一致--向量:{doc_number}=原问题:{len(list_data)}")
                self.vector.index.delete(True)
                logger.warning(f"已清空向量数据库索引下数据")
                self.redis_operated.drop_data(original_key)
                logger.warning(f"已清空原问题数据")
            else:
                logger.info(f"向量数据与原问题数据数量一致,已同步")
                return
        except Exception as e:
            logger.error(f"EmbeddingSearch 检查向量数据数量异常:{e}")


    def _update_data(self):
        try:
            logger.info(f"EmbeddingSearch 更新缓存数据")
            original_key = "qa_original_queries"
            redis_queries = set()
            mysql_queries = set()
            mysql_get_queries = self.mysql_operated.get_question()

            if not mysql_get_queries:
                logger.info(f"MySQL不存在缓存内容")
                return
            mysql_original_queries = list(mysql_get_queries)
            for q in mysql_original_queries:
                mysql_queries.add(q[0])

            redis_get_queries = self.redis_operated.get_list_data(original_key)
            if redis_get_queries:
                for q in redis_get_queries:
                    redis_queries.add(q)
            else:
                logger.info(f"redis不存在缓存数据")
            difference_queries = list(mysql_queries - redis_queries)
            if difference_queries:
                if self.vector.index.exists():
                    self.vector.add_texts(difference_queries,metadatas=[{"question": mes} for mes in difference_queries])
                else:
                    self.vector.from_texts(difference_queries,metadatas=[{"question": mes} for mes in difference_queries], embedding=self.model, config=self.redis_config)
                logger.info(f"Redis向量库已存入问题列表")
                self.redis_operated.push_data(original_key,difference_queries)
                logger.info(f"Redis数据库已存入问题列表")
            else:
                logger.info(f"无需更新Redis缓存")
                return
        except Exception as e:
            logger.error(f"EmbeddingSearch 更新缓存数据失败:{e}")



    def search(self,query,threshold=0.85):
        try:
            self._update_data()
            if not self.vector.index.exists() or not self.mysql_operated.get_question():
                logger.info("缓存数据库数据未加载，使用RAG检索")
                return False,True

            cache_answer = self.redis_operated.get_answer(query)
            print("redis_缓存:",cache_answer)
            if cache_answer:
                logger.info(f"Redis缓存击中，直接召回答案")
                return cache_answer,False

            result = self.vector.similarity_search_with_score(query=query,k=1)
            if result:

                cache_answer = self.redis_operated.get_answer(query)
                print("redis_缓存:", cache_answer)
                if cache_answer:
                    logger.info(f"Redis缓存击中，直接召回答案")
                    return cache_answer, False

                doc,score = result[0]
                question = doc.metadata["question"]
                logger.info(f"向量库检索的分数为{1.0-score}；召回的问题:{question}")
                if (1.0-score) >= threshold:
                    answer = self.mysql_operated.get_answer(question)
                    if answer:
                        logger.info(f"MySQL缓存击中，直接召回答案")
                        self.redis_operated.set_data(f"answer:{query}",answer)
                        return answer,False
                    else:
                        logger.warning(f"缓存库中未查询到相关答案")
                        return False,True
                else:
                    logger.warning(f"缓存库中未查询到相关答案")
                    return False,True
        except Exception as e:
            logger.error(f"EmbedSearch 数据库检索失败:{e}")
            return False,True



if __name__ == "__main__":
    from client_managed.redis_client import RedisClientManagement
    from client_managed.mysql_client import MySQLClientManagement

    r_client = RedisClientManagement()
    r_client.init()

    m_client = MySQLClientManagement()
    m_client.init()

    embed = EmbeddingSearch(m_client.client,r_client.client)
    #embed.check_vector_store_content_num()
    embed.search("AIGC")




































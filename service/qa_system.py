from base.logger import logger
from base.config import config as conf
from langchain_process.model_tool.llm import InitModel
from operate_database_management.redis_operated import OperateRedis
from operate_database_management.mysql_operated import OperateMySQL
from operate_database_management.milvus_operated import OperateMilvus
#from rag_process.cache_search.bm25_search import BM25Search
from rag_process.cache_search.Embedding_search import EmbeddingSearch
from rag_process.core.rag_system import RAGSystem
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from pydantic import BaseModel
from typing import TypedDict,Annotated
from operator import add
import pymysql
import time


class IntegratedQASystem:
    def __init__(self,redis_client,
                 mysql_client,
                 milvus_client):
        self.logger = logger
        self.conf = conf
        self.chat_model = InitModel.chat_model()
        self.redis_operation = OperateRedis(redis_client)
        self.mysql_operation = OperateMySQL(mysql_client)
        self.milvus_operation = None
        #self.milvus_operation = OperateMilvus(milvus_client,self.conf.syspath.bge_m3,self.conf.syspath.bge_rerank_large)
        self.rag_system = RAGSystem(milvus_client,
                                              self.conf.syspath.bge_m3,
                                              self.conf.syspath.bge_rerank_large)
        #bm25检索效果不佳，采用vector取替
        #self.bm25_search = BM25Search(redis_client,mysql_client)
        self._init_conversation_table()
        self.knowledge_path = conf.syspath.databases
        self.cache_search = EmbeddingSearch(mysql_client=mysql_client,redis_client=redis_client)


    def document_load(self):
        try:
            self.rag_system.document_load(self.knowledge_path,
                      parent_chunk_size=conf.retrieval.parent_chunk_size,
                      child_chunk_size=conf.retrieval.child_chunk_size,
                      chunk_overlap=conf.retrieval.chunk_overlap)
        except Exception as e:
            logger.error(f"QA系统：文档载入失败:{e}")



    def _init_conversation_table(self):
        """初始化MySQL中的conversation表，用于存储对话历史"""
        try:
            cursor = self.mysql_operation.client.cursor()
            sql = "SELECT DATABASE()"
            cursor.execute(sql)
            response = cursor.fetchone()
            if self.conf.mysql.database == response[0]:
                list_tables = []
                sql = "SHOW TABLES"
                cursor.execute(sql)
                result = cursor.fetchall()
                for i in range(len(result)):
                    list_tables.append(result[i][0])
                if "conversations" not in list_tables:
                    #创建conversations表，包含会话ID、问题
                    logger.info(f"QA初始化：创建conversations表")
                    tem = """
                    CREATE TABLE IF NOT EXISTS conversations(
                        id INT PRIMARY KEY AUTO_INCREMENT,
                        session_id VARCHAR(36) NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        INDEX idx_session_id(session_id)
                    )
                    """
                    cursor.execute(tem)
                    self.mysql_operation.client.commit()
                else:
                    logger.info(f"QA初始化MySQL：conversations表完成")
            else:
                print(f"QA无法加载MySQL数据库{self.conf.mysql.database}")
        except Exception as e:
            self.logger.error(f"QA初始化MySQL：conversations表失败：{e}")


    def _fetch_recent_history(self,session_id):
        """
        取最近5轮对话历史(问题与回答)，根据timestamp字段
        :param session_id: 对话用户ID
        :return:
        """
        try:
            cursor = self.mysql_operation.client.cursor()
            sql = """
            SELECT question,answer FROM conversations 
            WHERE session_id=%s ORDER BY timestamp DESC LIMIT %s
            """
            cursor.execute(sql,(session_id,5))
            message = cursor.fetchall()
            history = [{'question':row[0],'answer':row[1]} for row in message]
            return history[::-1]
        except Exception as e:
            self.logger.error(f"QA流程：获取对话历史数据异常:{e}")
            return []

    def get_session_history(self,session_id:str):
        """从MySQL中获取会话历史"""
        return self._fetch_recent_history(session_id)


    def update_session_history(self,session_id:str,question:str,answer:str):
        """更新会话历史到MySQL，保留最近5轮对话"""
        try:
            cursor = self.mysql_operation.client.cursor()
            sql = """
            INSERT INTO conversations(session_id,question,answer,timestamp) VALUES(%s,%s,%s,NOW())
            """
            cursor.execute(sql,(session_id,question,answer,))

            history = self._fetch_recent_history(session_id)

            sql = """
            DELETE FROM conversations WHERE session_id=%s AND id NOT IN
                (SELECT id FROM 
                    (SELECT id FROM conversations WHERE session_id=%s ORDER BY timestamp DESC LIMIT %s)  
                AS sub)
            """
            cursor.execute(sql,(session_id,session_id,5))
            sql = ""
            self.mysql_operation.client.commit()
            self.logger.info(f"QA流程：会话{session_id}历史会话更新成功")
            return history
        except pymysql.MySQLError as e:
            self.logger.error(f"QA流程：会话{session_id}更新失败:{e}")
            self.mysql_operation.client.rollback()
            raise
        except Exception as e:
            self.logger.error(f"QA流程：会话{session_id}更新出现错误:{e}")
            self.mysql_operation.client.rollback()
            raise

    def clear_session_history(self,session_id:str):
        """清除指定用户会话历史"""
        try:
            cursor = self.mysql_operation.client.cursor()
            sql = """
            DELETE FROM conversations WHERE session_id=%s
            """,(session_id,)
            cursor.execute(sql)
            self.mysql_operation.client.commit()
            logger.info(f"会话{session_id}已清除")
        except pymysql.MySQLError as e:
            logger.error(f"会话{session_id}清除失败")
            self.mysql_operation.client.rollback()
            return False


    def save_mysql_qa(self,answer:str,query:str,source_filter:str):
        """存储每次问答对进MySQL"""
        try:
            cursor = self.mysql_operation.client.cursor()
            sql = f"""
            INSERT INTO {self.conf.mysql.table_name}(subject_name,question,answer) 
            VALUES(%s,%s,%s)
            """
            cursor.execute(sql,(source_filter,query,answer))
            self.mysql_operation.client.commit()
            logger.info(f"MySQL存储问答对完成")
        except Exception as e:
            logger.error(f"QA流程 MySQL存储问答对失败:{e}")


    def save_redis_qa(self,answer:str,query:str):
        """存储每次问答对进Redis"""
        try:
            self.redis_operation.set_data(f"answer:{query}",answer)
            logger.info(f"Redis存储问答对完成")
        except Exception as e:
            logger.error(f"QA流程：Redis存储问答对异常:{e}")


    def chat_bot(self,query,source_filter=None,history=None):
        """结合RAG，进行回答"""
        rag_prompt = self.rag_system.generate_RAG_prompt(query, source_filter=source_filter, history=history)
        chain = rag_prompt | self.chat_model | StrOutputParser()
        result = chain.stream({"query": query})
        for mes in result:
            if mes.strip() in ["", " ", "\n", "\n\n"]:
                continue
            yield mes


    def query(self,query,source_filter=None,session_id=None):
        """查询集成系统，支持对话历史流式输出"""
        start_time = time.time()
        self.logger.info(f"处理查询:'{query}' (会话ID:{session_id})")
        history = self.get_session_history(session_id)
        answer,need_rag = self.cache_search.search(query,threshold=0.85)

        if answer:
            self.logger.info(f"QA流程：MySQL找到问题答案")
            if session_id:
                self.update_session_history(session_id,question=query,answer=answer)
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时{processing_time:.2f}s")
            yield answer,True

        elif need_rag:
            self.logger.warning(f"QA流程：开启RAG功能")
            collected_answer = ""
            token = self.chat_bot(query,source_filter=source_filter,history=history)

            for mes in token:
                collected_answer += str(mes).strip()
                yield mes, False
            print("模型回答:",collected_answer)

            if session_id:
                self.update_session_history(session_id,question=query,answer=collected_answer)
            if collected_answer:
                self.save_mysql_qa(source_filter=source_filter,answer=collected_answer,query=query)
                self.save_redis_qa(answer=collected_answer,query=query)
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时{processing_time:.2f}s")
            yield "",True

        else:
            self.logger.info(f"未找到答案")
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时{processing_time:.2f}s")
            yield "未找到答案",True


if __name__ == "__main__":
    from client_managed.mysql_client import MySQLClientManagement
    from client_managed.redis_client import RedisClientManagement
    from client_managed.milvus_client import MilvusClientManagement
    redis_client = RedisClientManagement()
    mysql_client = MySQLClientManagement()
    milvus_client = MilvusClientManagement()
    redis_client.init()
    mysql_client.init()
    milvus_client.init()

    i = IntegratedQASystem(redis_client.client,mysql_client.client,milvus_client.client)
    #i.document_load()






























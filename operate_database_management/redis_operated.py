"""
作用：
    答案缓存
    BM25分词索引
    会话状态
    Redis：存储的是原始问题及问题分词

功能操作：
    存数据
    取数据
    获取答案
    查询所有数据
    删除数据
"""
from base.config import config as conf
from client_managed.redis_client import RedisClientManagement
from base.logger import logger
import json
import redis

class OperateRedis:
    def __init__(self,client:RedisClientManagement):
        self.client = client

    def set_data(self,key,value):
        """对存储内容覆盖 str"""
        try:
            self.client.set(name=key, value=json.dumps(value,ensure_ascii=False))
            #print(f"redis写入完成{key}:{value}")
        except Exception as e:
            logger.error(f"Redis-str写入异常:{e}")

    def push_data(self,key,value):
        """存入list数据格式"""
        try:
            self.client.rpush(key,json.dumps(value,ensure_ascii=False))
            #print(f"redis写入列表{key}:{value}")
        except Exception as e:
            logger.error(f"Redis-list写入异常:{e}")

    def get_list_data(self,key):
        """获取list数据"""
        try:
            list_data = self.client.lrange(key,0,-1)
            if list_data:
                data_list = []
                for result in list_data:
                    data_list += json.loads(result)
                #print(f"redis获取list数据{key}:{data_list}")
                return data_list
        except Exception as e:
            logger.error(f"Redis-list读取异常:{e}")

    def get_data(self,key):
        """str"""
        try:
            data = self.client.get(key)
            if data:
                #print("获取Redis内容成功")
                return data
        except Exception as e:
            logger.error(f"Redis-str读取异常:{e}")


    def get_answer(self,query):
        try:
            answer = self.client.get(query)
            if answer:
                #print("获取Redis答案")
                return answer
        except Exception as e:
            logger.error(f"Redis-str读取异常:{e}")


    def get_whole_data(self):
        try:
            data = self.client.get(name="*")
            if data:
                #print("获取Redis内容成功")
                return data
        except Exception as e:
            logger.error(f"Redis-str读取异常:{e}")


    def drop_data(self,key):
        try:
            self.client.delete(key)
            logger.warning(f"已删除{key}内容")
        except Exception as e:
            logger.error(f"Redis-str删除内容异常:{e}")


    def check_embeddings_num(self,document):
        """检查数据库下某个索引或文档下的数据条数"""
        try:
            info = self.client.execute_command("FT.INFO",document)
            #print("info",info)
            return info
        except Exception as e:
            logger.error(f"Redis查询内容异常:{e}")



if __name__ == "__main__":
    client = RedisClientManagement()
    client.init()
    operate = OperateRedis(client.client)
    #operate.set_data("name","王五")
    #result = operate.get_answer("name")
    #result = operate.check_embeddings_num(conf.redis.index_name)
    #print(result)



















"""
通过jieba分词再放进BM25进行计算

经过验证，该方法十分依赖BM25的TF-IDF检索功能，对于较短文本（或同样的文本进行检索时），出现的停用词较多的情况时，所得分数越低。采用停用词表后稍微好些，但无法解决IDF的问题
"""
from base.logger import logger
from operate_database_management.mysql_operated import OperateMySQL
from operate_database_management.redis_operated import OperateRedis
from rag_process.cache_search.data_preprocess import preprocess_text

from rank_bm25 import BM25Okapi
import numpy as np
from pathlib import Path





class BM25Search:
    def __init__(self,redis_client,mysql_client):
        self.redis_operation = OperateRedis(redis_client)
        self.mysql_operation = OperateMySQL(mysql_client)
        self.bm25 = None
        self.queries = None
        self.original_queries = None
        self._update_data()


    def _load_stopwords(self):
        txt_path = Path(__file__).parents[0] / "stopwords.txt"
        with open(txt_path,'r',encoding='utf-8') as f:
            stopwords = {line.strip() for line in f if line.strip()}
        return stopwords


    def _load_data(self):
        stopwords = self._load_stopwords()
        original_key = "qa_original_queries"
        tokenized_key = "qa_tokenized_queries"
        self.original_queries = self.redis_operation.get_list_data(original_key)
        tokenized_queries = self.redis_operation.get_list_data(tokenized_key)
        if not self.original_queries or not tokenized_queries:
            if self.mysql_operation.get_question():
                self.original_queries = list(set(self.mysql_operation.get_question()))
            else:
                logger.info(f"cache search初始化检查：MySQL数据库未加载问题")
                return None
            tokenized_queries = [preprocess_text(q[0]) for q in self.original_queries]
            self.redis_operation.push_data(original_key,self.original_queries)
            self.redis_operation.push_data(tokenized_key,tokenized_queries)
            logger.info(f"cache search流程：redis写入问题集完成")
        #停用词处理
        for chunk in tokenized_queries:
            for elem in chunk:
                if elem in stopwords:
                    chunk.remove(elem)
        self.queries = tokenized_queries
        print("queries:",self.queries)
        self.bm25 = BM25Okapi(self.queries)
        logger.info("BM25模型初始化完成")
        return True


    def _update_data(self):
        try:
            logger.info("bm25:开始更新redis缓存数据")
            original_key = "qa_original_queries"
            tokenized_key = "qa_tokenized_queries"
            redis_queries = set()
            #从数据库获取原问题数据
            mysql_get_original_queries = self.mysql_operation.get_question()
            redis_original_queries = self.redis_operation.get_list_data(original_key)
            if mysql_get_original_queries:
                mysql_original_queries = set(q[0] for q in mysql_get_original_queries)
            else:
                logger.warning(f"MySQL数据库未加载问题，停止更新Redis")
                return
            if redis_original_queries:
                for data in redis_original_queries:
                    redis_queries.add(data[0])
            difference_queries = list(mysql_original_queries - redis_queries)
            if difference_queries:
                logger.info(f"新增原问题:{difference_queries}")
                tokenized_queries = [preprocess_text(q) for q in difference_queries]
                #写入
                self.redis_operation.push_data(original_key, difference_queries)
                self.redis_operation.push_data(tokenized_key, tokenized_queries)
            else:
                logger.warning(f"无内容需更新")
        except Exception as e:
            logger.error(f"bm25流程 更新数据异常:{e}")




    def _softmax(self,scores):
        """
        使用softmax归一化，同步各数据区间范围，便于对比
        :param scores:
        :return:
        """
        #输入值减去最大值，防止溢出
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()


    def search(self,query,threshold=0.85):
        """
        查询缓存答案
        BM25Okapi计算值正值才算好的能匹配的词，且值越大越类似，但取值<=BM25设定的k值
        softmax的输入向量只有一个时，输出结果恒等于1
        :param query: 用户问题
        :param threshold: 阈值
        :return: 输出结构[缓存答案，是否启用RAG]
        """
        response = self._load_data()
        if not response:
            logger.info(f"未找到查询问题历史，退出cache search")
            return None,True
        logger.info(f"开始进行cache search")
        if not query or not isinstance(query,str):
            logger.error(f"无效查询，问题不符合文本类型要求")
            return None,False
        cache_answer = self.redis_operation.get_answer(str(query))
        if cache_answer:
            logger.info(f"Redis存在缓存答案，直接回答")
            return cache_answer,False
        try:
            query_tokens = preprocess_text(query)
            print("query_tokens:",query_tokens)
            print(f"分词结果：{query_tokens}")
            scores = self.bm25.get_scores(query_tokens)
            if len(scores) <= 1:
                logger.info("词库数量不足，无法有效匹配缓存内容")
                return False,True
            print(f"scores:{scores}")
            softmax_score = self._softmax(scores)
            print(f"softmax_score:{softmax_score}")
            best_idx = softmax_score.argmax()
            best_score = softmax_score[best_idx]
            #检查分数对比阈值
            if best_score >= threshold:
                original_query = self.original_queries[best_idx][0]
                logger.info(f"BM25检索出最佳问题{original_query}")
                answer = self.mysql_operation.get_answer(original_query)
                if answer:
                    #Redis问答对缓存
                    self.redis_operation.set_data(str(query),answer)
                    logger.info(f"搜索成功，SOFTMAX相似度:{best_score:.3f}")
                    return answer,False
            logger.info(f"未找到可靠答案，最高SoftMax相似度:{best_score:.3f}")
            return None,True
        except Exception as e:
            logger.error(f"BM25流程:搜索查询失败：{e}")
            return None,True


if __name__ == "__main__":
    from client_managed.redis_client import RedisClientManagement
    from client_managed.mysql_client import MySQLClientManagement

    r_client = RedisClientManagement()
    r_client.init()

    m_client = MySQLClientManagement()
    m_client.init()

    bm_search = BM25Search(r_client.client,m_client.client)
    #bm_search.update_data()
    result = bm_search.search("AIGC是什么")
    #print(result)
    #bm_search.test()










































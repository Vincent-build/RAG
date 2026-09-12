from langchain_process.model_tool.llm import InitModel
from langchain_process.prompts.prompts import RAGPrompts
from langchain_process.strategy_selector.strategy_selector import StrategySelector
from operate_database_management.milvus_operated import OperateMilvus
from rag_process.core.document_processor import process_documents

from base.logger import logger
from base.config import config as conf
import os

from typing import Union


class RAGSystem:
    def __init__(self,
                 milvus_client,
                 bge_m3_model_path:str,
                 bge_rerank_large_model_path:str):
        self.milvus_operated: Union[OperateMilvus,None] = OperateMilvus(milvus_client,
                                                                     bge_m3_path=bge_m3_model_path,
                                                                     bge_rerank_large=bge_rerank_large_model_path)
        self.chat_model = InitModel.chat_model()
        self.rag_prompt_class = RAGPrompts
        self.strategy_selector = StrategySelector()

    def document_load(self,
                      dir_path:str,
                      parent_chunk_size=conf.retrieval.parent_chunk_size,
                      child_chunk_size=conf.retrieval.child_chunk_size,
                      chunk_overlap=conf.retrieval.chunk_overlap
                      ):
        try:
            #该路径是总知识库文件夹
            documents = os.listdir(dir_path)
            if documents:
                for dir in documents:
                    source_path = os.path.join(dir_path,dir)
                    if os.path.isdir(source_path):
                        if os.listdir(source_path):
                            logger.info(f"加载路径{source_path}文档")
                            docs = process_documents(source_path,parent_chunk_size=parent_chunk_size,child_chunk_size=child_chunk_size,chunk_overlap=chunk_overlap)
                            if docs:
                                logger.info(f"文档开始写入数据库")
                                self.milvus_operated.add_documents(docs)
                        else:
                            logger.warning(f"路径{source_path}文件夹为空")
                            continue
                    else:
                        logger.warning(f"RAG流程：跳过非文件夹")
                        continue
            else:
                logger.warning(f"RAG流程：知识文档内部不存在文件")
        except Exception as e:
            logger.error(f"RAG系统：文档载入流程异常:{e}")

    def _retrieve_with_backtracking(self,query,source_filter):
        logger.info(f"使用回溯问题策略进行检索:{query}")
        backtrack_prompt_template = self.rag_prompt_class.back_tracking_prompt()
        try:
            simplified_query = self.chat_model.invoke(backtrack_prompt_template.format(query=query))
            print(f"生成回溯问题:{simplified_query.content}")
            return self.milvus_operated.hybird_search_with_rerank(query=str(simplified_query.content),source_filter=source_filter)
        except Exception as e:
            logger.error(f"回溯问题策略执行失败:{e}")
            return []

    def _retrieve_with_subqueries(self, query, source_filter):
        logger.info(f"使用子查询策略进行检索(查询:'{query}')")
        subquery_prompt_template = self.rag_prompt_class.subquery_prompt()
        try:
            subqueries_text = self.chat_model.invoke(subquery_prompt_template.format(query=query))
            subqueries = [q for q in subqueries_text.content.split("\n")]
            logger.info(f"子查询生成:{subqueries}")
            if not subqueries:
                logger.warning(f"未能生成有效的子查询")
                return []
            all_docs = []
            for subquery in subqueries:
                docs = self.milvus_operated.hybird_search_with_rerank(str(subquery),source_filter=source_filter)
                if docs:
                    all_docs.extend(docs)
                    logger.info(f"子查询 {subquery} 检索到{len(docs)}个结果")
            unique_docs_dict = {doc.page_content:doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())
            logger.info(f"所有子查询共检索到{len(all_docs)}个文档，去重后剩下{len(unique_docs)}")
            return unique_docs
        except Exception as e:
            logger.error(f"子查询执行失败:{e}")
            return []


    def _retrieve_with_hyde(self, query, source_filter):
        logger.info(f"使用HyDE策略进行检索(查询'{query}')")
        hyde_prompt_template = self.rag_prompt_class.hyde_prompt()
        try:
            hyde_response = self.chat_model.invoke(hyde_prompt_template.format(query=query))
            logger.info(f"HyDE生成的假设答案:'{hyde_response}'")
            return self.milvus_operated.hybird_search_with_rerank(str(hyde_response.content),source_filter=source_filter)
        except Exception as e:
            logger.error(f"HyDE执行失败：{e}")
            return []


    def retrieve_and_merge(self, query, source_filter,strategy=None):
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)
        if strategy == "回溯问题检索":
            ranked_chunks = self._retrieve_with_backtracking(query,source_filter=source_filter)
        elif strategy == "子查询检索":
            ranked_chunks = self._retrieve_with_subqueries(query,source_filter=source_filter)
        elif strategy == "假设问题检索":
            ranked_chunks = self._retrieve_with_hyde(query,source_filter=source_filter)
        else:
            logger.info(f"使用直接检索策略(查询'{query}')")
            ranked_chunks = self.milvus_operated.hybird_search_with_rerank(query,source_filter=source_filter)

        logger.info(f"策略{strategy}:检索到{len(ranked_chunks)}个候选文档[包含父文档]")
        final_context_docs = ranked_chunks[:conf.retrieval.candidate_m]
        logger.info(f"最终选取{len(final_context_docs)}个文档作为上下文")
        return final_context_docs


    def generate_RAG_prompt(self,query,source_filter=None,history=None):
        logger.info(f"开始处理查询:'{query}'，学科过滤:{source_filter}")

        if history is not None and not isinstance(history,list):
            logger.warning(f"无效的历史格式:{type(history)},忽略历史")
            history = []
        elif history:
            history = history[-5:]

        history_context = ""
        if history:
            history_context = "\n".join(f"Q:{h['question']}\nA:{h['answer']}" for h in history)
            logger.info(f"使用对话历史:{history_context[:50]}")

        logger.info(f"使用RAG检索功能")
        strategy = self.strategy_selector.select_strategy(query)
        context_docs = self.retrieve_and_merge(query=query,source_filter=source_filter,strategy=strategy)
        if context_docs:
            context = '\n\n'.join([doc.page_content for doc in context_docs])
            logger.info(f"获取RAG文档，共{len(context_docs)}个")
        else:
            context = ''
            logger.info("未检索到RAG相关文档")

        rag_prompt = self.rag_prompt_class.rag_prompt()

        output_prompt = rag_prompt.partial(history=history_context,
                                          context=context,
                                          phone=conf.app.customer_service_phone)
        return output_prompt


    """
    def test(self,query,source_filter):
        #return self._retrieve_with_backtracking(query,source_filter)
        #return self._retrieve_with_subqueries(query,source_filter)
        return self._retrieve_with_hyde(query,source_filter)
    """

if __name__ == "__main__":
    from client_managed.milvus_client import MilvusClientManagement
    client = MilvusClientManagement()
    client.init()
    rag = RAGSystem(client.client,conf.syspath.bge_m3,conf.syspath.bge_rerank_large)
    result = rag.generate_RAG_prompt("Introduce Python",source_filter="ai")
    print(result)








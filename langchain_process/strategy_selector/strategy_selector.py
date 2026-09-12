from langchain_core.prompts import PromptTemplate
from pathlib import Path
from base.config import config as conf
from base.logger import logger
from langchain_process.model_tool.llm import InitModel
import ollama

class StrategySelector:
    def __init__(self):
        self.chat_model = InitModel.chat_model()
        self.ollama_model = None
        self.strategy_prompt_template = self._get_strategy_prompt()

    def llm_response(self,prompt):
        result = "直接检索"
        try:
            response = self.chat_model.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"大模型服务器异常：{e}")
            return result

    def ollama_response(self,prompt):
        result = "直接检索"
        try:
            response = self.ollama_model = InitModel.init_ollama_model(stream=None,
                                                                        messages=[
                                                                            {'role':'system','content':'你是AI助手'},
                                                                            {'role':'user','content':prompt}],
                                                                        think=None,)
            return response
        except Exception as e:
            logger.error(f"ollama模型服务器异常:{e}")
            return result


    def _get_strategy_prompt(self):
        return PromptTemplate(
            template="""
            你是一名AI助手，负责分析用户查询{query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。
            
            以下是几种检索增强策略及其使用场景：
            
            1. **直接检索**
                *描述：对用户查询直接进行检索，不进行任何增强处理。
                *适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    *示列：
                        *查询：AI学科学费是多少？
                        *策略：直接检索
                    *查询：JAVA的课程大纲是什么？
                        *策略：直接检索
            2.**假设问题检索（HyDE）：**
                *描述：使用LLM生成一个假设的答案，然后基于假设答案进行检索。
                *适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    *示例：
                        *查询：人工智能在教育领域的应用有哪些？
                        *策略：假设问题检索
            3.**子查询检索**
                *描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                *适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    *示例：
                        *查询：比较Milvus和Qdrant的优缺点。
                        *策略：子查询检索
            4.**回溯问题检索：**
                *描述：将复杂的用户查询转化为更基础，更易于检索的问题，然后进行检索。
                *适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    *示例：
                        *查询：我有一个包含100亿条记录的数据集，想把它储存到Milvus中进行查询，可以吗？
                        *策略：回溯问题检索
            
            根据用户查询{query}，直接返回最适合的策略名称，例如“直接检索”，不要输出任何分析过程或其他内容。
                        
            """,
            input_variables=["query"]
        )

    def select_strategy(self,query):
        prompt = self.strategy_prompt_template.format_prompt(query=query)
        strategy = self.llm_response(prompt)
        return strategy

if __name__ == "__main__":
    s = StrategySelector()
    print(s.select_strategy(query="MySQL是否支持多个样本同时写入"))












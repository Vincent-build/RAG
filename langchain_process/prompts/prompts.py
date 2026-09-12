from langchain_core.prompts import PromptTemplate


class RAGPrompts:
    @staticmethod
    def rag_prompt():
        """
        带有历史上下文的RAG
        :return:
        """
        return PromptTemplate(
            template="""
            你是一个智能助手，负责帮助用户回答问题。请按照以下步骤处理：
                
                1.**分析问题和上下文**：
                    - 基于提供的上下文（如果有）和你的知识回答问题
                    - 如果答案来源于检索到的文档，请在回答中明确说明，例如：“根据提供的文档”
                
                2.**评估对话历史**：
                    - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）
                    - 如果对话历史与问题相关，请结合历史信息生成更准确的回答
                    - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答
                    
                3.**生成回答**：
                    - 提供清晰、准确的回答，避免无关信息。
                    - 如果上下文和历史消息均不足以回答问题，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}”
                    
                **对话历史**：
                {history}
                **上下文**:
                {context}
                **问题**:
                {query}
                
                **回答**:
                  ""
            """,
            input_variables=["history","context","query","phone"]
        )

    @staticmethod
    def hyde_prompt():
        return PromptTemplate(
            template="""
            分析问题，根据问题内容大意，请生成一个简短的假设答案：
            问题：{query}
            假设答案：
            
            """,
            input_variables=["query"]
        )


    @staticmethod
    def subquery_prompt():
        return PromptTemplate(
            template="""
            将以下复杂查询分解为多个简单子查询，每行一个子查询，最多生成两个子查询（只保留子查询问题，其他的文本都不需要）：
            eg：
            用户原始query：“Milvus和Qdrant在功能上有什么不同？”
            子查询：“Milvus有哪些功能？”，“Qdrant有哪些功能？”
            
            查询：{query}
            子查询:
            
            """,
            input_variables=["query"]
        )

    @staticmethod
    def back_tracking_prompt():
        return PromptTemplate(
            template="""
            将以下复杂查询简化为一个更简单的问题：
            查询：{query}
            简化问题:
            
            """,
            input_variables=["query"]
        )
















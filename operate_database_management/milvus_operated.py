"""
模型加载：
    milvus自身嵌入模型
    外部重排模型
功能操作：
    创建数据库
    创建集&加载集 collection
    创建Milvus集实体数据，插入集
    执行混合检索并重排序
"""
from pymilvus import DataType,AnnSearchRequest,WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from FlagEmbedding import FlagReranker
from base.config import config as conf
from base.logger import logger
from client_managed.milvus_client import MilvusClientManagement
from langchain_core.documents import Document
from pathlib import Path
import hashlib

print(Path(__file__))
length = len(Path(__file__).parents)
bge_m3_path = str(Path(__file__).parents[length-3] / "hugging_face_model" / "bge_m3" / "bge-m3" / "snapshots" / "bge-m3")
bge_rerank_large = str(Path(__file__).parents[length-3] / "hugging_face_model" / "bge_rerank_large" / "bge-reranker-large" / "snapshots" / "bge-large")


class OperateMilvus:
    def __init__(
            self,
            client:MilvusClientManagement,
            bge_m3_path:str,
            bge_rerank_large:str,
            collection_name:str=conf.milvus.collection_name,
            database_name:str=conf.milvus.database_name,
    ):
        self.client = client
        self.collection_name = collection_name
        self.database_name = database_name
        self.logger = logger
        self._connect_embed_model(bge_m3_path)
        self._connect_rerank_model(bge_rerank_large)
        self._create_database()
        self._create_collection()

    def _connect_embed_model(self,bge_m3_path):
        try:
            self.embed_model = BGEM3EmbeddingFunction(bge_m3_path,normalize_embeddings=True,use_fp16=False)
            self.dense_dim = self.embed_model.dim["dense"]
            logger.info(f"已加载嵌入模型：模型向量维度{self.dense_dim}")
        except Exception as e:
            logger.error(f"加载嵌入模型失败:{e}")


    def _connect_rerank_model(self,bge_rerank_large):
        try:

            self.reranker = FlagReranker(bge_rerank_large, use_fp16=False)
            # self.reranker = CrossEncoder(bge_rerank_large)
            logger.info(f"已加载重排序模型")
        except Exception as e:
            logger.error(f"加载重排模型失败:{e}")

    def _create_database(self):
        try:
            result = self.client.list_databases()
            if self.database_name not in result:
                self.client.create_database(db_name=self.database_name)
                logger.info(f"Milvus创建数据库{self.database_name}")
            self.client.use_database(db_name=self.database_name)
            logger.info(f"Milvus使用数据库{self.database_name}")
        except Exception as e:
            logger.error(f"Milvus创建数据库失败:{e}")

    def _create_collection(self):
        try:
            if not self.client.has_collection(self.collection_name):
                schema = self.client.create_schema(auto_id=False,enable_dynamic_field=True)
                schema.add_field(field_name="id",datatype=DataType.VARCHAR,is_primary=True,max_length=100)
                schema.add_field(field_name="text",datatype=DataType.VARCHAR,max_length=65535)
                schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=self.dense_dim)
                schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
                schema.add_field(field_name="parent_id",datatype=DataType.VARCHAR,max_length=100)
                schema.add_field(field_name="parent_content",datatype=DataType.VARCHAR,max_length=65535)
                schema.add_field(field_name="source",datatype=DataType.VARCHAR,max_length=50)
                schema.add_field(field_name="timestamp",datatype=DataType.VARCHAR,max_length=50)


                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name="dense_vector",
                    index_name="dense_index",
                    index_type="IVF_FLAT",
                    metric_type="IP",
                    params={"nlist":128}
                )

                index_params.add_index(
                    field_name="sparse_vector",
                    index_name="sparse_index",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="IP",
                    params={"drop_ratio_build":0.2}
                )

                self.client.create_collection(collection_name=self.collection_name,schema=schema,index_params=index_params)
                logger.info(f"Milvus已创建集:{self.collection_name}")
            else:
                logger.info(f"Milvus存在集:{self.collection_name}")
            self.client.load_collection(self.collection_name)
            logger.info(f"已加载{self.collection_name}集")
        except Exception as e:
            logger.error(f"Milvus创建集失败:{e}")


    def add_documents(self,documents:list[Document]):
        """对已加载的多个文档内容进行入库"""
        try:
            texts = [doc.page_content for doc in documents]
            embeddings = self.embed_model(texts)
            data = []
            for index,doc in enumerate(documents):
                text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                sparse_vector = {}
                embed_row = embeddings["sparse"]
                values = embed_row.data[embed_row.indptr[index]:embed_row.indptr[index+1]]
                indices = embed_row.indices[embed_row.indptr[index]:embed_row.indptr[index+1]]
                for index,value in zip(indices,values):
                    sparse_vector[index] = value
                data.append({
                    "id":text_hash,
                    "text":doc.page_content,
                    "dense_vector":embeddings["dense"][0],
                    "sparse_vector":sparse_vector,
                    "parent_id":doc.metadata["parent_id"],
                    "parent_content":doc.metadata["parent_content"],
                    "source":doc.metadata.get("source","unknown"),
                    "timestamp":doc.metadata.get("timestamp","unknown")
                })
            if data:
                self.client.insert(collection_name=self.collection_name,data=data)
                logger.info(f"已写入Milvus.{self.collection_name}:{len(data)}组数据")
        except Exception as e:
            logger.error(f"MilvusOperated：写入文档失败:{e}")


    def hybird_search_with_rerank(self,query:str,k:int=conf.retrieval.retrieval_k,source_filter:str=None):
        """混合检索，并排序"""
        try:
            query_embed = self.embed_model([query])
            query_dense = query_embed["dense"][0]
            query_sparse = query_embed["sparse"]
            query_data_row = query_sparse.data[query_sparse.indptr[0]:query_sparse.indptr[1]]
            query_indices_row = query_sparse.indices[query_sparse.indptr[0]:query_sparse.indptr[1]]
            sparse_vector = {}
            for index,value in zip(query_indices_row,query_data_row):
                sparse_vector[index] = value
            filter = f"source == '{source_filter}'" if source_filter else ""

            req_dense = AnnSearchRequest(
                data=[query_dense],
                limit=k,
                expr=filter,
                anns_field="dense_vector",
                param={"metric_type":"IP","params":{"nprobe":10}},
            )
            logger.info(f"请求Milvus稠密检索")

            req_sparse = AnnSearchRequest(
                data=[sparse_vector],
                limit=k,
                expr=filter,
                anns_field="sparse_vector",
                param={"metric_type":"IP"}
            )
            logger.info(f"请求Milvus稀疏检索")

            ranker = WeightedRanker(1.0,0.7)#稠密/稀疏
            result = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[req_dense,req_sparse],
                ranker=ranker,
                limit=k,
                output_fields=["text","parent_id","parent_content","source","timestamp"],
            )[0]
            logger.info(f"请求Milvus进行混合检索")
            logger.info(f"混合检索结果:{result}")
            sub_docs = [self._doc_from_hit(hit["entity"]) for hit in result ] #检索时设定output_field参数
            parent_docs = self._get_unique_parent_docs(sub_docs)
            if len(parent_docs)<2:
                return parent_docs[:conf.retrieval.candidate_m]
            if parent_docs:
                pairs = [(query,doc.page_content) for doc in parent_docs]
                parent_docs_score = self.reranker.compute_score(pairs)
                reranked_parent_docs = [doc for index,doc in sorted(zip(parent_docs_score,parent_docs),reverse=True)]
            else:
                reranked_parent_docs = []

            return reranked_parent_docs[:conf.retrieval.candidate_m]
        except Exception as e:
            logger.error(f"混合检索流程 检索失败：{e}")


    def _get_unique_parent_docs(self,sub_docs):
        parent_contents = set()
        unique_docs = []
        for chunk in sub_docs:
            parent_content = chunk.metadata.get("parent_content",chunk.page_content)
            if parent_content and parent_content not in parent_contents:
                parent_contents.add(parent_content)
                unique_docs.append(Document(page_content=chunk.page_content,metadata=chunk.metadata))
        return unique_docs

    def _doc_from_hit(self,hit):
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id":hit.get("parent_id"),
                "parent_content":hit.get("parent_content"),
                "source":hit.get("source"),
                "timestamp":hit.get("timestamp")
            }
        )


if __name__ == "__main__":

    from rag_process.core.document_processor import process_documents
    docs = process_documents(conf.syspath.databases)

    client = MilvusClientManagement()
    client.init()
    operate = OperateMilvus(client=client.client,bge_m3_path=bge_m3_path,bge_rerank_large=bge_rerank_large)
    embeds = operate.add_documents(docs)


































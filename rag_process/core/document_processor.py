"""
加载文档，获取文件，文件格式转换Document，文本切分

"""

from rag_process.document_loaders.json_loader import JsonLoader
from rag_process.document_loaders.markdown_loader import MarkdownLoader
from rag_process.document_loaders.pdf_loader import PdfLoader
from rag_process.document_loaders.text_loader import TextDocLoader
from rag_process.document_loaders.word_loader import WordDocumentLoader
from rag_process.document_loaders.csv_loader import CsvLoader


from rag_process.text_splitters.chinese_text_splitter import ChineseTextSplitter
from langchain_text_splitters import MarkdownTextSplitter

from base.config import config as conf
from base.logger import logger

from datetime import datetime
from pathlib import Path
from typing import Union
import os

document_loaders = {
    ".txt":TextDocLoader,
    ".md":MarkdownLoader,
    ".pdf":PdfLoader,
    ".docx":WordDocumentLoader,
    ".json":JsonLoader,
    ".csv":CsvLoader,
}


def _load_documents_from_directory(directory_path:Union[str,Path],structured_out:bool):
    """
    读取知识文件夹，文档加载
    :param directory_path: 文档路径
    :param structured_out: 文档加载是否需要结构化输出，还是整篇输出
    :return: 文档列表
    """
    documents = []
    supported_extensions = document_loaders.keys()
    source = os.path.basename(directory_path).split("_")[0]
    for root,_,files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension in supported_extensions:
                    try:
                        loader_class = document_loaders[file_extension]
                        if file_extension in [".md",".docx",".pdf"]:
                            doc_loader = loader_class(str(file_path),structured=structured_out)
                        else:
                            doc_loader = loader_class(str(file_path))
                        docs = doc_loader.load()
                        for doc in docs:
                            doc.metadata["source"] = source
                            doc.metadata["file_path"] = file_path
                            doc.metadata["timestamp"] = datetime.now().isoformat()
                        documents.extend(docs)
                        logger.info(f"成功加载文件：{file}")
                    except Exception as e:
                        logger.error(f"加载{file}文件失败:{e} \n路径：{file_path}")
                else:
                    logger.warning(f"不支持此类文件的加载:{file}")
    return documents


def process_documents(directory_path:Union[str,Path],
                     parent_chunk_size:int=conf.retrieval.parent_chunk_size,
                     child_chunk_size:int=conf.retrieval.child_chunk_size,
                     chunk_overlap:int=conf.retrieval.chunk_overlap,
                     ):
    """
    已加载文档进行切块操作
    :param directory_path: 文档路径
    :param parent_chunk_size: 父块大小
    :param sub_chunk_size: 子块大小
    :param chunk_overlap: 块重叠度
    :return:
    """
    documents = _load_documents_from_directory(directory_path,False)
    logger.info(f"路径{directory_path}:需要加载的文档数量{len(documents)}")
    #通用分割器【按设定的separator】
    parent_splitter = ChineseTextSplitter(chunk_size=parent_chunk_size,chunk_overlap=chunk_overlap)
    child_splitter = ChineseTextSplitter(chunk_size=child_chunk_size,chunk_overlap=chunk_overlap)
    #Markdown分割器【按标题层级】
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size,chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size,chunk_overlap=chunk_overlap)

    child_chunks = []
    for i,doc in enumerate(documents):
        file_path = doc.metadata.get("file_path")
        file_extension = os.path.splitext(file_path)[1].lower()
        parent_splitter_used = markdown_parent_splitter if file_extension == ".md" else parent_splitter
        child_splitter_used = markdown_child_splitter if file_extension == ".md" else child_splitter
        logger.info(f"开始处理文档切分:{file_path}，使用的分割器:{'Markdown_splitter' if file_extension == '.md' else 'Chinese_splitter'}")
        parent_docs = parent_splitter_used.split_documents([doc])
        for j,parent_doc in enumerate(parent_docs):
            parent_id = f"doc_{i}_parent_{j}"
            sub_chunks = child_splitter_used.split_documents([parent_doc])
            for k,sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata["parent_id"] = parent_id
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                child_chunks.append(sub_chunk)
    logger.info(f"子块数量：{len(child_chunks)}")
    return child_chunks


if __name__ == "__main__":
    import json
    from langchain_core.messages import message_to_dict,BaseMessage
    path_ = Path(__file__).parents[2] / "data" / "ai_data"

    l = process_documents(path_)
    for i in l:
        print(json.dumps(i,indent=4,ensure_ascii=False,default=lambda x:message_to_dict(x) if isinstance(x,BaseMessage) else str(x)))












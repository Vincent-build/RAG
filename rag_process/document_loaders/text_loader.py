from langchain_community.document_loaders import TextLoader
from base.logger import logger
from typing import Union
from pathlib import Path



class TextDocLoader:
    def __init__(self,file_path:Union[str,Path]):
        self.file_path = file_path

    def load(self):
        logger.info(f"开始Text文档加载")
        doc_loader = TextLoader(self.file_path,encoding='utf-8')
        docs = doc_loader.load()
        return docs

if __name__ == "__main__":
    from pathlib import Path
    length = len(Path(__file__).parents)
    doc_path = str(Path(__file__).parents[length - 3]  / "sample" / "rag.txt")
    doc_load = TextDocLoader(doc_path)
    docs = doc_load.document_load()
    for doc in docs:
        print(doc)






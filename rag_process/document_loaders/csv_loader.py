from langchain_community.document_loaders import CSVLoader
from base.logger import logger
from pathlib import Path
from typing import Union


class CsvLoader:
    def __init__(self,file_path:Union[str,Path]):
        self.file_path = file_path

    def load(self):
        logger.info(f"开始CSV文档加载")
        doc_loader = CSVLoader(self.file_path,encoding="utf-8")
        docs = doc_loader.load()
        return docs


if __name__ == "__main__":
    pass

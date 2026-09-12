from langchain_community.document_loaders import JSONLoader
from base.logger import logger
from pathlib import Path
from typing import Union

class JsonLoader:
    def __init__(self,file_path:Union[str,Path]):
        self.file_path = file_path

    def load(self):
        logger.info(f"开始JSON文档加载")
        try:
            doc_loader = JSONLoader(self.file_path,jq_schema=".")
            docs = doc_loader.load()
            return docs
        except Exception as e:
            print(f"加载失败:{str(e)[:60]}... \n正在重新加载")
            if "Set `text_content=False`" in str(e):
                doc_loader = JSONLoader(self.file_path,text_content=False,jq_schema=".")
                docs = doc_loader.load()
                return docs


if __name__ == "__main__":
    from pathlib import Path
    length = len(Path(__file__).parents)
    doc_path = Path(__file__).parents[length - 3] / "sample" / "sample.json"
    doc_load = JsonLoader(doc_path)
    docs = doc_load.document_load()
    for doc in docs:
        print(doc)













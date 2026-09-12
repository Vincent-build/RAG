from langchain_community.document_loaders import PyPDFLoader
from base.logger import logger
from typing import Annotated
from pathlib import Path
from typing import Union

class PdfLoader:
    def __init__(self,file_path:Union[str,Path],structured:Annotated[bool,"是否按页数结构输出"]):
        self.file_path = file_path
        self.structured = structured

    def _judge_mode(self):
        if self.structured:
            self.mode = "page"
        else:
            self.mode = "single"
        return self.mode

    def load(self):
        logger.info(f"开始PDF文档加载")
        mode = self._judge_mode()
        doc_loader = PyPDFLoader(self.file_path,mode=mode)
        docs = doc_loader.load()
        return docs


if __name__ == "__main__":
    from pathlib import Path
    length = len(Path(__file__).parents)
    doc_path = str(Path(__file__).parents[length - 3] / "sample" / "example.pdf")
    doc_load = PdfLoader(doc_path,True)
    docs = doc_load.document_load()
    for doc in docs:
        print(doc)
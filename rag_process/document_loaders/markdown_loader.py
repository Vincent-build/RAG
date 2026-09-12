from langchain_community.document_loaders import UnstructuredMarkdownLoader
from base.logger import logger
from typing import Annotated,Union
from pathlib import Path

class MarkdownLoader:
    """
    file_path:文档路径
    structured：True:按文本结构输出 False:整体输出
    """
    def __init__(self,file_path:Union[str,Path],structured:Annotated[bool,"是否按文本结构输出"]):
        self.file_path = file_path
        self.structured = structured
        self.mode = None

    def _judge_mode(self):
        if self.structured:
            self.mode = "elements"
        else:
            self.mode = "single"
        return self.mode

    def load(self):
        logger.info(f"开始Markdown文档加载")
        mode = self._judge_mode()
        doc_loader = UnstructuredMarkdownLoader(self.file_path,mode=mode)
        docs = doc_loader.load()
        return docs


if __name__ == "__main__":
    from pathlib import Path
    length = len(Path(__file__).parents)
    doc_path = Path(__file__).parents[length - 3] / "sample" / "sample.md"
    doc_load = MarkdownLoader(doc_path,False)
    docs = doc_load.document_load()
    for doc in docs:
        print(doc)
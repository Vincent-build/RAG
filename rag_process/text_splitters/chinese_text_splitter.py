from langchain_text_splitters import RecursiveCharacterTextSplitter
from base.logger import logger
from typing import Literal,Any,List
import re

def _split_text_with_regex_from_end(text:str,separator:str,keep_separator:bool)->List[str]:
    if separator:
        #文本块中保留分隔符
        if keep_separator:
            #不区分"start"和"end"
            splits_ = re.split(f"({separator})",text)
            splits = ["".join(i) for i in zip(splits_[0::2],splits_[1::2])]
            if len(splits_) % 2 == 1:
                splits += splits_[-1:]
        else:
            splits = re.split(separator,text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,  # noqa: FBT001,FBT002
        is_separator_regex: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        #中文优先级：段落 → 行 → 句号|叹号|问号 → 英文：句点|叹号|问好 → 分号 → 逗号
        self._separators = separators or [
            "\n\n",
            "\n",
            "。｜！｜？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s",
            " ",
            ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, s_ in enumerate(separators):
            separator_ = s_ if self._is_separator_regex else re.escape(s_)
            if not s_:
                separator = s_
                break
            if re.search(separator_, text):
                separator = s_
                new_separators = separators[i + 1 :]
                break

        separator_ = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(
            text, separator_, keep_separator=self._keep_separator
        )

        # Now go merging things, recursively splitting longer texts.
        good_splits = []
        separator_ = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator_)
                    final_chunks.extend(merged_text)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator_)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}","\n",chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]


if __name__ == "__main__":
    text_splitter = ChineseTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=150,
        chunk_overlap=10
    )


    for index,text in enumerate():
        print(index)
        chunks = text_splitter.split_text(text)
        print(chunks[1])
        for chunk in chunks:
            print(chunk)
            print("*" * 40)



































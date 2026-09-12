import jieba
import os
from pathlib import Path
from base.logger import logger

def preprocess_text(text):
    logger.info("文本预处理")
    try:
        return jieba.lcut(text.lower().replace(" ",""),)
    except Exception as e:
        print(f"文本预处理失败:{e}")
        return []

if __name__ == "__main__":
    print(preprocess_text("""精确模式‌：jieba.lcut("文本") 返回列表，切分最准确，适合文本分析 。
                            ‌全模式‌：jieba.lcut("文本", cut_all=True) 扫描所有可能词语，速度快但有冗余 。
‌搜索引擎模式‌：jieba.lcut_for_search("文本") 对长词再次切分，适合搜索场景 。
注意 jieba.cut() 返回生成器，需遍历查看；jieba.lcut() 直接返回列表更方便 。"""))





























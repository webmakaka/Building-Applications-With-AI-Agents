# pip install rank-bm25
from rank_bm25 import BM25Okapi
from typing import List
import string

# 定义一个简单的分词/清洗函数
def tokenizer(text: str) -> List[str]:
    # 1. 转小写
    # 2. 去除标点符号 (将标点替换为空)
    # 3. 按空格拆分
    clean_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return clean_text.split()

# 原始数据
raw_corpus = [
    "Agent J is the fresh recruit with attitude",
    "Agent K has years of MIB experience and a cool neuralyzer",
    "The galaxy is saved by two Agents in black suits",
]

# 1. 使用清洗函数处理 Corpus
tokenized_corpus = [tokenizer(doc) for doc in raw_corpus]

# 2. 建立索引
bm25 = BM25Okapi(tokenized_corpus)

# 3. 同样使用清洗函数处理 Query
query_str = "Who is a recruit?"
tokenized_query = tokenizer(query_str)

# 打印一下看看实际用于匹配的词
print(f"处理后的 Query tokens: {tokenized_query}")

# 4. 获取结果
top_n = bm25.get_top_n(tokenized_query, raw_corpus, n=2)

print("-------------------")
print(f"Query: {query_str}")
print("Top matching lines:")
for line in top_n:
    print(" •", line)


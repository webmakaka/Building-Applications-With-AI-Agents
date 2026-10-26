import tiktoken_ext.openai_public
import inspect
import hashlib

print(dir(tiktoken_ext.openai_public))
# The encoder we want is o200k_base, we see this as a possible function
# print(inspect.getsource(tiktoken_ext.openai_public.gpt2))
# The URL should be in the 'load_tiktoken_bpe function call'
print(inspect.getsource(tiktoken_ext.openai_public.o200k_base))
blobpath = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
print(cache_key)
# fb374d419588a4632f3f557e76b4b70aebbca790

"""
 $ cd /home/ubuntu/dataset/test
 $ mkdir tiktoken
 $ cd tiktoken
 $ wget https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
 $ mv o200k_base.tiktoken fb374d419588a4632f3f557e76b4b70aebbca790
 $ export TIKTOKEN_CACHE_DIR=/home/ubuntu/dataset/test/tiktoken

 conda create -n test python=3.12
 conda activate test
 pip install graphrag
 mkdir -p ./ragtest/input
 curl https://www.gutenberg.org/cache/epub/103/pg103.txt -o ./ragtest/input/book.txt
 cat ./ragtest/input/book.txt

 graphrag init --root ./ragtest
 graphrag index --root ./ragtest
 graphrag query \
 --root ./ragtest \
 --method global \
 --query "What are the key themes in this novel?"
 graphrag query \
 --root ./ragtest \
 --method local \
 --query "Who is Phileas Fogg and what motivates his journey?"
"""

from langchain.chat_models import init_chat_model

from utils.env_utils import DASHSCOPE_API_KEY, ALIBABA_BASE_URL, ZHIPU_API_KEY, ZHIPU_BASE_URL, OPENAI_API_KEY, \
    OPENAI_BASE_URL
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

llm = init_chat_model(
    model='qwen3.5-flash-2026-02-23',
    model_provider='openai',
    temperature=1.0,
    api_key=DASHSCOPE_API_KEY,
    base_url=ALIBABA_BASE_URL
)

glm = init_chat_model(
    model='glm-4.7-flash',
    model_provider='openai',
    temperature=1.0,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)

gpt = init_chat_model(
    model='gpt-4o-mini',
    model_provider='openai',
    temperature=1.0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)
bge_large = HuggingFaceEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs={
        'device': 'cuda',
        'local_files_only': True,   # 只使用本地文件
    },
    encode_kwargs={'normalize_embeddings': True}  # set True to compute cosine similarity
)


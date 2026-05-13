from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

from utils.env_utils import DASHSCOPE_API_KEY, ALIBABA_BASE_URL, ZHIPU_API_KEY, ZHIPU_BASE_URL, OPENAI_API_KEY, \
    OPENAI_BASE_URL
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2 / 3,  # <-- rpm=40
    check_every_n_seconds=0.1,  # 每隔多少秒检查一次令牌是否可用
)
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

nvidia = init_chat_model(
    model='deepseek-ai/deepseek-v4-pro',
    model_provider='nvidia',
    temperature=1.0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    reasoning_effort='max',
    rate_limiter=rate_limiter
)
bge_large = HuggingFaceEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs={
        'device': 'cuda',
        'local_files_only': True,  # 只使用本地文件
    },
    encode_kwargs={'normalize_embeddings': True}  # set True to compute cosine similarity
)

gme_st = SentenceTransformer("iic/gme-Qwen2-VL-2B-Instruct")
import gc
import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from ragas.embeddings.base import embedding_factory

from utils.env_utils import (
    DASHSCOPE_API_KEY, ALIBABA_BASE_URL,
    ZHIPU_API_KEY, ZHIPU_BASE_URL,
    OPENAI_API_KEY, OPENAI_BASE_URL,
    GME_MODEL_PATH, NVIDIA_API_KEY, NVIDIA_BASE_URL,
)
from utils.gme_inference import GmeQwen2VL

# ============ 轻量级资源（模块级安全） ============
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2 / 3,
    check_every_n_seconds=0.1,
)

llm = init_chat_model(
    model='qwen3.5-flash-2026-02-23',
    model_provider='openai',
    temperature=1.0,
    api_key=DASHSCOPE_API_KEY,
    base_url=ALIBABA_BASE_URL
)

multiModal_llm = init_chat_model(
    model='qwen3.5-omni-plus',
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
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
    rate_limiter=rate_limiter,
    model_kwargs={"reasoning_effort": "max"}
)



# ============ 大模型懒加载（单例） ============
_gme_model_instance = None

def get_gme_model():
    global _gme_model_instance
    if _gme_model_instance is None:
        _gme_model_instance = GmeQwen2VL(model_path=GME_MODEL_PATH)
    return _gme_model_instance

_bge_large_instance = None

def get_bge_large():
    global _bge_large_instance
    if _bge_large_instance is None:
        _bge_large_instance = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-zh-v1.5',
            model_kwargs={
                'device': 'cuda',
                'local_files_only': True,
            },
            encode_kwargs={'normalize_embeddings': True}
        )
    return _bge_large_instance

def unload_bge_model():
    global _bge_large_instance
    if _bge_large_instance is not None:
        # 如果模型内部有 .to('cpu') 等操作，可忽略；直接删除对象
        del _bge_large_instance
        _bge_large_instance = None
        gc.collect()
        # 如果之前使用了 GPU，清理显存缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 不再暴露 gme_model 变量！
# 其他模块导入时请使用：from my_llm import get_gme_model


# ============ 新版RAGAS评估框架，LLM配置 ============
from openai import AsyncOpenAI
from ragas.llms import llm_factory

client = AsyncOpenAI(
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL

)
embedding_client = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=ALIBABA_BASE_URL

)
ragas_llm = llm_factory("glm-4.7-flash", provider="openai", client=client,max_tokens=65536)
ragas_embedding= embedding_factory(model="tongyi-embedding-vision-plus-2026-03-06", provider="openai", client=embedding_client)
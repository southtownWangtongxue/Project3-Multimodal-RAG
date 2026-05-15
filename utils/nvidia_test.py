from langchain_nvidia_ai_endpoints import ChatNVIDIA

from utils.env_utils import NVIDIA_API_KEY

client = ChatNVIDIA(
    model="deepseek-ai/deepseek-v4-pro",
    api_key=NVIDIA_API_KEY,
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    extra_body={"chat_template_kwargs": {"thinking": False}},
)

for chunk in client.stream([{"role": "user", "content": "你好，你是谁"}]):
    print(chunk.content, end="")


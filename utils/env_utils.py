import os

from dotenv import load_dotenv

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

ALIBABA_BASE_URL = os.getenv("ALIBABA_BASE_URL")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL")
BAIDU_WEB_SEARCH_URL = os.getenv("BAIDU_WEB_SEARCH_URL")
MYSQL_CONNECTION_STRING = os.getenv("MYSQL_CONNECTION_STRING")

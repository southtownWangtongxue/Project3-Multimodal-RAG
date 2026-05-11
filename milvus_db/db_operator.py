import os
import random
import time
from typing import Dict, List

from langchain_core.documents import Document
from pymilvus import MilvusException

from milvus_db.collections_operator import COLLECTION_NAME, milvus_client
from utils.embeddings_utils import build_work_items, limiter, RETRY_ON_429, process_item_with_guard, MAX_429_RETRIES, \
    BASE_BACKOFF
from utils.log_utils import log


def doc_to_dict(docs: List[Document]) -> List[Dict]:
    """
        将 Document 对象列表转换为指定格式的字典列表。

        参数:
            doc_list: 包含 Document 对象的列表

        返回:
            list: 包含转换后字典的列表
        """
    result_list = []

    for doc in docs:
        # 初始化一个空字典来存储当前文档的信息
        doc_dict = {}
        metadata = doc.metadata

        # 1. 提取 text (仅当 embedding_type 为 'text' 时)
        if metadata.get('embedding_type') == 'text':
            doc_dict['text'] = doc.page_content
        else:
            doc_dict['text'] = None  # 或者设置为空字符串 ''，根据需要调整

        # 2. 提取 category (embedding_type)
        doc_dict['category'] = metadata.get('embedding_type', '')

        # 3. 提取 filename (source)
        source = metadata.get('source', '')
        doc_dict['filename'] = source

        # 4. 提取 filetype (source 中文件名的后缀)
        _, file_extension = os.path.splitext(source)
        doc_dict['filetype'] = file_extension.lower()  # 转换为小写，如 '.jpg'

        # 5. 提取 image_path (仅当 embedding_type 为 'image' 时)
        if metadata.get('embedding_type') == 'image':
            doc_dict['image_path'] = doc.page_content
        else:
            doc_dict['image_path'] = None  # 或者设置为空字符串 ''，根据需要调整

        # 6. 提取 title (拼接所有 Header 层级)
        headers = []
        # 假设 Header 的键可能为 'Header 1', 'Header 2', 'Header 3' 等，我们按层级顺序拼接
        # 我们需要先收集所有存在的 Header 键，并按层级排序
        header_keys = [key for key in metadata.keys() if key.startswith('Header')]
        # 按 Header 后的数字排序，确保层级顺序
        header_keys_sorted = sorted(header_keys, key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else x)

        for key in header_keys_sorted:
            value = metadata.get(key, '').strip()
            if value:  # 只添加非空的 Header 值
                headers.append(value)

        # 将所有非空的 Header 值用连字符或空格连接起来
        doc_dict['title'] = ' --> '.join(headers) if headers else ''  # 你也可以用其他连接符，如空格
        if not doc_dict['image_path']:
            doc_dict['text'] = doc_dict['title'] + ' ：' + doc_dict['text']
        # 将当前文档的字典添加到结果列表中
        result_list.append(doc_dict)

    return result_list

def save_to_milvus(processed_data:List[Dict]) :
    if not  processed_data:
        log.info("[Milvus] 没有可写入的数据。")
        return
    try:
        insert_result=milvus_client.insert(collection_name=COLLECTION_NAME, data=processed_data)
        log.info(f"[Milvus] 成功插入 {insert_result['insert_count']} 条记录。IDs 示例: {insert_result['ids'][:5]}")
    except MilvusException as e:
        log.exception(e)

def embedding_to_save(split_data:List[Document]) :
    """
        第一步：
        把Splitter之后的的数据（document对象列表），先转换为字典；
        第二步：
        把字典中的文本 和图片 ，进行向量化，然后再存入字典。
        第三步：
        最后写入向量数据库
        :param split_data:
        :return:
        """
    dicts=doc_to_dict(split_data)
    work_items=build_work_items(dicts)
    processed_data: List[Dict] = []
    for idx, (item, mode, api_img) in enumerate(work_items, start=1):
        limiter.acquire()

        if RETRY_ON_429:
            attempts = 0
            while True:
                result = process_item_with_guard(item.copy(), mode=mode, api_image=api_img)
                if result.get("dense"):
                    processed_data.append(result)
                    break
                attempts += 1
                if attempts > MAX_429_RETRIES:
                    log.warning(f"[429重试] 超过最大重试次数，跳过 idx={idx}, mode={mode}")
                    processed_data.append(result)
                    break
                backoff = BASE_BACKOFF * (2 ** (attempts - 1)) * (0.8 + random.random() * 0.4)
                log.warning(f"[429重试] 第{attempts}次，sleep {backoff:.2f}s …")
                time.sleep(backoff)
        else:
            result = process_item_with_guard(item.copy(), mode=mode, api_image=api_img)
            processed_data.append(result)

        if idx % 20 == 0:
            log.info(f"[进度] 已处理 {idx}/{len(work_items)}")
    # 第三步
    save_to_milvus(processed_data)
    return processed_data

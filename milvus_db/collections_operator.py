from pymilvus import MilvusClient, DataType, Function, FunctionType, IndexType
from pymilvus.client.types import MetricType

milvus_client = MilvusClient(uri="http://192.168.1.15:19530")
COLLECTION_NAME = 't_collection_Multimodal_RAG'


def create_collection():
    """
            创建 Milvus 数据库 collection 表

            该方法创建一个自定义的 Milvus collection，配置以下组件：
            - Schema：定义包含 id、text、category、source 等字段的数据结构
            - 分词器：使用 jieba 中文分词器对 text 字段进行分词
            - 向量字段：包含 dense(1024 维稠密向量) 和 sparse(稀疏向量) 两种向量类型
            - BM25 函数：自动从 text 字段生成稀疏向量用于全文检索
            - 索引配置：为 dense 向量建立 HNSW 索引，为 sparse 向量建立倒排索引

            如果 collection 已存在，会先释放并删除原有 collection 及其索引，然后重新创建

            Returns:
                None
            """
    # 创建数据模式并添加各个字段定义
    schema = milvus_client.create_schema()
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                     analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
    schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='image_path', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=1024)
    # 配置 BM25 函数，自动从 text 字段生成稀疏向量用于全文检索
    bm25_function = Function(
        name="text_bm25_emb",  # Function name
        input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
        output_field_names=["sparse"],  # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
        function_type=FunctionType.BM25,  # Set to `BM25`
    )
    schema.add_function(bm25_function)

    # 准备索引参数配置
    index_params = milvus_client.prepare_index_params()
    # 为 dense 向量字段配置 HNSW 索引，支持高效的近似最近邻搜索
    index_params.add_index(
        field_name="dense",
        index_name="dense_inverted_index",
        index_type=IndexType.HNSW,  # Inverted index type for sparse vectors
        metric_type=MetricType.IP,
        params={"M": 16, "efConstruction": 64}  # M :邻接节点数，efConstruction: 搜索范围
    )
    # 为 sparse 向量字段配置倒排索引，使用 BM25 相似度度量进行全文检索
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        },
    )

    # 检查 collection 是否已存在，若存在则先释放并删除
    if COLLECTION_NAME in milvus_client.list_collections():
        # 先释放，再删除索引，再删除 collection
        milvus_client.release_collection(collection_name=COLLECTION_NAME)
        milvus_client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_inverted_index')
        milvus_client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_inverted_index')
        milvus_client.drop_collection(collection_name=COLLECTION_NAME)

    # 创建新的 collection，应用配置好的 schema 和索引参数
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )


if __name__ == '__main__':
    create_collection()
    # 查看集合信息
    res = milvus_client.describe_collection(
        collection_name=COLLECTION_NAME
    )

    print(res)

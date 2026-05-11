from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker


class MilvusRetriever:
    def __init__(self, collection_name: str, milvus_client: MilvusClient, top_k: int = 5):
        self.collection_name = collection_name
        self.client: MilvusClient = milvus_client,
        self.top_k = top_k


    def dense_search(self, query_embedding, limit=5):
        """
        密集向量检索
        :param query_embedding: 已经向量后的内容
        :param limit:
        :return:
        """
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="dense",  # 密集向量中有图片和文本
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
            search_params=search_params,
        )
        return res[0]

    def sparse_search(self, query, limit=5):
        """
        稀疏向量搜索： 全文检索。
        :param query:  搜索的关键词文本
        :param limit:
        :return:
        """
        return self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="sparse",  # 全文检索： 只能检索文本
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
            search_params={"metric_type": "BM25", "params": {'drop_ratio_search': 0.2}},
        )[0]



    def hybrid_search(self, query_embedding,query, limit=5):
        """
            混合检索
            混合搜索是通过在hybrid_search() 函数中创建多个AnnSearchRequest 来实现的，
            其中每个AnnSearchRequest 代表一个特定向量场的基本 ANN 搜索请求。
            因此，在进行混合搜索之前，有必要为每个向量场创建一个AnnSearchRequest
            :return:
            """
        params1={
            "data": [query_embedding],
            "anns_field": "dense",
            "params": {"nprobe": 10},
            "limit": limit,
            "output_fields": ["text", 'category', 'filename', 'image_path', 'title'],
        }
        params2={
            "data": [query],
            "anns_field": "sparse",
            "params": {"nprobe": 10},
            "limit": limit,
            "output_fields": ["text", 'category', 'filename', 'image_path', 'title'],
        }
        dense_params=AnnSearchRequest(**params1)
        sparse_params=AnnSearchRequest(**params2)
        params = [dense_params, sparse_params]
        res=self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=params,
            ranker=RRFRanker(60)
        )
        return res[0]
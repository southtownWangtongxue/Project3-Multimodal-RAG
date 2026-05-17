"""
RAGAS 评估器
"""
import asyncio
from typing import Dict, List

from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference, LLMContextPrecisionWithoutReference
from ragas.metrics.collections import ContextRelevance

from milvus_db.collections_operator import COLLECTION_NAME, milvus_client
from milvus_db.db_retriever import MilvusRetriever
from my_llm import glm, get_bge_large
from utils.log_utils import log


def answer_generator(question: str, retrieve_context: List[Dict]) -> str:
    """
    根据问题和检索结果生成标准答案
    :param question:
    :param retrieve_context:
    :return:
    """
    context_str = "\n\n".join([f"上下文{i + 1}:{context['text']}" for i, context in enumerate(retrieve_context)])
    prompt = f"""
    你是一个AI助手，需要根据提供的上下文回答用户的问题。请确保回答基于提供的上下文，不用添加额外信息
    用户的问题：{question}
    
    检索到的上下文：{context_str}
    """
    res = glm.invoke(input=prompt)
    return res.content


class RagEvaluator:
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model

    async def evaluate_context(self, question: str, contexts: List[str]) -> float:
        """上下文相关性评估: 检索到的上下文（块或段落）是否与用户输入相关。"""
        # 0 → 检索到的上下文与用户查询完全不相关。
        # 1 → 上下文部分相关。
        # 2 → 上下文完全相关。
        # SingleTurnSample用于表示单轮对话的评估样本
        sample = SingleTurnSample(
            user_input=question,  # 用户输入的问题
            retrieved_contexts=contexts,  # 检索到的上下文
        )
        scorer = ContextRelevance(llm=self.llm)
        return await scorer.single_turn_ascore(sample)


    async def evaluate_answer(self, question: str, contexts: List[Dict], response: str) -> float:
        """评估生成的答案质量"""
        # SingleTurnSample用于表示单轮对话的评估样本
        sample = SingleTurnSample(
            user_input=question,  # 用户输入的问题
            retrieved_contexts=[context['text'] for context in contexts],  # 检索到的上下文
            response=response,  # 生成的答案
        )
        log.info(f"开始评估答案质量, 评估样本为：{sample}")
        scorer = ResponseRelevancy(llm=self.llm, embeddings=self.embedding_model)
        return await scorer.single_turn_ascore(sample)

    async def evaluate_metrics(self, question: str, contexts: List[Dict], response: str, reference: str=None):
        """
        评估RAG模型

        Args:
            question: 用户问题
            contexts: 检索到的上下文列表
            response: LLM生成的答案
            reference: 可选，参考答案 (用于评估的基准答案，通常为已知的正确答案)
        """
        # 1. 创建评估样本 (SingleTurnSample)
        sample = SingleTurnSample(
            user_input=question,  # 用户输入的问题
            retrieved_contexts=[context['text'] for context in contexts],  # 检索到的上下文
            response=response,  # 生成的答案
            reference=reference  # 参考答案 (用于需要参考答案的指标)
        )

        # 2. 初始化评估指标
        if reference:
            # 如果有参考答案，则初始化指标为LLMContextPrecisionWithReference
            context_precision = LLMContextPrecisionWithReference(llm=self.llm)
        else:
            # 如果没有参考答案，则初始化指标为LLMContextPrecisionWithoutReference
            context_precision = LLMContextPrecisionWithoutReference(llm=self.llm)

        # 3、执行评估指标得到结果
        context_precision_score = await context_precision.single_turn_ascore(sample)
        print(f"上下文精确度指标的 Score: {context_precision_score}")



async def main():
    ragas_llm = LangchainLLMWrapper(glm)
    ragas_embedding_model = LangchainEmbeddingsWrapper(get_bge_large())
    # 创建评估器
    rag_evaluator = RagEvaluator(ragas_llm, ragas_embedding_model)

    query='有界流和无界流有什么区别'
    retriever= MilvusRetriever(COLLECTION_NAME,milvus_client=milvus_client)
    retrieve_context=retriever.retrieve(query=query)
    response= answer_generator(question=query, retrieve_context=retrieve_context)
    log.info(f"生成的答案：{response}")
    await rag_evaluator.evaluate(question=query, retrieve_context=retrieve_context, response=response)


if __name__ == '__main__':
    asyncio.run(main())
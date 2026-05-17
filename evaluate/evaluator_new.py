"""
RAGAS 评估器
"""
import asyncio
from typing import Dict, List

from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy

from milvus_db.collections_operator import COLLECTION_NAME, milvus_client
from milvus_db.db_retriever import MilvusRetriever
from my_llm import glm, get_bge_large, ragas_llm, ragas_embedding
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

    async def evaluate(self, question: str, retrieve_context: List[Dict], response: str, reference: str = None):
        """
        评估函数
        :param question:用户问题
        :param retrieve_context:检索的上下文
        :param response:大模型生成的答案
        :param reference:（可选）用于参考的基准答案
        :return:
        """
        # 创建评估样本
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=[context['text'] for context in retrieve_context],
            response=response,
            reference=reference
        )
        # 初始化评估指标

        # 上下文精确率（Context Precision）是一个评估检索器能力的指标，用于衡量其在检索到的上下文中，为给定查询将相关内容块（chunks）排在不相关内容块之前的能力。
        context_precision = ContextPrecision(llm=self.llm)
        scorer = await context_precision.ascore(
            user_input=question,
            retrieved_contexts=[context['text'] for context in retrieve_context],
            reference=response
        )
        log.info(f"上下文精度指标分数：{scorer.value}")
        #上下文召回率（Context Recall）衡量的是检索到了多少相关的文档（或信息片段）。
        context_recall = ContextRecall(llm=self.llm)
        scorer = await context_recall.ascore(
            user_input=question,
            retrieved_contexts=[context['text'] for context in retrieve_context],
            reference=response
        )
        log.info(f"上下文召回率指标分数：{scorer.value}")
        # 忠实性 (Faithfulness) 指标衡量响应在事实上与检索到的上下文的一致程度。其取值范围为 0 到 1，分数越高表示一致性越好。
        faithfulness = Faithfulness(llm=self.llm)
        scorer = await faithfulness.ascore(
            user_input=question,
            retrieved_contexts=[context['text'] for context in retrieve_context],
            response=response
        )
        log.info(f"忠实性指标分数：{scorer.value}")
        # 回答相关性 (Answer Relevancy) 指标衡量响应与用户输入的相关程度。其取值范围为 0 到 1，分数越高表示与用户输入的对齐程度越好。
        answer_relevancy = AnswerRelevancy(llm=self.llm,embeddings=self.embedding_model)
        #  执行指标返回结果
        score = await answer_relevancy.ascore(
            user_input=question,
            response=response
        )
        log.info(f"回答相关性分数：{score.value}")
        return score


async def main():

    # 创建评估器
    rag_evaluator = RagEvaluator(ragas_llm,ragas_embedding)
    query='有界流和无界流有什么区别'
    retriever= MilvusRetriever(COLLECTION_NAME,milvus_client=milvus_client)
    retrieve_context=retriever.retrieve(query=query)
    response= answer_generator(question=query, retrieve_context=retrieve_context)
    log.info(f"生成的答案：{response}")
    await rag_evaluator.evaluate(question=query, retrieve_context=retrieve_context, response=response)


if __name__ == '__main__':
    asyncio.run(main())
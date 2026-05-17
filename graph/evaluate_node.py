from langchain_core.messages import AIMessage
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from graph.rag_state import MultiModalRAGState
from evaluate.evaluator import RagEvaluator
from my_llm import llm, get_bge_large
from utils.log_utils import log

evaluator_llm = LangchainLLMWrapper(llm)
embedding=get_bge_large()
evaluator_embeddings = LangchainEmbeddingsWrapper(embedding)

# 创建RAG评估器
rag_evaluator = RagEvaluator(evaluator_llm, evaluator_embeddings)

async def evaluate_answer(state: MultiModalRAGState):
    """评估大模型的响应和用户输入之间的相关性"""
    context_retrieved = state.get('context_retrieved')
    input_text = state.get('input_text')
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        answer = last_message.content
    score = await rag_evaluator.evaluate_answer(input_text, context_retrieved, answer)
    log.info(f"RAG Evaluation Score: {score}")
    return {"evaluate_score": float(score)}
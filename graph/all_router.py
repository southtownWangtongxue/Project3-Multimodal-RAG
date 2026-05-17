from langgraph.constants import END

from graph.rag_state import MultiModalRAGState
from utils.log_utils import log


def route_only_image(state: MultiModalRAGState):
    """
    动态路由函数，如果用户仅仅输入图片，则进入LLM节点，否则进入知识库检索节点
    """
    next_node='first_chatbot'
    if state.get('input_type') == 'only_image':
        next_node='retriever_node'
        log.info(f"下个节点：{next_node}")
        return next_node
    log.info(f"下个节点：{next_node}")
    return next_node

def route_llm_or_retriever(state: MultiModalRAGState):
    """
    动态路由函数，如果上下文检索到结果，则进入LLM节点，否则进入知识库检索节点
    """
    next_node = 'second_chatbot'
    if messages := state.get("messages", []):
        tool_message = messages[-1]
    else:
        raise ValueError("No message found in input")

    if not tool_message.content or tool_message.content == "没有找到相关的历史上下文信息。":
        next_node = 'retriever_node'
        log.info(f"下个节点：{next_node}")
        return next_node
    log.info(f"下个节点：{next_node}")
    return next_node



def route_evaluate_node(state: MultiModalRAGState):
    """
    动态路由函数，如果用户仅仅输入图片，则不进行评估（目前RAGAS还不支持多模态评估），其他情况下进入评估节点
    """
    next_node = 'evaluate_node'
    if state.get('input_type') == 'only_image':
        next_node =END
        log.info(f"下个节点：{next_node}")
        return next_node
    log.info(f"下个节点：{next_node}")
    return next_node

def route_human_node(state: MultiModalRAGState):
    """
    动态路由函数，如果评估后的分值低于0.7，则进入人工介入节点
    """
    next_node = 'human_approval'
    if state.get('evaluate_score') >= 0.7:
        next_node = END
        log.info(f"下个节点：{next_node}")
        return next_node
    log.info(f"下个节点：{next_node}")
    return next_node


def route_human_approval_node(state: MultiModalRAGState):
    """
    动态路由函数，如果用户输入的是：approve 则结束，否则进入网络搜索
    """
    next_node = 'fourth_chatbot'
    if state.get('human_answer') == 'approve':
        next_node = END
        log.info(f"下个节点：{next_node}")
        return next_node
    log.info(f"下个节点：{next_node}")
    return next_node


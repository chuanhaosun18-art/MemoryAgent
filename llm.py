"""
llm.py - DeepSeek LLM 封装
===========================

提供两个核心函数：
1. chat_with_tools() - 主对话循环用，支持 Function Calling，返回完整 choice 对象
2. chat_extract() - 偏好提取用，简单调用，低温度，返回纯文本

使用 OpenAI SDK 调用 DeepSeek API（兼容 OpenAI 格式）。
"""

from openai import OpenAI
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TEMPERATURE, MAX_TOKENS,
)

# 初始化 OpenAI 兼容客户端
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def chat_with_tools(messages, tools=None):
    """
    调用 DeepSeek API，支持 Function Calling。

    与搜索agent的 llm.py 不同，这里需要返回完整的 choice 对象，
    因为调用方需要检查 finish_reason 和 tool_calls 字段来决定下一步操作。

    参数:
        messages: 完整的消息列表（含 system prompt）
        tools: 工具定义列表（OpenAI 格式），为 None 时不传工具

    返回:
        ChatCompletion.choices[0] - 包含 message 和 finish_reason
    """
    kwargs = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(**kwargs)
    return response.choices[0]


def chat_extract(messages):
    """
    用于偏好提取的简单 LLM 调用。

    特点：
    - 不传工具（纯文本对话）
    - temperature=0.0（确定性输出，提取结果更稳定）
    - max_tokens=512（偏好描述很短，节省 token）

    参数:
        messages: 消息列表（通常只有 system + user 两条）

    返回:
        str: 模型回复的纯文本
    """
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()

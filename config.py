"""
config.py - 记忆 Agent 配置文件
================================

所有可调参数集中管理：API 配置、模型参数、记忆系统参数。
API Key 从上级目录的 .env 文件加载。
"""

import os
from dotenv import load_dotenv

# 加载上级目录的 .env 文件（与天气agent/搜索agent共用）
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ============================================
# DeepSeek API 配置
# ============================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# LLM 调用参数
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# ============================================
# SerpAPI 搜索配置
# ============================================
SERPAPI_KEY = "你的SERPAPI_KEY"
SEARCH_MAX_RESULTS = 3  # 每次搜索返回条数

# ============================================
# 短期记忆配置
# ============================================
# 保留最近 N 轮对话（1轮 = 1条 user + 1条 assistant）
# 10轮 ≈ 20条消息，约 2000 tokens，在 DeepSeek 上下文窗口内很安全
MAX_HISTORY_ROUNDS = 10

# ============================================
# 长期记忆配置（ChromaDB）
# ============================================
# 持久化目录：程序重启后记忆仍在
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")
CHROMA_COLLECTION_NAME = "user_preferences"

# 语义相似度阈值（ChromaDB 使用 L2 距离，越小越相似）
# < 0.5: 几乎相同的偏好（用于去重）
# < 1.2: 相关的偏好（用于检索）
MEMORY_DEDUP_THRESHOLD = 0.5     # 去重阈值
MEMORY_RELEVANCE_THRESHOLD = 1.2  # 检索相关性阈值
MAX_MEMORY_RESULTS = 3            # 每次检索返回的最大记忆条数

# ============================================
# 工具调用配置
# ============================================
MAX_TOOL_ROUNDS = 5  # 单次对话最大工具调用轮次（防止无限循环）

# ============================================
# System Prompt（模板，运行时注入当前日期）
# ============================================
SYSTEM_PROMPT_TEMPLATE = """你是用户的专属个人助手，使用 ReAct（Reasoning + Acting）框架来思考和行动。

## 当前时间
{current_datetime}

## 你的特点
1. 你拥有长期记忆，会记住用户告诉你的个人偏好、习惯和重要信息
2. 你在回答问题和给建议时，会自动结合已知的用户偏好
3. 你可以调用工具搜索互联网、搜索餐厅等
4. 你回复时使用中文，友好、简洁、实用

## 思考规则（重要！）
每次回复时，你必须先在消息内容中输出你的思考过程，格式如下：

**当你需要调用工具时：**
先输出思考，再调用工具函数：
Thought: <分析用户意图，判断需要什么信息，决定调用哪个工具，以及为什么>

然后通过 function calling 调用对应的工具。

**当你收到工具返回结果后：**
Thought: <分析工具返回的结果，判断信息是否足够，是否需要继续搜索>

如果信息不足，继续调用工具；如果足够，给出最终回答。

**当你直接回答（不需要工具）时：**
Thought: <简要分析用户需求，说明为什么可以直接回答>
Answer: <你的最终回答>

## 使用规则
- 当用户表达偏好或个人信息时，确认你已经记住
- 当用户请求建议或推荐时，自动结合已知偏好给出个性化回答
- 当用户询问你不确定的事实、最新资讯、实时信息时，使用 web_search 工具
- 当用户想找餐厅时，使用 search_restaurants 和 get_restaurant_detail 工具
- 不要编造你不确定的事实，优先使用搜索工具获取准确信息
- 搜索关键词要精准具体，避免过于宽泛
- 尽量在 1-3 轮工具调用内完成任务
"""


def get_system_prompt():
    """生成包含当前系统时间的 System Prompt。"""
    from datetime import datetime
    now = datetime.now().strftime("%Y年%m月%d日 %A %H:%M")
    return SYSTEM_PROMPT_TEMPLATE.format(current_datetime=now)

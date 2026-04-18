"""
memory.py - 记忆管理模块（核心）
================================

本模块是记忆 Agent 的核心创新点，实现两种记忆机制：

1. 短期记忆（Short-term Memory）
   - 数据结构：消息列表 self.short_term: list[dict]
   - 管理策略：滑动窗口，保留最近 N 轮对话
   - 作用：让 Agent 能理解上下文（"刚才说的那个餐厅"）

2. 长期记忆（Long-term Memory）
   - 数据结构：ChromaDB 向量数据库
   - 管理策略：语义检索 + 自动去重
   - 作用：跨会话记住用户偏好（"我不吃辣"→ 下次自动避开辣菜）

关键设计决策 - 记忆的读写时机：
=====================================

写入时机（偏好提取）：
  用户消息 → 关键词门控 → LLM 结构化提取 → ChromaDB 去重写入
  - 不是每条消息都触发，通过关键词门控节省 API 调用
  - 使用 LLM 提取（而非正则），因为自然语言表达多样

检索时机（记忆召回）：
  用户消息 → 是否需要推荐/建议？→ ChromaDB 语义检索 → 注入 system prompt
  - 寒暄类消息不检索，节省计算
  - 检索结果注入到 system prompt，让 LLM 自然地考虑偏好
"""

import json
import re
import hashlib
from datetime import datetime

import chromadb

from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
    MEMORY_DEDUP_THRESHOLD, MEMORY_RELEVANCE_THRESHOLD,
    MAX_MEMORY_RESULTS, MAX_HISTORY_ROUNDS, get_system_prompt,
)
from llm import chat_extract


# ============================================
# 关键词列表（用于门控，不需要精确，宁多勿少）
# ============================================

# 偏好提取门控关键词：消息中出现这些词时，可能包含个人偏好信息
PREFERENCE_KEYWORDS = [
    # 通用偏好表达
    "喜欢", "不喜欢", "讨厌", "偏好", "习惯", "爱好",
    "喜爱", "热爱", "不爱", "受不了", "不习惯",
    # 饮食相关
    "爱吃", "不吃", "不能吃", "过敏", "忌口", "口味",
    "素食", "清真", "减肥", "辣", "甜", "清淡",
    # 个人信息
    "我是", "我在", "我住", "我的工作", "我的职业",
    "我叫", "我姓", "我今年", "我的名字",
    # 生活习惯
    "早起", "晚睡", "熬夜", "运动", "健身", "跑步",
    "早睡", "作息", "锻炼",
    # 工作学习
    "工作", "学习", "编程", "开发", "写代码", "专业",
    "语言", "技术栈", "用的是",
    # 兴趣爱好
    "音乐", "电影", "游戏", "阅读", "旅游", "摄影",
    "书", "歌", "运动", "球",
    # 预算和消费
    "预算", "便宜", "贵", "省钱", "价格",
]

# 记忆检索门控关键词：消息中出现这些词时，需要检索长期记忆
RETRIEVAL_KEYWORDS = [
    "推荐", "建议", "帮我", "怎么", "如何", "什么",
    "哪里", "哪个", "哪家", "应该", "适合",
    "规划", "计划", "安排", "选择",
    "根据", "结合", "考虑", "记得", "之前说",
    "上次", "我说过",
]


class MemoryManager:
    """
    记忆管理器 - 统一管理短期和长期记忆

    使用示例:
        memory = MemoryManager()

        # 写入偏好
        memory.extract_and_store_preferences("我不能吃辣，肠胃不好")

        # 添加对话消息
        memory.add_user_message("推荐一家好吃的餐厅")

        # 构建包含记忆的消息（用于发送给 LLM）
        messages = memory.build_messages_with_memory("推荐一家好吃的餐厅")

        # 对话结束后裁剪
        memory.add_assistant_message("根据你的偏好，推荐...")
        memory.trim_history()
    """

    def __init__(self):
        """初始化短期记忆和长期记忆。"""

        # ---- 短期记忆：对话历史 ----
        # messages[0] 永远是 system prompt
        self.short_term = [
            {"role": "system", "content": get_system_prompt()}
        ]

        # ---- 长期记忆：ChromaDB 向量数据库 ----
        # PersistentClient 会在 CHROMA_PERSIST_DIR 下创建数据库文件
        # 程序重启后数据仍然保留（这就是"长期"记忆的含义）
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # get_or_create_collection：首次运行时创建，后续运行时复用
        # 内置 embedding 使用 all-MiniLM-L6-v2 模型（首次自动下载）
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},  # 使用 L2（欧几里得）距离度量
        )

    # ============================================
    # 短期记忆操作
    # ============================================

    def add_user_message(self, content):
        """将用户消息加入短期历史。"""
        self.short_term.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        """将助手回复加入短期历史。"""
        self.short_term.append({"role": "assistant", "content": content})

    def add_tool_call_messages(self, assistant_message, tool_results):
        """
        将一次 Function Calling 交互加入短期历史。

        一次工具调用涉及多条消息：
        1. assistant 消息（含 tool_calls 字段）
        2. 一条或多条 tool 角色的结果消息

        参数:
            assistant_message: LLM 返回的包含 tool_calls 的消息对象
            tool_results: tool 角色的结果消息列表
        """
        # 保存 assistant 消息（含 tool_calls）
        msg = {
            "role": "assistant",
            "content": assistant_message.content or "",
        }
        # 将 tool_calls 转换为可序列化的 dict
        if assistant_message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_message.tool_calls
            ]
        self.short_term.append(msg)

        # 保存每个工具结果
        for result in tool_results:
            self.short_term.append(result)

    def trim_history(self):
        """
        滑动窗口裁剪：保留 system prompt + 最近 N 轮对话。

        核心逻辑：
        1. messages[0] 是 system prompt，永远保留
        2. 从 messages[1:] 中计算"轮次"（每条 user 消息算一轮的开始）
        3. 只保留最后 MAX_HISTORY_ROUNDS 轮

        边界情况处理：
        - 不能拆分 tool_call 序列（assistant+tool_calls → tool → ... → assistant）
        - 如果裁剪后第一条非 system 消息是 tool 类型，继续往后删直到遇到 user
        """
        if len(self.short_term) <= 1:
            return

        messages = self.short_term[1:]  # 排除 system prompt

        # 找到所有 user 消息的索引（每个 user 消息是一轮的开始）
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]

        if len(user_indices) <= MAX_HISTORY_ROUNDS:
            return  # 还没超过限制，不需要裁剪

        # 保留最后 MAX_HISTORY_ROUNDS 轮
        # 从倒数第 N 个 user 消息的位置开始保留
        keep_from = user_indices[-MAX_HISTORY_ROUNDS]
        trimmed = messages[keep_from:]

        # 安全检查：确保第一条消息不是 tool 类型（避免孤立的工具结果）
        while trimmed and trimmed[0]["role"] == "tool":
            trimmed.pop(0)

        self.short_term = [self.short_term[0]] + trimmed

    # ============================================
    # 长期记忆操作 - 写入
    # ============================================

    def store_preference(self, preference_text, category, source_message):
        """
        将一条用户偏好写入 ChromaDB。

        去重逻辑：
        - 写入前先查询最相似的已有记录
        - 如果 L2 距离 < MEMORY_DEDUP_THRESHOLD（0.5），说明语义几乎相同
        - 此时 upsert（更新），避免"不吃辣"存三遍

        参数:
            preference_text: 偏好描述，如 "不喜欢辣的食物"
            category: 分类，如 "food_dislike"
            source_message: 原始用户消息，用于调试溯源
        """
        # 生成唯一 ID
        timestamp = datetime.now().isoformat()
        hash_suffix = hashlib.md5(preference_text.encode()).hexdigest()[:8]
        doc_id = f"pref_{hash_suffix}"

        metadata = {
            "category": category,
            "source_message": source_message[:200],  # 截断，避免过长
            "created_at": timestamp,
        }

        # 去重检查：查询最相似的已有记录
        existing = self.collection.query(
            query_texts=[preference_text],
            n_results=1,
        )

        if (existing["distances"]
                and existing["distances"][0]
                and existing["distances"][0][0] < MEMORY_DEDUP_THRESHOLD):
            # 语义高度相似的偏好已存在，更新它
            existing_id = existing["ids"][0][0]
            self.collection.update(
                ids=[existing_id],
                documents=[preference_text],
                metadatas=[metadata],
            )
            return f"更新已有记忆: {preference_text}"
        else:
            # 新偏好，添加
            self.collection.add(
                ids=[doc_id],
                documents=[preference_text],
                metadatas=[metadata],
            )
            return f"记住新偏好: {preference_text}"

    def extract_and_store_preferences(self, user_message):
        """
        检测用户消息中的偏好并存储到长期记忆。

        完整流程：
        1. 关键词门控 → 消息中是否包含饮食偏好相关关键词？
        2. LLM 提取 → 调用 chat_extract() 让模型结构化提取偏好
        3. JSON 解析 → 处理模型输出（可能带 markdown 代码块）
        4. 逐条存储 → 调用 store_preference() 写入 ChromaDB（自动去重）

        参数:
            user_message: 用户的原始消息

        返回:
            list[str]: 新记住的偏好列表（用于在界面上提示用户）
        """
        # 第一步：关键词门控（快速过滤，避免不必要的 LLM 调用）
        if not any(kw in user_message for kw in PREFERENCE_KEYWORDS):
            return []  # 消息中没有偏好相关关键词，跳过

        # 第二步：调用 LLM 提取偏好
        extraction_prompt = [
            {
                "role": "system",
                "content": (
                    "你是一个用户偏好提取器。分析用户消息，提取关于用户的个人偏好、习惯、身份信息。\n\n"
                    "如果包含个人信息或偏好，用 JSON 数组返回：\n"
                    '[{"preference": "偏好的简洁描述", "category": "分类"}]\n\n'
                    "分类只能是以下之一：\n"
                    "- personal_info: 个人身份信息（姓名、年龄、城市、职业等）\n"
                    "- food_preference: 饮食偏好（喜欢/不喜欢的食物、口味、过敏等）\n"
                    "- lifestyle: 生活习惯（作息、运动、日常习惯等）\n"
                    "- interest: 兴趣爱好（音乐、电影、游戏、运动等）\n"
                    "- work_study: 工作学习相关（技术栈、专业、工具偏好等）\n"
                    "- other: 其他个人偏好\n\n"
                    "如果消息不包含任何个人偏好信息，返回空数组：[]\n\n"
                    "只返回 JSON，不要其他文字。"
                ),
            },
            {
                "role": "user",
                "content": f"用户消息：{user_message}",
            },
        ]

        try:
            llm_response = chat_extract(extraction_prompt)
        except Exception as e:
            print(f"  [记忆] 偏好提取 LLM 调用失败: {e}")
            return []

        # 第三步：解析 JSON（处理 LLM 可能返回的 markdown 代码块包裹）
        preferences = self._parse_preferences_json(llm_response)

        if not preferences:
            return []

        # 第四步：逐条存储到 ChromaDB
        stored = []
        for pref in preferences:
            pref_text = pref.get("preference", "")
            category = pref.get("category", "other")
            if pref_text:
                result = self.store_preference(pref_text, category, user_message)
                stored.append(result)

        return stored

    def _parse_preferences_json(self, llm_response):
        """
        解析 LLM 返回的偏好 JSON。

        LLM 有时会用 markdown 代码块包裹 JSON，需要清理：
        ```json
        [{"preference": "...", "category": "..."}]
        ```

        返回:
            list[dict]: 解析后的偏好列表，解析失败返回 []
        """
        text = llm_response.strip()
        # 去掉 markdown 代码块标记
        text = re.sub(r"```json?\s*", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError:
            return []

    # ============================================
    # 长期记忆操作 - 检索
    # ============================================

    def retrieve_relevant_memories(self, query):
        """
        根据查询文本检索相关的长期记忆。

        使用 ChromaDB 的向量相似度搜索：
        1. query 被自动编码为向量（使用内置 embedding 模型）
        2. 在所有已存储的偏好中找到最相似的 N 条
        3. 过滤掉距离超过阈值的结果（不相关的不返回）

        参数:
            query: 查询文本（通常就是用户的消息）

        返回:
            list[str]: 相关偏好描述列表
        """
        # 如果数据库为空，直接返回
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(MAX_MEMORY_RESULTS, self.collection.count()),
        )

        # 过滤：只返回距离在阈值内的结果
        memories = []
        if results["documents"] and results["distances"]:
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                if dist < MEMORY_RELEVANCE_THRESHOLD:
                    memories.append(doc)

        return memories

    def should_retrieve_memory(self, user_message):
        """
        判断是否需要检索长期记忆。

        规则（宁可多检索也不要漏掉）：
        - 包含推荐/建议/餐厅等关键词 → True
        - 消息很短且是纯寒暄 → False
        - 数据库为空 → False（没有记忆可检索）
        - 默认 → True

        参数:
            user_message: 用户消息

        返回:
            bool
        """
        # 数据库为空，无需检索
        if self.collection.count() == 0:
            return False

        # 包含检索关键词，一定要检索
        if any(kw in user_message for kw in RETRIEVAL_KEYWORDS):
            return True

        # 很短的寒暄消息，不检索
        greetings = ["你好", "嗨", "hi", "hello", "谢谢", "感谢", "再见", "拜拜", "ok", "好的"]
        if user_message.strip().lower() in greetings:
            return False

        # 默认检索（宁可多检索一次）
        return True

    def get_all_preferences(self):
        """
        返回所有存储的用户偏好（用于 "记忆" 调试命令）。

        返回:
            list[dict]: 每条偏好的文本和分类
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.get()
        preferences = []
        for doc, metadata in zip(results["documents"], results["metadatas"]):
            preferences.append({
                "preference": doc,
                "category": metadata.get("category", "unknown"),
                "created_at": metadata.get("created_at", "unknown"),
            })
        return preferences

    # ============================================
    # 消息构建（整合短期 + 长期记忆）
    # ============================================

    def build_messages_with_memory(self, user_message):
        """
        构建包含记忆增强的消息列表，用于发送给 LLM。

        核心思路：
        - 基础 system prompt 始终存在
        - 如果检索到相关的长期记忆，动态追加到 system prompt 末尾
        - 短期历史消息保持不变

        返回的消息结构：
        [
            {"role": "system", "content": "基础prompt + 【用户偏好记忆】..."},
            ...短期历史消息（不含原始 system prompt）...
        ]

        参数:
            user_message: 当前用户消息（用于检索相关记忆）

        返回:
            list[dict]: 完整的消息列表
        """
        # 基础 system prompt
        system_content = get_system_prompt()

        # 检查是否需要检索长期记忆
        if self.should_retrieve_memory(user_message):
            memories = self.retrieve_relevant_memories(user_message)
            if memories:
                # 将检索到的偏好追加到 system prompt
                memory_section = "\n\n【用户记忆】\n"
                memory_section += "以下是你记住的关于用户的个人信息和偏好，请在回答时自动考虑：\n"
                for mem in memories:
                    memory_section += f"- {mem}\n"
                system_content += memory_section

        # 组装消息列表
        messages = [{"role": "system", "content": system_content}]
        # 加入短期历史（跳过 messages[0] 即原始 system prompt）
        messages.extend(self.short_term[1:])

        return messages

"""
agent.py - ReAct + Function Calling 记忆增强个人助手
====================================================

核心模块，将 ReAct 推理框架、Function Calling 工具调用和记忆管理融合。

ReAct + Function Calling 混合架构：
- Thought（思考）：LLM 在 message.content 中输出推理过程
- Action（行动）：LLM 通过 tool_calls 发起结构化工具调用
- Observation（观察）：工具执行结果作为 tool 消息反馈给 LLM
- Answer（回答）：LLM 不再调用工具时，content 中包含最终回答

每一步通过 on_step 回调实时推送给调用方（main.py），用于展示思考过程。
"""

import json
import re
from memory import MemoryManager
from llm import chat_with_tools
from tools import TOOLS, TOOL_FUNCTIONS
from config import MAX_TOOL_ROUNDS


def _parse_thought(content):
    """
    从 LLM 的 content 中提取 Thought 部分。

    LLM 被要求在 content 中以 "Thought: ..." 格式输出思考过程。
    如果没有匹配到格式，返回整段 content 作为思考内容。

    返回:
        tuple: (thought_text, answer_text)
            - thought_text: 思考部分
            - answer_text: Answer 部分（如果有，否则为 ""）
    """
    if not content:
        return "", ""

    # 提取 Thought 部分
    thought = ""
    answer = ""

    thought_match = re.search(r"Thought:\s*(.+?)(?=Answer:|$)", content, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
    else:
        # 没有严格格式，整段都当作思考
        thought = content.strip()

    # 提取 Answer 部分（如果有）
    answer_match = re.search(r"Answer:\s*(.+)", content, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    return thought, answer


def _truncate_observation(text, max_len=800):
    """压缩工具返回结果，减少传给 LLM 的 token 数。"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n...(结果已截断)"


class MemoryAgent:
    """
    ReAct + Function Calling 记忆增强个人助手

    核心能力：
    1. 短期记忆 - 滑动窗口对话历史（最近 10 轮）
    2. 长期记忆 - ChromaDB 向量存储用户偏好（跨会话持久化）
    3. ReAct 推理 - LLM 输出思考过程（Thought），可见推理链
    4. Function Calling - 结构化工具调用（web_search、餐厅搜索等）

    使用方式：
        def my_callback(step_type, content, step_num):
            print(f"[Step {step_num}] {step_type}: {content}")

        agent = MemoryAgent(on_step=my_callback)
        response, notes = agent.chat("今天北京天气怎么样")
    """

    def __init__(self, on_step=None):
        """
        初始化 Agent。

        参数:
            on_step: 回调函数 on_step(step_type, content, step_num)
                step_type 取值：
                    "thought"     - 模型的思考过程
                    "action"      - 工具调用（工具名 + 参数）
                    "observation" - 工具返回结果
                    "answer"      - 最终回答
                step_num: 当前是第几步（从 1 开始）
        """
        self.memory = MemoryManager()
        self.on_step = on_step or (lambda *_: None)

    def chat(self, user_input):
        """
        处理一轮用户输入，返回 Agent 的回复。

        完整流程：
            偏好提取 → 加入历史 → 检索记忆 → ReAct 循环（Thought→Action→Observation→...）→ 保存 → 裁剪

        参数:
            user_input: 用户的输入文本

        返回:
            tuple: (response_text, memory_notes)
                - response_text: Agent 的最终回复文本
                - memory_notes: 本轮新记住的偏好列表
        """

        # ======= 阶段1: 偏好提取（Memory Write） =======
        memory_notes = self.memory.extract_and_store_preferences(user_input)

        # ======= 阶段2: 加入短期历史 =======
        self.memory.add_user_message(user_input)

        # ======= 阶段3: 构建增强消息（Memory Read） =======
        messages = self.memory.build_messages_with_memory(user_input)

        # ======= 阶段4: ReAct + Function Calling 循环 =======
        step_num = 0
        response_text = ""

        for _ in range(MAX_TOOL_ROUNDS):
            choice = chat_with_tools(messages, tools=TOOLS)
            assistant_msg = choice.message
            content = assistant_msg.content or ""

            # 解析 Thought 和 Answer
            thought, answer = _parse_thought(content)

            # --- 没有工具调用 → 最终回答 ---
            if choice.finish_reason != "tool_calls" or not assistant_msg.tool_calls:
                if thought and not answer:
                    # 整段 content 就是最终回答（模型可能没用 Answer: 前缀）
                    # 检查是否有 Thought: 前缀，有则分开显示
                    if "Thought:" in content and thought != content.strip():
                        step_num += 1
                        self.on_step("thought", thought, step_num)
                    response_text = answer if answer else content.strip()
                    # 清除 response 中可能残留的 Thought:/Answer: 前缀
                    response_text = re.sub(r"^Thought:.*?(?=Answer:|$)", "", response_text, flags=re.DOTALL).strip()
                    response_text = re.sub(r"^Answer:\s*", "", response_text).strip()
                    if not response_text:
                        response_text = content.strip()
                elif answer:
                    step_num += 1
                    self.on_step("thought", thought, step_num)
                    response_text = answer
                else:
                    response_text = content.strip()
                break

            # --- 有工具调用 → Thought + Action + Observation ---
            step_num += 1

            # 输出 Thought
            if thought:
                self.on_step("thought", thought, step_num)

            # 执行每个工具调用
            tool_results = []
            for tool_call in assistant_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # 输出 Action（工具名 + 参数）
                args_str = ", ".join(f"{k}={v!r}" for k, v in func_args.items())
                self.on_step("action", f"{func_name}({args_str})", step_num)

                # 输出 tool_start（正在执行中…）
                self.on_step("tool_start", func_name, step_num)

                # 执行工具函数
                func = TOOL_FUNCTIONS.get(func_name)
                if func:
                    result = func(**func_args)
                else:
                    result = json.dumps({"error": f"未知工具: {func_name}"}, ensure_ascii=False)

                # 输出 Observation（传原始 JSON + 工具名，方便 main.py 格式化）
                self.on_step("observation", {
                    "tool": func_name,
                    "raw": result,
                }, step_num)

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # 将 assistant 消息（含 tool_calls）和工具结果加入消息列表
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ],
            })
            messages.extend(tool_results)

            # 同时保存到短期历史
            self.memory.add_tool_call_messages(assistant_msg, tool_results)

        # ======= 阶段5: 保存回复到短期历史 =======
        self.memory.add_assistant_message(response_text)

        # ======= 阶段6: 滑动窗口裁剪 =======
        self.memory.trim_history()

        return response_text, memory_notes

    def get_memories(self):
        """获取所有长期记忆（用于调试命令）。"""
        return self.memory.get_all_preferences()

    def get_history_length(self):
        """获取当前短期历史长度（用于调试）。"""
        return len(self.memory.short_term)

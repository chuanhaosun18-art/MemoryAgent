"""
main.py - ReAct 记忆增强个人助手 命令行入口
============================================

使用 rich 库提供彩色终端输出，实时展示 ReAct 推理过程：
  💭 Thought     - 模型思考（黄色）
  ⚡ Action      - 工具调用（蓝色）
  🔄 Searching   - 搜索进行中（动画）
  👁 Observation  - 工具结果（格式化展示）
  💬 Answer      - 最终回答（绿色面板）

特殊命令：
- "记忆" / "memory" : 查看所有存储的长期记忆
- "历史" / "history" : 查看当前短期历史长度
- "exit" / "quit" / "退出" : 退出程序
"""

import json

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from agent import MemoryAgent

console = Console()


def show_memories(agent):
    """显示所有长期记忆（调试/透明度功能）。"""
    memories = agent.get_memories()

    if not memories:
        console.print("[dim]还没有记住任何偏好。试试告诉我你的口味吧！[/dim]")
        return

    table = Table(title="长期记忆（用户偏好与信息）", border_style="cyan")
    table.add_column("偏好", style="bold")
    table.add_column("分类", style="dim")
    table.add_column("记录时间", style="dim")

    category_map = {
        "personal_info": "个人信息",
        "food_preference": "饮食偏好",
        "lifestyle": "生活习惯",
        "interest": "兴趣爱好",
        "work_study": "工作学习",
        "other": "其他",
    }

    for mem in memories:
        category = category_map.get(mem["category"], mem["category"])
        created = mem["created_at"][:16] if mem["created_at"] != "unknown" else "-"
        table.add_row(mem["preference"], category, created)

    console.print(table)


def show_history_info(agent):
    """显示短期历史信息。"""
    length = agent.get_history_length()
    console.print(f"[dim]短期历史中有 {length} 条消息（含 system prompt）[/dim]")


def _format_search_results(raw_json):
    """
    将 web_search 返回的 JSON 格式化为可读的搜索结果。

    输入: '[{"index":1,"title":"...","snippet":"...","link":"..."},...]'
    输出: 格式化的多行字符串
    """
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return raw_json  # 解析失败，原样返回

    # 错误或空结果
    if isinstance(data, dict):
        return data.get("error", data.get("message", str(data)))

    lines = []
    for item in data:
        idx = item.get("index", "")
        title = item.get("title", "无标题")
        snippet = item.get("snippet", "无摘要")
        link = item.get("link", "")
        lines.append(f"[{idx}] {title}")
        lines.append(f"    {snippet}")
        if link:
            lines.append(f"    🔗 {link}")
        lines.append("")  # 空行分隔

    return "\n".join(lines).rstrip()


def _format_restaurant_results(raw_json):
    """将餐厅搜索结果格式化为可读文本。"""
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return raw_json

    if isinstance(data, dict):
        # 单条餐厅详情或错误信息
        if "error" in data or "message" in data:
            return data.get("error", data.get("message", str(data)))
        name = data.get("name", "")
        cuisine = data.get("cuisine", "")
        price = data.get("price", "")
        rating = data.get("rating", "")
        features = ", ".join(data.get("features", []))
        spicy = data.get("spicy_level", "")
        seafood = data.get("has_seafood", "")
        veg = data.get("vegetarian_friendly", "")
        lines = [f"🏪 {name}  ({cuisine})"]
        lines.append(f"   💰 {price}  ⭐ {rating}  🌶️ {spicy}")
        if features:
            lines.append(f"   特色: {features}")
        if seafood:
            lines.append(f"   {seafood} | {veg}")
        return "\n".join(lines)

    # 多条餐厅列表
    lines = []
    for r in data:
        name = r.get("name", "")
        cuisine = r.get("cuisine", "")
        price = r.get("price", "")
        rating = r.get("rating", "")
        features = ", ".join(r.get("features", []))
        lines.append(f"🏪 {name}  ({cuisine})  💰 {price}  ⭐ {rating}")
        if features:
            lines.append(f"   特色: {features}")
    return "\n".join(lines)


# 工具名称的中文映射
TOOL_DISPLAY_NAMES = {
    "web_search": "🔍 Google 搜索",
    "search_restaurants": "🍽️ 餐厅搜索",
    "get_restaurant_detail": "📋 餐厅详情",
}


def on_react_step(step_type, content, step_num):
    """
    ReAct 步骤回调 —— 实时在终端展示模型的推理过程。

    参数:
        step_type: "thought" | "action" | "tool_start" | "observation"
        content: 该步骤的内容（字符串或字典）
        step_num: 第几步
    """
    if step_type == "thought":
        console.print(f"\n  [bold yellow]💭 Thought (Step {step_num})[/bold yellow]")
        for line in content.split("\n"):
            console.print(f"     [yellow]{line}[/yellow]")

    elif step_type == "action":
        console.print(f"  [bold blue]⚡ Action:[/bold blue] [blue]{content}[/blue]")

    elif step_type == "tool_start":
        tool_display = TOOL_DISPLAY_NAMES.get(content, content)
        console.print(f"  [bold magenta]🔄 正在执行 {tool_display} ...[/bold magenta]")

    elif step_type == "observation":
        # content 是 {"tool": "web_search", "raw": "...JSON..."}
        tool_name = content.get("tool", "")
        raw = content.get("raw", "")

        console.print(f"  [bold green]👁 Observation:[/bold green]")

        # 根据工具类型格式化输出
        if tool_name == "web_search":
            formatted = _format_search_results(raw)
        elif tool_name in ("search_restaurants", "get_restaurant_detail"):
            formatted = _format_restaurant_results(raw)
        else:
            formatted = raw

        # 用 Panel 包裹观察结果，清晰区分
        console.print(Panel(
            formatted,
            border_style="dim",
            padding=(0, 2),
        ))


def main():
    """命令行交互主循环。"""

    # 欢迎界面
    console.print(Panel(
        (
            "[bold cyan]记忆个人助手 (ReAct Mode)[/bold cyan]\n\n"
            "我是你的专属个人助手，使用 ReAct 推理框架，你可以看到我的完整思考过程！\n\n"
            "你可以：\n"
            "  - 告诉我关于你的信息：[green]\"我在北京工作\"[/green] [green]\"我不吃辣\"[/green]\n"
            "  - 让我帮你做事时自动考虑偏好：[green]\"帮我推荐周末活动\"[/green]\n"
            "  - 搜索互联网获取实时信息：[green]\"今天北京天气怎么样\"[/green]\n"
            "  - 搜索餐厅推荐：[green]\"帮我找一家日料餐厅\"[/green]\n"
            "  - 查看我记住了什么：输入 [bold]记忆[/bold]\n"
            "  - 退出：输入 [bold red]exit[/bold red]\n\n"
            "[dim]💭 Thought = 思考  ⚡ Action = 工具调用  🔄 执行中  👁 Observation = 结果[/dim]"
        ),
        title="欢迎",
        border_style="cyan",
    ))

    # 创建 Agent，注入 ReAct 步骤回调
    agent = MemoryAgent(on_step=on_react_step)

    # 检查是否有已保存的记忆
    existing_memories = agent.get_memories()
    if existing_memories:
        console.print(
            f"[dim]已从上次会话中恢复 {len(existing_memories)} 条偏好记忆。[/dim]"
        )

    # 主对话循环
    while True:
        try:
            user_input = console.input("\n[bold cyan]你> [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见！你的偏好已保存，下次见～[/dim]")
            break

        if not user_input:
            continue

        # 特殊命令处理
        if user_input.lower() in ("exit", "quit", "退出"):
            console.print("[dim]再见！你的偏好已保存，下次见～[/dim]")
            break

        if user_input in ("记忆", "memory"):
            show_memories(agent)
            continue

        if user_input in ("历史", "history"):
            show_history_info(agent)
            continue

        # 正常对话
        try:
            # agent.chat() 内部会通过 on_step 回调实时输出 Thought/Action/Observation
            response, memory_notes = agent.chat(user_input)

            # 显示新记住的偏好
            if memory_notes:
                for note in memory_notes:
                    console.print(f"  [dim italic]💾 {note}[/dim italic]")

            # 显示最终回答
            console.print(Panel(
                Markdown(response),
                title="[bold green]💬 Answer[/bold green]",
                border_style="green",
            ))

        except Exception as e:
            console.print(f"[bold red]错误:[/bold red] {e}")
            console.print("[dim]请检查网络连接和 API Key 配置。[/dim]")


if __name__ == "__main__":
    main()

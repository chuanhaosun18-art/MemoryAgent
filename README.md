# MemoryAgent - Memory-Enhanced Personal Assistant / 记忆增强个人助手

[English](#english) | [中文](#中文)

## English

A memory-enhanced personal assistant based on the ReAct (Reasoning + Acting) framework, using DeepSeek API, supporting short-term memory, long-term memory (ChromaDB vector storage), and tool calling.

### Features

- **Short-term Memory**: Sliding window keeps the last 10 rounds of conversation history
- **Long-term Memory**: ChromaDB vector database persistently stores user preferences across sessions (e.g., "I don't eat spicy food")
- **ReAct Reasoning**: Real-time display of model thinking process (💭 Thought → ⚡ Action → 👁 Observation)
- **Tool Calling**:
  - 🔍 Google Search (SerpAPI)
  - 🍽️ Restaurant Search & Details

### Quick Start

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Configure API Key

Create a `.env` file in the parent directory:
```
DEEPSEEK_API_KEY=your_DeepSeek_API_Key
```

#### Run

```bash
python main.py
```

### Project Structure

```
├── main.py      # CLI entry, Rich colored terminal output
├── agent.py     # ReAct + Function Calling core logic
├── memory.py    # Short/Long-term memory management (ChromaDB)
├── llm.py       # DeepSeek API wrapper
├── tools.py     # Tool functions (search, restaurants)
├── config.py    # Configuration file
└── chroma_data/ # ChromaDB persistent storage (ignored)
```

### Usage Example

```
You> I work in Beijing, I don't eat spicy food
💭 Thought: ...
⚡ Action: ...
👁 Observation: ...

You> Recommend a restaurant for me
→ Automatically considers your preferences, recommends non-spicy restaurants
```

### Special Commands

- `记忆` / `memory` - View stored long-term memories
- `历史` / `history` - View short-term history length
- `exit` / `quit` - Exit the program

### Tech Stack

- Python 3.10+
- DeepSeek API (OpenAI compatible)
- ChromaDB (Vector database)
- Rich (Terminal beautification)
- SerpAPI (Google Search)

### License

MIT


---

## 中文

基于 ReAct（Reasoning + Acting）框架的记忆增强个人助手，使用 DeepSeek API，支持短期记忆、长期记忆（ChromaDB 向量存储）和工具调用。

### 功能特点

- **短期记忆**：滑动窗口保留最近10轮对话历史
- **长期记忆**：ChromaDB 向量数据库持久化存储用户偏好，跨会话记住"我不吃辣"等偏好
- **ReAct 推理**：实时展示模型思考过程（💭 Thought → ⚡ Action → 👁 Observation）
- **工具调用**：
  - 🔍 Google 搜索（SerpAPI）
  - 🍽️ 餐厅搜索与详情查询

### 快速开始

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 配置 API Key

在上级目录创建 `.env` 文件：
```
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
```

#### 运行

```bash
python main.py
```

### 项目结构

```
├── main.py      # 命令行入口，Rich 彩色终端输出
├── agent.py     # ReAct + Function Calling 核心逻辑
├── memory.py    # 短期/长期记忆管理（ChromaDB）
├── llm.py       # DeepSeek API 封装
├── tools.py     # 工具函数（搜索、餐厅）
├── config.py    # 配置文件
└── chroma_data/ # ChromaDB 持久化存储（已忽略）
```

### 使用示例

```
你> 我在北京工作，不吃辣
💭 Thought: ...
⚡ Action: ...
👁 Observation: ...

你> 帮我推荐一家餐厅
→ 自动考虑用户偏好，推荐不辣的餐厅
```

### 特殊命令

- `记忆` / `memory` - 查看已存储的长期记忆
- `历史` / `history` - 查看短期历史长度
- `exit` / `quit` - 退出程序

### 技术栈

- Python 3.10+
- DeepSeek API (OpenAI 兼容)
- ChromaDB (向量数据库)
- Rich (终端美化)
- SerpAPI (Google 搜索)

### License

MIT

---


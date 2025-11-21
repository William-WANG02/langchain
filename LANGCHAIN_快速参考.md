# LangChain 快速参考手册 (Quick Reference)

> 本文档提供LangChain常用API和代码模式的快速查询

## 目录
- [快速开始](#快速开始)
- [核心组件速查](#核心组件速查)
- [常用模式](#常用模式)
- [代码速查表](#代码速查表)
- [性能优化](#性能优化)
- [调试技巧](#调试技巧)

---

## 快速开始

### 安装

```bash
# 基础安装
pip install langchain langchain-openai

# 常用集成
pip install langchain-community  # 社区集成
pip install langchain-anthropic  # Anthropic
pip install chromadb faiss-cpu   # 向量存储
pip install pypdf                # PDF处理
```

### 最小可运行示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 创建模型
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 创建prompt
prompt = ChatPromptTemplate.from_template("告诉我关于{topic}的故事")

# 组合并运行
chain = prompt | llm
result = chain.invoke({"topic": "人工智能"})
print(result.content)
```

---

## 核心组件速查

### 1. Models (模型)

#### Chat Models

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    timeout=30,
    max_retries=3
)

# Anthropic
llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0
)

# 调用方式
result = llm.invoke("问题")                    # 同步
result = await llm.ainvoke("问题")             # 异步
for chunk in llm.stream("问题"): ...          # 流式
results = llm.batch(["问1", "问2"])           # 批量
```

#### LLMs (文本生成模型)

```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
result = llm.invoke("完成这句话：人工智能是")
```

### 2. Messages (消息)

```python
from langchain_core.messages import (
    HumanMessage,      # 用户消息
    AIMessage,         # AI回复
    SystemMessage,     # 系统消息
    ToolMessage,       # 工具返回
)

messages = [
    SystemMessage(content="你是Python专家"),
    HumanMessage(content="什么是装饰器？"),
    AIMessage(content="装饰器是..."),
    HumanMessage(content="能举例吗？")
]

result = llm.invoke(messages)
```

### 3. Prompts (提示词)

#### 基础模板

```python
from langchain_core.prompts import PromptTemplate

# 字符串模板
prompt = PromptTemplate.from_template(
    "你是{role}，请回答：{question}"
)
prompt.invoke({"role": "老师", "question": "什么是Python?"})
```

#### Chat模板

```python
from langchain_core.prompts import ChatPromptTemplate

# 方式1：from_template
prompt = ChatPromptTemplate.from_template(
    "你是{role}，回答：{question}"
)

# 方式2：from_messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}"),
])

# 方式3：详细定义
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是{role}"),
    HumanMessagePromptTemplate.from_template("{question}"),
])
```

#### 部分变量

```python
# 预先填充某些变量
prompt = ChatPromptTemplate.from_template("你是{role}，{question}")
partial_prompt = prompt.partial(role="Python专家")

# 使用时只需提供question
result = partial_prompt.invoke({"question": "什么是装饰器?"})
```

### 4. Output Parsers (输出解析)

```python
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import (
    PydanticOutputParser,
    JsonOutputParser,
)
from pydantic import BaseModel, Field

# 字符串解析
parser = StrOutputParser()
chain = prompt | llm | parser  # 返回str

# Pydantic解析
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")

parser = PydanticOutputParser(pydantic_object=Person)
chain = prompt | llm | parser  # 返回Person对象

# JSON解析
parser = JsonOutputParser()
chain = prompt | llm | parser  # 返回dict
```

### 5. Document Loaders (文档加载)

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
)

# 文本文件
loader = TextLoader("file.txt")
docs = loader.load()

# PDF
loader = PyPDFLoader("file.pdf")
docs = loader.load()

# 网页
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# 目录
loader = DirectoryLoader("./docs", glob="**/*.txt")
docs = loader.load()
```

### 6. Text Splitters (文本分割)

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# 推荐：递归分割
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # chunk大小
    chunk_overlap=200,      # 重叠部分
    length_function=len,
    separators=["\n\n", "\n", "。", ".", " ", ""]
)

splits = splitter.split_documents(documents)

# 或从文本创建
splits = splitter.create_documents([text1, text2])
```

### 7. Embeddings (嵌入)

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# 嵌入文档
doc_vectors = embeddings.embed_documents([
    "文档1",
    "文档2"
])

# 嵌入查询
query_vector = embeddings.embed_query("查询文本")
```

### 8. Vector Stores (向量存储)

```python
from langchain_community.vectorstores import Chroma

# 创建
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 加载已存在的
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 添加文档
vectorstore.add_documents(new_docs)

# 搜索
docs = vectorstore.similarity_search("查询", k=5)
docs_with_scores = vectorstore.similarity_search_with_score("查询")

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",    # 或 "mmr"
    search_kwargs={"k": 5}
)
```

### 9. Retrievers (检索器)

```python
# 基础检索器
from langchain_core.retrievers import BaseRetriever

retriever = vectorstore.as_retriever(k=5)
docs = retriever.invoke("问题")

# 混合检索
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(documents)
ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.3, 0.7]
)

# 压缩检索
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

---

## 常用模式

### 模式1：基础链式调用

```python
# 简单链
chain = prompt | model | output_parser
result = chain.invoke(input_data)

# 带字典输入
chain = (
    {"topic": RunnablePassthrough()}
    | prompt
    | model
    | parser
)
```

### 模式2：并行处理

```python
from langchain_core.runnables import RunnableParallel

# 多个分支并行执行
parallel_chain = RunnableParallel(
    summary=summarize_chain,
    sentiment=sentiment_chain,
    keywords=keyword_chain
)

result = parallel_chain.invoke(input)
# result = {
#     "summary": "...",
#     "sentiment": "...",
#     "keywords": [...]
# }
```

### 模式3：条件分支

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "code" in x, code_chain),
    (lambda x: "math" in x, math_chain),
    default_chain  # 默认分支
)
```

### 模式4：RAG模式

```python
from langchain_core.runnables import RunnablePassthrough

# 标准RAG链
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 使用
answer = rag_chain.invoke("用户问题")
```

### 模式5：带记忆的对话

```python
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是AI助手"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = (
    {
        "input": RunnablePassthrough(),
        "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
    }
    | prompt
    | model
)

# 使用
response = chain.invoke("问题")
memory.save_context({"input": "问题"}, {"output": response.content})
```

### 模式6：Agent模式

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="搜索信息"
    )
]

# 创建agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用
result = agent_executor.invoke({"input": "任务描述"})
```

---

## 代码速查表

### 错误处理

```python
from langchain_core.runnables import RunnableRetry, RunnableFallback

# 重试
chain_with_retry = RunnableRetry(
    chain,
    max_attempts=3
)

# 降级
chain_with_fallback = chain.with_fallbacks([backup_chain])

# 超时
chain.invoke(input, config={"timeout": 30})
```

### 配置传递

```python
# 基础配置
config = {
    "callbacks": [callback_handler],
    "tags": ["production"],
    "metadata": {"user_id": "123"},
    "max_concurrency": 5,
}

result = chain.invoke(input, config=config)
```

### 流式输出

```python
# 同步流式
for chunk in chain.stream(input):
    print(chunk, end="", flush=True)

# 异步流式
async for chunk in chain.astream(input):
    print(chunk, end="", flush=True)
```

### 批量处理

```python
# 批量调用
results = chain.batch([input1, input2, input3])

# 带配置的批量
results = chain.batch(
    inputs,
    config={"max_concurrency": 5}
)
```

### 缓存

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# 内存缓存
set_llm_cache(InMemoryCache())

# SQLite缓存
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

### 回调

```python
from langchain.callbacks.base import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM开始: {prompts}")
    
    def on_llm_end(self, response, **kwargs):
        print(f"LLM结束: {response}")

# 使用
chain.invoke(input, config={"callbacks": [MyCallback()]})
```

---

## 性能优化

### 1. 使用流式输出

```python
# ❌ 等待完整响应
response = chain.invoke(input)

# ✅ 流式输出，降低首字节时间
for chunk in chain.stream(input):
    display(chunk)
```

### 2. 批量处理

```python
# ❌ 循环调用
results = [chain.invoke(inp) for inp in inputs]

# ✅ 批量调用
results = chain.batch(inputs, config={"max_concurrency": 5})
```

### 3. 并发处理

```python
import asyncio

# ✅ 异步并发
async def process_all():
    tasks = [chain.ainvoke(inp) for inp in inputs]
    return await asyncio.gather(*tasks)
```

### 4. 缓存

```python
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

# ✅ 启用缓存
set_llm_cache(SQLiteCache())
```

### 5. 限流

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,
    check_every_n_seconds=0.1,
)

llm = ChatOpenAI(rate_limiter=rate_limiter)
```

---

## 调试技巧

### 1. 启用详细日志

```python
import langchain
langchain.verbose = True

# 或在chain级别
chain = prompt | model
chain.invoke(input, config={"verbose": True})
```

### 2. 打印中间步骤

```python
from langchain_core.runnables import RunnablePassthrough

def debug_print(x):
    print(f"中间结果: {x}")
    return x

chain = (
    prompt
    | model
    | RunnablePassthrough(debug_print)  # 打印中间结果
    | parser
)
```

### 3. LangSmith追踪

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# 自动追踪所有链
result = chain.invoke(input)
```

### 4. 检查Prompt

```python
# 查看最终的prompt
messages = prompt.invoke({"role": "助手", "question": "问题"})
print(messages)
```

### 5. 计算Token

```python
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# 使用
prompt_text = prompt.format(role="助手", question="问题")
print(f"Token数: {count_tokens(prompt_text)}")
```

---

## 环境变量

### 常用环境变量

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"  # 可选

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="..."

# LangSmith
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="..."
export LANGCHAIN_PROJECT="project-name"

# 其他
export LANGCHAIN_VERBOSE=true
```

### 在代码中设置

```python
import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# 或使用python-dotenv
from dotenv import load_dotenv
load_dotenv()  # 从.env文件加载
```

---

## 常见错误及解决

### 错误1: Token超限

```python
# 问题
# This model's maximum context length is 4096 tokens...

# 解决
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,  # 留出空间
    chunk_overlap=200
)
```

### 错误2: Rate Limit

```python
# 问题
# Rate limit exceeded

# 解决1: 添加重试
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def call_with_retry():
    return chain.invoke(input)

# 解决2: 使用rate limiter
rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)
llm = ChatOpenAI(rate_limiter=rate_limiter)
```

### 错误3: API Key未设置

```python
# 问题
# No API key provided

# 解决
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# 或直接传入
llm = ChatOpenAI(api_key="sk-...")
```

---

## 有用的工具函数

### 1. 格式化文档

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

### 2. 提取JSON

```python
import json
import re

def extract_json(text: str) -> dict:
    # 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass
    
    # 提取代码块中的JSON
    match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    
    # 提取大括号内容
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    
    return {}
```

### 3. 清理文本

```python
def clean_text(text: str) -> str:
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:]', '', text)
    return text.strip()
```

### 4. 分割长列表

```python
def chunk_list(lst, chunk_size):
    """将列表分成固定大小的块"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# 使用
for batch in chunk_list(documents, 100):
    process(batch)
```

---

## 快速参考卡片

### LCEL 操作符

```python
chain1 | chain2          # 顺序执行
{"key": chain}           # 并行执行（字典输入）
RunnableParallel(...)    # 并行执行
RunnableBranch(...)      # 条件分支
chain.with_fallbacks([]) # 降级
chain.with_retry()       # 重试
```

### Runnable 方法

```python
.invoke(input)              # 单次调用
.ainvoke(input)             # 异步调用
.stream(input)              # 流式
.astream(input)             # 异步流式
.batch(inputs)              # 批量
.abatch(inputs)             # 异步批量
```

### 常用导入

```python
# 核心
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 模型
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

# 向量存储
from langchain_community.vectorstores import Chroma

# 文档处理
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Agent
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
```

---

## 更多资源

- **官方文档**: https://docs.langchain.com/
- **API参考**: https://reference.langchain.com/python
- **GitHub**: https://github.com/langchain-ai/langchain
- **社区**: https://forum.langchain.com
- **示例**: https://github.com/langchain-ai/langchain/tree/master/cookbook

---

**提示**: 
- 将此文档加入书签以便快速查询
- 结合官方文档使用效果更佳
- 遇到问题先查此文档，再查官方文档
- 实践是最好的学习方式

**更新时间**: 2024年11月
**适用版本**: LangChain 0.1.x+

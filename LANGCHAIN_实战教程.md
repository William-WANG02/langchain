# LangChain 实战教程 - 从零到项目

## 目录
1. [环境搭建](#环境搭建)
2. [基础实战](#基础实战)
3. [进阶项目](#进阶项目)
4. [生产环境最佳实践](#生产环境最佳实践)
5. [常见问题与解决方案](#常见问题与解决方案)

---

## 环境搭建

### 1. 安装依赖

```bash
# 基础安装
pip install langchain langchain-openai langchain-community

# 如果需要其他模型
pip install langchain-anthropic  # Anthropic Claude
pip install langchain-google-genai  # Google Gemini

# 向量存储
pip install chromadb  # Chroma
pip install faiss-cpu  # FAISS

# 文档处理
pip install pypdf  # PDF处理
pip install beautifulsoup4  # 网页处理
pip install langchain-text-splitters  # 文本分割

# 开发工具
pip install langsmith  # 可观测性
pip install python-dotenv  # 环境变量管理
```

### 2. 配置API密钥

创建 `.env` 文件：

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google
GOOGLE_API_KEY=...

# LangSmith (可选，用于追踪)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=my-project
```

在代码中加载：

```python
from dotenv import load_dotenv
import os

load_dotenv()

# 验证密钥已加载
print("OpenAI API Key:", os.getenv("OPENAI_API_KEY")[:10] + "...")
```

---

## 基础实战

### 项目1：智能问答助手

**目标**：创建一个能回答问题的AI助手

```python
# chat_assistant.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化模型
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# 2. 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI助手。请简洁、准确地回答问题。"),
    ("user", "{question}")
])

# 3. 创建输出解析器
output_parser = StrOutputParser()

# 4. 构建chain
chain = prompt | model | output_parser

# 5. 使用
if __name__ == "__main__":
    while True:
        question = input("\n你的问题: ")
        if question.lower() in ['退出', 'exit', 'quit']:
            break
        
        answer = chain.invoke({"question": question})
        print(f"\n回答: {answer}")
```

**运行效果：**
```
你的问题: 什么是机器学习？

回答: 机器学习是人工智能的一个分支，它让计算机系统能够从数据中学习
和改进，而无需被明确编程。通过算法分析数据模式，系统可以做出预测
或决策。
```

### 项目2：流式聊天机器人

**目标**：实现实时流式输出，提升用户体验

```python
# streaming_chat.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

class StreamingChatBot:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True  # 启用流式
        )
        
        self.chat_history = []
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的聊天机器人"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])
    
    def chat(self, user_input: str):
        """流式聊天"""
        print("\nAI: ", end="", flush=True)
        
        # 准备输入
        chain = self.prompt | self.model
        
        # 流式输出
        full_response = ""
        for chunk in chain.stream({
            "chat_history": self.chat_history,
            "input": user_input
        }):
            content = chunk.content
            print(content, end="", flush=True)
            full_response += content
        
        print()  # 换行
        
        # 保存历史
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=full_response))
        
        return full_response

# 使用
if __name__ == "__main__":
    bot = StreamingChatBot()
    
    print("聊天机器人已启动！输入 'exit' 退出")
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() == 'exit':
            break
        
        bot.chat(user_input)
```

### 项目3：结构化数据提取

**目标**：从文本中提取结构化信息

```python
# data_extraction.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# 1. 定义数据结构
class Person(BaseModel):
    name: str = Field(description="人名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")
    education: str = Field(description="学历")
    skills: list[str] = Field(description="技能列表")

# 2. 创建解析器
parser = PydanticOutputParser(pydantic_object=Person)

# 3. 创建提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个信息提取专家。从文本中提取人物信息。"),
    ("user", "{text}\n\n{format_instructions}")
])

# 4. 构建chain
model = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | model | parser

# 5. 测试
if __name__ == "__main__":
    text = """
    张三，今年28岁，毕业于清华大学计算机系，硕士学位。
    目前在一家互联网公司担任算法工程师，主要负责推荐系统的开发。
    擅长Python、机器学习、深度学习、自然语言处理等技术。
    """
    
    result = chain.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions()
    })
    
    print("提取结果：")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"职业: {result.occupation}")
    print(f"学历: {result.education}")
    print(f"技能: {', '.join(result.skills)}")
```

**输出：**
```
提取结果：
姓名: 张三
年龄: 28
职业: 算法工程师
学历: 硕士
技能: Python, 机器学习, 深度学习, 自然语言处理
```

---

## 进阶项目

### 项目4：企业知识库问答系统（RAG）

**目标**：构建基于公司文档的智能问答系统

#### 步骤1：文档处理

```python
# rag_system.py
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

class DocumentProcessor:
    def __init__(self, docs_directory: str):
        self.docs_directory = docs_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
    
    def load_documents(self):
        """加载文档"""
        documents = []
        
        # 加载PDF
        pdf_loader = DirectoryLoader(
            self.docs_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        # 加载文本文件
        txt_loader = DirectoryLoader(
            self.docs_directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents.extend(txt_loader.load())
        
        print(f"已加载 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents):
        """分割文档"""
        splits = self.text_splitter.split_documents(documents)
        print(f"分割成 {len(splits)} 个chunk")
        return splits
    
    def create_vectorstore(self, splits):
        """创建向量存储"""
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("向量存储创建完成")
        return vectorstore
```

#### 步骤2：RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGSystem:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # 返回最相似的5个chunk
        )
        
        # 创建prompt
        self.prompt = ChatPromptTemplate.from_template("""
你是一个专业的AI助手。请仅根据以下上下文信息回答问题。

上下文信息：
{context}

问题：{question}

要求：
1. 如果上下文中没有相关信息，请明确说明"根据提供的文档，我无法回答这个问题"
2. 回答要准确、简洁
3. 如果可以，请引用具体的文档内容

回答：
        """)
        
        # 创建model
        self.model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        # 构建chain
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )
    
    def _format_docs(self, docs):
        """格式化检索到的文档"""
        return "\n\n".join(
            f"文档{i+1}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )
    
    def query(self, question: str) -> dict:
        """查询"""
        # 获取相关文档
        relevant_docs = self.retriever.invoke(question)
        
        # 生成答案
        answer = self.chain.invoke(question)
        
        return {
            "answer": answer,
            "source_documents": relevant_docs
        }

# 完整使用流程
if __name__ == "__main__":
    # 1. 处理文档
    processor = DocumentProcessor("./company_docs")
    documents = processor.load_documents()
    splits = processor.split_documents(documents)
    vectorstore = processor.create_vectorstore(splits)
    
    # 2. 创建RAG系统
    rag = RAGSystem(vectorstore)
    
    # 3. 交互式问答
    print("\n知识库问答系统已启动！")
    while True:
        question = input("\n你的问题: ")
        if question.lower() == 'exit':
            break
        
        result = rag.query(question)
        
        print(f"\n答案: {result['answer']}")
        print(f"\n参考文档: {len(result['source_documents'])} 个相关文档")
        for i, doc in enumerate(result['source_documents'][:2]):
            print(f"\n文档{i+1}片段:")
            print(doc.page_content[:200] + "...")
```

### 项目5：多功能智能Agent

**目标**：创建能使用多种工具的智能代理

```python
# intelligent_agent.py
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
import requests
from datetime import datetime
import json

# 1. 定义工具
def search_web(query: str) -> str:
    """搜索网络信息（这里用示例，实际应该调用真实API）"""
    # 实际应该使用 SerpAPI 或其他搜索API
    return f"关于'{query}'的搜索结果：这是一个示例结果。"

def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间"""
    now = datetime.now()
    return f"当前时间是: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 注意：生产环境应该使用更安全的计算方法
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

def get_weather(city: str) -> str:
    """获取天气信息（示例）"""
    # 实际应该调用天气API
    weather_data = {
        "北京": "晴天，15-25度",
        "上海": "多云，18-26度",
        "深圳": "小雨，20-28度"
    }
    return weather_data.get(city, f"未找到{city}的天气信息")

def save_note(note: str) -> str:
    """保存笔记"""
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now()}] {note}\n")
    return "笔记已保存"

# 2. 创建工具列表
tools = [
    Tool(
        name="Search",
        func=search_web,
        description="搜索网络信息。输入应该是搜索查询字符串。"
    ),
    Tool(
        name="GetTime",
        func=get_current_time,
        description="获取当前时间。不需要输入参数。"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="进行数学计算。输入应该是数学表达式，例如：'2+2' 或 '10*5+3'"
    ),
    Tool(
        name="Weather",
        func=get_weather,
        description="获取城市天气。输入应该是城市名称，例如：'北京'"
    ),
    Tool(
        name="SaveNote",
        func=save_note,
        description="保存笔记到文件。输入应该是要保存的笔记内容。"
    )
]

# 3. 创建Agent
class IntelligentAgent:
    def __init__(self):
        # 创建LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0
        )
        
        # 创建prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个强大的AI助手，可以使用多种工具来帮助用户。

请遵循以下原则：
1. 仔细分析用户的需求
2. 选择合适的工具
3. 如果需要多个步骤，逐步执行
4. 给出清晰的答案

你可以使用的工具：
- Search: 搜索网络信息
- GetTime: 获取当前时间
- Calculator: 进行数学计算
- Weather: 查询天气
- SaveNote: 保存笔记
            """),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # 创建agent
        self.agent = create_openai_functions_agent(
            self.llm,
            tools,
            self.prompt
        )
        
        # 创建executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True,  # 显示思考过程
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def run(self, user_input: str) -> str:
        """执行任务"""
        try:
            result = self.agent_executor.invoke({"input": user_input})
            return result["output"]
        except Exception as e:
            return f"执行出错: {str(e)}"

# 4. 使用
if __name__ == "__main__":
    agent = IntelligentAgent()
    
    print("智能Agent已启动！输入 'exit' 退出")
    print("我可以帮你：搜索信息、查询时间/天气、计算、保存笔记等\n")
    
    # 测试用例
    test_queries = [
        "现在几点了？",
        "北京的天气怎么样？",
        "计算 123 * 456 + 789",
        "帮我保存一个笔记：明天要开会讨论项目进度",
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"用户: {query}")
        print(f"{'='*50}")
        result = agent.run(query)
        print(f"\nAgent: {result}\n")
```

### 项目6：对话式数据分析助手

**目标**：让用户用自然语言查询和分析数据

```python
# data_analyst_agent.py
from langchain.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

class DataAnalystAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # 创建pandas agent
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            verbose=True,
            agent_type="openai-functions",
            allow_dangerous_code=True  # 注意：生产环境需要额外安全措施
        )
    
    def query(self, question: str) -> str:
        """查询数据"""
        try:
            result = self.agent.invoke(question)
            return result["output"]
        except Exception as e:
            return f"查询出错: {str(e)}"

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    data = {
        '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
        '部门': ['技术', '销售', '技术', '人力', '销售'],
        '工资': [15000, 12000, 18000, 10000, 13000],
        '入职年份': [2020, 2021, 2019, 2022, 2021]
    }
    df = pd.DataFrame(data)
    
    print("数据预览：")
    print(df)
    print("\n" + "="*50)
    
    # 创建分析agent
    analyst = DataAnalystAgent(df)
    
    # 测试查询
    queries = [
        "平均工资是多少？",
        "技术部门有几个人？",
        "工资最高的是谁？",
        "按部门统计平均工资",
    ]
    
    for query in queries:
        print(f"\n问题: {query}")
        answer = analyst.query(query)
        print(f"回答: {answer}")
        print("-" * 50)
```

---

## 生产环境最佳实践

### 1. 错误处理和重试

```python
# production_best_practices.py
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableRetry
from langchain.callbacks.base import BaseCallbackHandler
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionCallbackHandler(BaseCallbackHandler):
    """生产环境回调处理器"""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        logger.info(f"LLM调用开始: {prompts[0][:100]}...")
    
    def on_llm_end(self, response, **kwargs):
        logger.info(f"LLM调用成功")
    
    def on_llm_error(self, error, **kwargs):
        logger.error(f"LLM调用失败: {str(error)}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        logger.info(f"Chain开始: {serialized.get('name', 'Unknown')}")
    
    def on_chain_error(self, error, **kwargs):
        logger.error(f"Chain失败: {str(error)}")

def create_production_chain():
    """创建生产级chain"""
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_retries=3,  # OpenAI SDK自带重试
        request_timeout=30  # 超时设置
    )
    
    # 添加额外的重试层
    model_with_retry = RunnableRetry(
        model,
        max_attempts=3,
        wait_exponential_jitter=True
    )
    
    return model_with_retry

# 使用
if __name__ == "__main__":
    chain = create_production_chain()
    
    try:
        result = chain.invoke(
            "测试问题",
            config={"callbacks": [ProductionCallbackHandler()]}
        )
        print(result.content)
    except Exception as e:
        logger.error(f"最终失败: {str(e)}")
```

### 2. 成本监控

```python
# cost_monitoring.py
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any
import tiktoken

class CostMonitoringCallback(BaseCallbackHandler):
    """监控Token使用和成本"""
    
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
        # GPT-3.5-turbo 价格 (美元/1K tokens)
        self.pricing = {
            "prompt": 0.0015,
            "completion": 0.002
        }
    
    def on_llm_end(self, response, **kwargs):
        """记录token使用"""
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens
    
    def get_cost_report(self) -> Dict[str, Any]:
        """获取成本报告"""
        prompt_cost = (self.prompt_tokens / 1000) * self.pricing["prompt"]
        completion_cost = (self.completion_tokens / 1000) * self.pricing["completion"]
        total_cost = prompt_cost + completion_cost
        
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_cost_usd": round(prompt_cost, 4),
            "completion_cost_usd": round(completion_cost, 4),
            "total_cost_usd": round(total_cost, 4)
        }

# 使用
cost_monitor = CostMonitoringCallback()

chain.invoke(
    input_data,
    config={"callbacks": [cost_monitor]}
)

report = cost_monitor.get_cost_report()
print(f"总Token: {report['total_tokens']}")
print(f"总成本: ${report['total_cost_usd']}")
```

### 3. 缓存策略

```python
# caching_strategy.py
from langchain.cache import SQLiteCache, InMemoryCache
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI
import hashlib
import json
from typing import Optional
import redis

class RedisCache:
    """Redis缓存实现"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = 3600  # 1小时过期
    
    def _make_key(self, prompt: str, **kwargs) -> str:
        """生成缓存键"""
        cache_str = f"{prompt}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """获取缓存"""
        key = self._make_key(prompt, **kwargs)
        value = self.redis_client.get(key)
        return value.decode() if value else None
    
    def set(self, prompt: str, response: str, **kwargs):
        """设置缓存"""
        key = self._make_key(prompt, **kwargs)
        self.redis_client.setex(key, self.ttl, response)

# 使用不同的缓存策略
def setup_caching(cache_type: str = "memory"):
    """配置缓存"""
    if cache_type == "memory":
        set_llm_cache(InMemoryCache())
    elif cache_type == "sqlite":
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    elif cache_type == "redis":
        # 需要自定义适配器
        pass

# 开发环境
setup_caching("memory")

# 生产环境
setup_caching("sqlite")
```

### 4. 安全性考虑

```python
# security_practices.py
import re
from typing import List

class SecurityFilter:
    """安全过滤器"""
    
    # 敏感信息模式
    PATTERNS = {
        'api_key': r'sk-[a-zA-Z0-9]{32,}',
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone': r'1[3-9]\d{9}',
        'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
        'id_card': r'\d{17}[\dXx]',
    }
    
    # 危险关键词
    DANGEROUS_KEYWORDS = [
        'delete', 'drop', 'truncate',  # SQL
        'rm -rf', 'format',  # 系统命令
        '__import__', 'exec', 'eval',  # Python危险函数
    ]
    
    @classmethod
    def sanitize_input(cls, text: str) -> str:
        """清理输入"""
        # 移除敏感信息
        for pattern_name, pattern in cls.PATTERNS.items():
            text = re.sub(pattern, f"[{pattern_name.upper()}_REDACTED]", text)
        
        return text
    
    @classmethod
    def validate_input(cls, text: str) -> bool:
        """验证输入是否安全"""
        text_lower = text.lower()
        
        # 检查危险关键词
        for keyword in cls.DANGEROUS_KEYWORDS:
            if keyword in text_lower:
                return False
        
        # 检查长度
        if len(text) > 10000:
            return False
        
        return True
    
    @classmethod
    def sanitize_output(cls, text: str) -> str:
        """清理输出"""
        # 移除可能的敏感信息
        return cls.sanitize_input(text)

def safe_invoke(chain, user_input: str):
    """安全的调用包装"""
    # 验证输入
    if not SecurityFilter.validate_input(user_input):
        return "输入包含不安全的内容，请重新输入"
    
    # 清理输入
    clean_input = SecurityFilter.sanitize_input(user_input)
    
    try:
        # 执行
        output = chain.invoke(clean_input)
        
        # 清理输出
        clean_output = SecurityFilter.sanitize_output(output)
        
        return clean_output
    except Exception as e:
        # 不要泄露详细错误信息
        logger.error(f"Error: {str(e)}")
        return "处理请求时发生错误，请稍后重试"
```

---

## 常见问题与解决方案

### 问题1：Token超出限制

**症状**：`This model's maximum context length is 4096 tokens...`

**解决方案：**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

# 方案1：分割输入
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,  # 留出空间给prompt和回复
    chunk_overlap=200
)
docs = text_splitter.create_documents([long_text])

# 方案2：使用Map-Reduce
# 对每个chunk并行处理，然后合并结果

# 方案3：使用长上下文模型
model = ChatOpenAI(model="gpt-4-turbo-preview")  # 128k context
```

### 问题2：API调用频率限制

**症状**：`Rate limit exceeded`

**解决方案：**

```python
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

# 创建限流器
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,  # 每秒0.5个请求
    check_every_n_seconds=0.1,
    max_bucket_size=10
)

# 应用到模型
model = ChatOpenAI(
    rate_limiter=rate_limiter
)

# 或使用exponential backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_with_retry():
    return chain.invoke(input)
```

### 问题3：响应时间过长

**解决方案：**

```python
# 1. 使用流式输出
for chunk in chain.stream(input):
    print(chunk, end="", flush=True)

# 2. 异步处理
import asyncio

async def process_multiple():
    tasks = [chain.ainvoke(inp) for inp in inputs]
    results = await asyncio.gather(*tasks)
    return results

# 3. 使用更快的模型
model = ChatOpenAI(model="gpt-3.5-turbo")  # 比gpt-4快

# 4. 缓存常见查询
set_llm_cache(SQLiteCache())
```

### 问题4：内存占用过高

**解决方案：**

```python
# 1. 限制对话历史长度
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=5,  # 只保留最近5轮对话
    return_messages=True
)

# 2. 使用摘要记忆
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(),
    return_messages=True
)

# 3. 清理向量存储
# 定期删除旧的embeddings

# 4. 使用生成器而不是列表
def process_documents():
    for doc in document_generator:
        yield process(doc)
```

### 问题5：向量检索不准确

**解决方案：**

```python
# 1. 改进文本分割策略
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 更小的chunk
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " "]  # 按语义分割
)

# 2. 使用更好的embedding模型
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 3. 混合搜索（关键词+向量）
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)

# 4. 重排序
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever
)

# 5. 添加元数据过滤
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 10,
        "filter": {"source": "official_docs"}
    }
)
```

---

## 项目示例总结

通过以上实战项目，你应该能够：

✅ **基础能力**
- 创建简单的问答系统
- 实现流式对话
- 提取结构化数据

✅ **进阶能力**
- 构建RAG知识库系统
- 开发多工具智能Agent
- 实现数据分析助手

✅ **生产能力**
- 处理错误和异常
- 监控成本和性能
- 实施安全措施
- 优化系统性能

### 下一步学习建议

1. **深入源码**：阅读感兴趣的组件源码
2. **实际项目**：将LangChain应用到真实业务场景
3. **性能优化**：学习如何优化大规模应用
4. **LangGraph**：学习更复杂的工作流编排
5. **LangSmith**：掌握监控和调试工具

### 推荐资源

- 官方文档：https://docs.langchain.com/
- API参考：https://reference.langchain.com/python
- GitHub：https://github.com/langchain-ai/langchain
- Discord社区：加入讨论和求助
- LangChain Blog：阅读最新最佳实践

祝你在LangChain开发之路上越走越远！💪

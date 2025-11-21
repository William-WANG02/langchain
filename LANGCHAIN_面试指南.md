# LangChain 面试准备指南 - 大模型算法工程师岗位

## 目录
1. [自我介绍模板](#自我介绍模板)
2. [项目介绍话术](#项目介绍话术)
3. [技术问题详解](#技术问题详解)
4. [场景设计题](#场景设计题)
5. [代码实现题](#代码实现题)
6. [行为面试准备](#行为面试准备)

---

## 自我介绍模板

### 基础版本（1-2分钟）

```
您好，我是[姓名]。我对大语言模型和AI应用开发非常感兴趣，
最近深入学习了LangChain框架。

在学习过程中，我：
1. 系统学习了LangChain的核心架构，包括Runnable抽象、LCEL表达式语言等
2. 实践了多个项目，包括RAG问答系统、智能Agent、对话机器人等
3. 深入理解了提示词工程、向量检索、Agent设计等核心技术
4. 关注生产环境的最佳实践，如错误处理、成本优化、性能调优

我能够独立使用LangChain开发完整的LLM应用，并且理解底层原理。
我期待能将所学应用到实际业务中，为团队创造价值。
```

### 技术深度版本（2-3分钟）

```
您好，我是[姓名]。我深入学习并实践了LangChain框架，
对大语言模型应用开发有系统性的理解。

技术深度方面：
- 掌握LangChain的核心设计模式，特别是Runnable接口的组合模式，
  理解如何通过管道操作符实现声明式编程
- 熟悉不同的语言模型抽象，能够快速集成OpenAI、Anthropic、
  开源模型等多种LLM
- 深入研究了RAG系统的实现原理，包括文档分割策略、嵌入向量选择、
  检索优化等关键技术点

实践项目经验：
1. 实现了企业知识库问答系统
   - 使用RecursiveCharacterTextSplitter优化文本分割
   - 采用混合检索（BM25+向量）提升准确率
   - 实现了上下文压缩和重排序
   
2. 开发了多工具智能Agent
   - 基于ReAct框架实现推理-行动循环
   - 集成了搜索、计算、数据查询等多种工具
   - 处理了工具调用失败、无限循环等边界情况

3. 关注生产环境质量
   - 实现了完整的错误处理和重试机制
   - 添加了Token使用监控和成本控制
   - 使用LangSmith进行链路追踪

我期待能够参与真实的LLM项目，解决实际业务问题。
```

---

## 项目介绍话术

### 项目1：企业知识库问答系统（RAG）

**项目背景：**
```
我开发了一个基于RAG的企业知识库问答系统，
目标是让用户能够用自然语言查询公司内部文档。
```

**技术架构：**
```
系统主要包含三个核心模块：

1. 文档处理模块
   - 支持PDF、Word、文本等多种格式
   - 使用RecursiveCharacterTextSplitter进行语义分割
   - 每个chunk大小1000字符，overlap 200字符
   - 为每个chunk生成元数据（来源、页码等）

2. 向量存储模块
   - 使用OpenAI的text-embedding-3-large生成向量
   - Chroma作为向量数据库，支持持久化存储
   - 实现了相似度搜索和元数据过滤

3. 检索生成模块
   - 采用混合检索策略（BM25 + 向量检索）
   - 权重比例3:7，平衡关键词和语义匹配
   - 使用ContextualCompression进行结果重排序
   - GPT-3.5-turbo生成答案，配置temperature=0保证稳定性
```

**技术亮点：**
```
1. 检索优化
   - 实现了两阶段检索：先召回20个候选，再精排到前5个
   - 使用LLM对检索结果进行相关性评分
   - 结果准确率从60%提升到85%

2. 性能优化
   - 实现了查询缓存，相同问题直接返回
   - 使用批量embedding降低API调用次数
   - 异步处理文档索引，不阻塞用户查询

3. 用户体验
   - 流式输出答案，降低等待时间
   - 显示引用来源，增强可信度
   - 支持追问功能，维护对话上下文
```

**遇到的挑战：**
```
挑战1：长文档处理
- 问题：单个文档超过100页，如何有效分割？
- 解决：采用递归分割，优先按章节分割，再按段落、句子分割
- 效果：保持了语义完整性，检索准确率提升20%

挑战2：多语言支持
- 问题：中英文混合文档，分词和检索效果差
- 解决：使用multilingual-e5-large模型
- 效果：跨语言检索F1分数从0.45提升到0.72

挑战3：实时性要求
- 问题：新文档需要立即可查询
- 解决：实现增量索引，新文档5分钟内可检索
- 效果：满足业务实时性要求
```

### 项目2：多工具智能Agent

**项目描述：**
```
开发了一个能够自主使用多种工具完成复杂任务的智能代理。
```

**核心功能：**
```
1. 工具集成
   - 网络搜索（SerpAPI）
   - 数学计算
   - 数据库查询（SQL）
   - 文件操作
   - API调用

2. 推理能力
   - 基于ReAct框架：Reasoning + Acting
   - 能够分解复杂任务为多个步骤
   - 根据中间结果调整执行策略

3. 错误处理
   - 工具调用失败自动重试
   - 检测并打断无限循环
   - 超过最大迭代次数时优雅降级
```

**技术实现：**
```python
# Agent执行流程伪代码
while not task_completed and iterations < max_iterations:
    # 1. Agent思考下一步
    thought = llm.generate(
        prompt=f"当前状态: {state}\n需要完成: {task}\n可用工具: {tools}"
    )
    
    # 2. 选择工具
    action = parse_action(thought)
    
    # 3. 执行工具
    observation = execute_tool(action.tool, action.input)
    
    # 4. 更新状态
    state.append((thought, action, observation))
    
    # 5. 检查是否完成
    if is_final_answer(thought):
        return extract_answer(thought)
    
    iterations += 1
```

**实际应用案例：**
```
场景：用户问"对比一下北京和上海的房价，并计算两者的差异百分比"

Agent执行过程：
1. Thought: 需要先获取两个城市的房价数据
2. Action: 使用Search工具查询"北京平均房价"
3. Observation: 北京平均房价约6万元/平米
4. Action: 使用Search工具查询"上海平均房价" 
5. Observation: 上海平均房价约5.5万元/平米
6. Thought: 已获取数据，需要计算差异百分比
7. Action: 使用Calculator工具计算"(60000-55000)/55000*100"
8. Observation: 9.09
9. Thought: 可以给出最终答案
10. Final Answer: 北京房价约6万元/平米，上海约5.5万元/平米，
                  北京比上海高约9.09%
```

---

## 技术问题详解

### Q1: 详细解释RAG系统的工作流程

**标准答案：**

```
RAG（Retrieval-Augmented Generation）系统分为两个阶段：

【离线阶段 - 文档索引】
1. 文档收集
   - 从各种来源获取文档（PDF、网页、数据库等）
   - 使用Document Loaders加载

2. 文本处理
   - 清洗文本（去除格式、特殊字符）
   - 使用Text Splitter分割成chunks
   - 选择合适的chunk_size和overlap

3. 向量化
   - 使用Embedding模型将文本转为向量
   - 常用模型：OpenAI embedding, sentence-transformers
   - 维度通常为768或1536

4. 存储
   - 保存到向量数据库（Chroma, Pinecone, Qdrant等）
   - 建立索引以支持快速检索

【在线阶段 - 查询响应】
1. 查询向量化
   - 用户输入问题
   - 使用相同的Embedding模型转为向量

2. 相似度检索
   - 在向量数据库中进行近似最近邻搜索（ANN）
   - 常用算法：HNSW, IVF
   - 返回top-k个最相似的chunks

3. 上下文构建
   - 将检索到的文档和用户问题组合
   - 构造完整的prompt

4. 生成答案
   - 将prompt发送给LLM
   - LLM基于上下文生成答案
   - 可选：引用来源、置信度评分

【关键技术点】
- Chunk大小权衡：太小缺少上下文，太大影响检索精度
- Embedding质量：直接影响检索效果
- 检索策略：相似度、MMR、混合检索等
- Prompt设计：如何有效组织上下文信息
```

### Q2: 如何优化RAG系统的检索准确率？

**深度回答：**

```
我会从多个维度优化：

【1. 文本分割优化】
- 语义分割：按段落、章节分割，保持语义完整
- 动态chunk：根据文本类型调整大小
  * 技术文档：chunk_size=1000
  * 对话记录：chunk_size=500
- 合适的overlap：通常设为chunk_size的10-20%

【2. Embedding优化】
- 选择更好的模型
  * 通用场景：text-embedding-3-large
  * 专业领域：微调embedding模型
  * 多语言：multilingual-e5
- 双塔模型：查询和文档使用不同encoder

【3. 检索策略优化】
方案A：混合检索
- BM25（关键词） + 向量检索（语义）
- 适合包含专业术语的查询

方案B：两阶段检索
- 第一阶段：快速召回top-100
- 第二阶段：LLM重排序到top-5
- 准确率提升但增加延迟

方案C：MMR（最大边际相关性）
- 在相关性和多样性之间平衡
- 避免返回重复内容

【4. 查询优化】
- Query扩展：生成相关查询词
- Query改写：让LLM优化用户查询
- 多查询：生成多个视角的查询，聚合结果

【5. 元数据过滤】
- 添加时间、类别、来源等元数据
- 先过滤再检索，提高精度

【6. 上下文优化】
- 上下文压缩：只保留相关句子
- 摘要增强：添加文档摘要
- 结构化信息：提取表格、列表等

【7. 评估和迭代】
指标：
- 召回率：相关文档是否被检索到
- 准确率：检索结果的相关性
- MRR：正确文档的排名

方法：
- 构建评估数据集（问题-答案-来源文档）
- A/B测试不同策略
- 收集用户反馈持续优化
```

### Q3: Agent可能遇到哪些问题？如何解决？

**系统性回答：**

```
【问题1：无限循环】
现象：Agent重复执行相同的行动
原因：
- LLM对当前状态判断不准确
- 工具返回结果不明确
- 缺少循环检测机制

解决方案：
1. 设置最大迭代次数限制
2. 检测重复的action-observation模式
3. 在prompt中明确要求"如果尝试3次仍失败，则说明无法完成"
4. 记录访问过的状态，避免重复

代码示例：
```python
class AgentExecutor:
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.seen_states = set()
    
    def detect_loop(self, current_state):
        state_hash = hash(str(current_state))
        if state_hash in self.seen_states:
            return True
        self.seen_states.add(state_hash)
        return False
```

【问题2：工具调用失败】
现象：工具执行报错或返回空结果
原因：
- 参数格式错误
- 外部API不可用
- 权限问题

解决方案：
1. 工具输入验证
```python
def validate_tool_input(tool_name, input_data):
    schema = TOOL_SCHEMAS[tool_name]
    if not validate(input_data, schema):
        return f"参数格式错误，期望：{schema}"
```

2. 优雅降级
```python
try:
    result = tool.run(input)
except APIError as e:
    result = f"工具暂时不可用：{e}。建议使用替代方案。"
```

3. 重试机制
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def call_tool_with_retry(tool, input):
    return tool.run(input)
```

【问题3：成本过高】
现象：复杂任务需要多次LLM调用，成本高
解决方案：
1. 使用更小的模型进行planning
2. 缓存常见的tool调用结果
3. 批量处理相似任务
4. 预先判断任务复杂度，简单任务不用Agent

【问题4：推理不准确】
现象：Agent选择错误的工具或给出错误答案
解决方案：
1. 改进prompt
   - 提供详细的工具描述
   - 给出正确的使用示例
   - 明确工具的适用场景

2. Few-shot examples
```python
examples = """
示例1：
问题：北京的天气如何？
思考：需要获取天气信息
行动：使用Weather工具，输入"北京"
结果：晴天，25度
...
"""
```

3. 使用更强的LLM（GPT-4而不是GPT-3.5）

4. 添加self-reflection
```python
# Agent反思自己的行动
reflection = llm.generate(
    f"我刚才的行动是{action}，结果是{observation}。"
    f"这个行动是否合理？是否需要调整？"
)
```

【问题5：多步推理错误传播】
现象：前面步骤的小错误导致最终答案完全错误
解决方案：
1. 中间结果验证
```python
def verify_intermediate_result(result, expected_format):
    if not matches(result, expected_format):
        return "结果格式不对，需要重新执行"
```

2. 提供纠错机会
```python
# 在最终答案前让Agent review
review_prompt = f"""
你之前的推理过程：
{reasoning_history}

最终答案：{final_answer}

请检查：
1. 每一步推理是否合理
2. 使用的数据是否正确
3. 最终答案是否回答了原始问题

如果发现问题，请指出并重新推理。
"""
```
```

### Q4: 如何设计一个高质量的Prompt？

**实战答案：**

```
好的Prompt应该包含以下要素：

【1. 角色定义（Role）】
明确AI的身份和专业领域
```
你是一个资深的Python专家，擅长代码优化和调试。
```

【2. 上下文信息（Context）】
提供必要的背景信息
```
用户是Python初学者，正在学习面向对象编程。
他们的代码运行报错。
```

【3. 任务描述（Task）】
清晰描述期望的输出
```
请帮助分析以下代码的错误，并提供修正建议。
```

【4. 约束条件（Constraints）】
限定输出格式和范围
```
要求：
1. 用简单易懂的语言解释
2. 提供修改后的完整代码
3. 解释为什么这样修改
4. 回答限制在300字以内
```

【5. 示例（Examples）】
提供输入输出示例（Few-shot）
```
示例1：
错误代码：lst.appen(1)
问题：appen应该是append
修正：lst.append(1)
解释：appen是拼写错误...
```

【6. 输出格式（Format）】
指定结构化输出
```
请按以下格式回答：
## 问题诊断
[错误原因]

## 修正代码
```python
[修改后的代码]
```

## 解释
[为什么这样改]
```

【完整示例】
```python
prompt_template = """
你是一个经验丰富的Python导师（Role）。

学生背景：刚开始学习Python，对语法不太熟悉（Context）

任务：分析以下代码错误并给出建议（Task）

代码：
```python
{code}
```

错误信息：
{error}

要求（Constraints）：
1. 用简单的语言解释错误原因
2. 提供修改后的正确代码
3. 给出学习建议
4. 语气要友好鼓励

输出格式（Format）：
## 🔍 问题分析
[简单解释错误]

## ✅ 正确代码
```python
[修改后代码]
```

## 💡 学习建议
[如何避免类似错误]

## 🌟 鼓励
[鼓励的话]

参考示例（Examples）：
[示例...]
"""
```

【高级技巧】
1. **Chain-of-Thought (CoT)**
```
让我们一步步思考：
1. 首先分析...
2. 然后考虑...
3. 最后得出...
```

2. **Self-Consistency**
生成多个答案，选择最一致的

3. **思维树（Tree of Thoughts）**
探索多个推理路径

4. **动态Prompt**
根据输入调整prompt内容
```python
def create_prompt(difficulty_level):
    if difficulty_level == "beginner":
        return "请用简单的语言解释..."
    else:
        return "请提供深入的技术分析..."
```

【常见错误】
❌ Prompt太模糊："帮我写代码"
✅ 具体明确："用Python写一个函数，输入列表，返回去重后的结果"

❌ 没有约束："介绍机器学习"（可能回答10000字）
✅ 有明确限制："用200字介绍机器学习的核心概念"

❌ 没有格式要求："提取信息"（返回格式不确定）
✅ 指定格式："以JSON格式返回，包含name、age、occupation字段"
```

---

## 场景设计题

### 场景1：设计一个智能客服系统

**问题：**
```
公司要开发一个智能客服系统，需要能够：
1. 回答常见问题（FAQ）
2. 查询订单状态
3. 处理投诉和建议
4. 无法处理时转人工

请设计技术方案。
```

**回答思路：**

```
【系统架构】

1. 意图识别层
   - 使用分类模型识别用户意图
   - 类别：FAQ查询、订单查询、投诉、闲聊、其他
   - 实现：LLM做zero-shot分类或训练专门的分类器

2. FAQ问答模块（RAG）
   - 将FAQ文档向量化存储
   - 用户问题检索相似问题
   - 返回标准答案
   - 添加"这个答案有帮助吗？"反馈

3. 订单查询模块（Agent+Tool）
   - Tool1: 订单查询API
   - Tool2: 物流查询API
   - Agent根据对话自动调用工具

4. 投诉处理模块
   - 记录投诉内容
   - 情感分析判断严重程度
   - 高优先级自动通知人工
   - 生成安抚性回复

5. 转人工策略
   触发条件：
   - 用户明确要求人工
   - 3次未能解决问题
   - 情绪负面且持续
   - 涉及敏感问题（退款、法律）

【技术实现】

```python
class IntelligentCustomerService:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.faq_system = FAQSystem()  # RAG
        self.order_agent = OrderAgent()  # Agent
        self.complaint_handler = ComplaintHandler()
        self.human_handoff = HumanHandoff()
    
    def process_message(self, user_message, session_id):
        # 1. 意图识别
        intent = self.intent_classifier.classify(user_message)
        
        # 2. 根据意图路由
        if intent == "faq":
            response = self.faq_system.answer(user_message)
            
        elif intent == "order_query":
            response = self.order_agent.handle(user_message, session_id)
            
        elif intent == "complaint":
            response = self.complaint_handler.handle(user_message)
            
            # 检查是否需要转人工
            if self.should_handoff(user_message, session_id):
                return self.human_handoff.transfer(session_id)
        
        else:
            # 默认用通用对话处理
            response = self.general_chat(user_message)
        
        # 3. 记录对话
        self.save_conversation(session_id, user_message, response)
        
        return response
    
    def should_handoff(self, message, session_id):
        # 转人工判断逻辑
        history = self.get_conversation_history(session_id)
        
        # 规则1：用户主动要求
        if "人工" in message or "转人工" in message:
            return True
        
        # 规则2：连续失败
        if self.consecutive_failures(history) >= 3:
            return True
        
        # 规则3：情绪分析
        sentiment = self.analyze_sentiment(message)
        if sentiment < -0.5:  # 非常负面
            return True
        
        return False
```

【关键考虑点】

1. 性能优化
   - FAQ检索缓存热门问题
   - 意图分类使用小模型（快速）
   - 异步处理降低延迟

2. 准确性保障
   - FAQ答案需要定期更新
   - 订单信息需要实时同步
   - 错误答案收集和纠正

3. 用户体验
   - 流式输出减少等待感
   - 快速响应（<2秒）
   - 对话自然不生硬

4. 安全性
   - 订单查询需要身份验证
   - 敏感信息脱敏
   - 防止prompt注入攻击

5. 可扩展性
   - 模块化设计易于添加新功能
   - 支持多渠道（网页、微信、APP）
   - 支持多语言

【评估指标】
- 自动解决率：>80%
- 平均响应时间：<2秒
- 用户满意度：>4.5/5
- 转人工率：<15%
```

### 场景2：如何构建代码审查助手

**问题：**
```
团队需要一个AI代码审查助手，能够：
- 检查代码质量问题
- 给出改进建议
- 检测安全漏洞
请设计方案。
```

**回答：**

```
【系统设计】

1. 静态分析层
   - 使用工具：pylint, flake8, mypy
   - 检查：语法错误、类型错误、风格问题
   - 自动化执行，无需LLM

2. LLM审查层
   - 代码理解：分析代码逻辑和设计
   - 质量评估：可读性、可维护性、性能
   - 建议生成：具体的改进建议

3. 安全检查层
   - 使用工具：bandit（Python安全检查）
   - LLM辅助：识别逻辑漏洞
   - 检查项：SQL注入、XSS、敏感数据泄露

【实现方案】

```python
class CodeReviewAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.static_analyzers = {
            'pylint': PylintAnalyzer(),
            'bandit': BanditAnalyzer(),
        }
    
    def review_code(self, code: str, language: str = "python"):
        report = {
            "static_analysis": [],
            "quality_review": {},
            "security_issues": [],
            "suggestions": []
        }
        
        # 1. 静态分析
        for name, analyzer in self.static_analyzers.items():
            issues = analyzer.analyze(code)
            report["static_analysis"].extend(issues)
        
        # 2. LLM质量审查
        quality_review = self.llm_quality_review(code)
        report["quality_review"] = quality_review
        
        # 3. 安全检查
        security_issues = self.security_check(code)
        report["security_issues"] = security_issues
        
        # 4. 生成改进建议
        suggestions = self.generate_suggestions(report)
        report["suggestions"] = suggestions
        
        return report
    
    def llm_quality_review(self, code):
        prompt = f"""
作为资深代码审查专家，请评估以下代码：

```python
{code}
```

请从以下维度评分（1-10）并说明理由：
1. 可读性：代码是否易于理解
2. 可维护性：是否易于修改和扩展  
3. 性能：是否有性能问题
4. 设计：是否遵循最佳实践
5. 测试性：是否易于测试

请以JSON格式返回。
        """
        
        response = self.llm.invoke(prompt)
        return parse_json(response.content)
    
    def generate_suggestions(self, report):
        # 基于分析结果生成具体建议
        prompt = f"""
基于以下代码审查报告，生成具体的改进建议：

{json.dumps(report, indent=2)}

要求：
1. 优先级排序（高、中、低）
2. 每条建议包含：问题描述、改进方案、示例代码
3. 最多5条最重要的建议

格式：
## 建议1：[标题]
**优先级**：高/中/低
**问题**：[具体问题]
**建议**：[如何改进]
**示例**：
```python
[改进后的代码]
```
        """
        
        response = self.llm.invoke(prompt)
        return response.content

# 使用示例
reviewer = CodeReviewAssistant()

code = """
def process_user_data(data):
    # 未验证输入
    user_id = data['user_id']
    # SQL注入风险
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = execute_query(query)
    return result
"""

report = reviewer.review_code(code)
print(json.dumps(report, indent=2, ensure_ascii=False))
```

【输出示例】

```json
{
  "static_analysis": [
    {
      "line": 3,
      "type": "error",
      "message": "Possible KeyError if 'user_id' not in data"
    }
  ],
  "quality_review": {
    "readability": {
      "score": 6,
      "reason": "缺少类型注解和文档字符串"
    },
    "maintainability": {
      "score": 4,
      "reason": "硬编码SQL，难以维护"
    },
    "performance": {
      "score": 7,
      "reason": "性能可接受"
    }
  },
  "security_issues": [
    {
      "severity": "HIGH",
      "type": "SQL Injection",
      "line": 5,
      "description": "直接拼接用户输入到SQL语句"
    }
  ],
  "suggestions": [
    {
      "priority": "HIGH",
      "title": "修复SQL注入漏洞",
      "problem": "使用f-string直接拼接用户输入",
      "solution": "使用参数化查询",
      "example": "query = 'SELECT * FROM users WHERE id = %s'\nresult = execute_query(query, (user_id,))"
    }
  ]
}
```

【优化方向】
1. 增量审查：只审查变更的代码
2. 上下文理解：结合整个项目结构
3. 学习反馈：记录被采纳的建议，持续改进
4. 集成CI/CD：自动触发审查
```

---

## 代码实现题

### 题目1：实现一个简单的RAG系统

**要求：**
```
用LangChain实现一个基本的RAG问答系统，包括：
1. 加载文本文档
2. 分割和向量化
3. 检索和生成答案
时间：20-30分钟
```

**参考答案：**

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class SimpleRAG:
    def __init__(self, document_path: str):
        # 1. 加载文档
        loader = TextLoader(document_path)
        documents = loader.load()
        
        # 2. 分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # 3. 创建向量存储
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        
        # 4. 创建检索器
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # 5. 创建生成chain
        template = """根据以下上下文回答问题。如果上下文中没有相关信息，
请说"我不知道"。

上下文：
{context}

问题：{question}

答案："""
        
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
        )
    
    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query(self, question: str) -> str:
        return self.chain.invoke(question)

# 使用
if __name__ == "__main__":
    # 假设有一个文档文件
    rag = SimpleRAG("company_docs.txt")
    
    # 查询
    answer = rag.query("公司的假期政策是什么？")
    print(answer)
```

**评分要点：**
- ✅ 正确使用Document Loader
- ✅ 合理的chunk参数
- ✅ 正确创建向量存储
- ✅ 使用LCEL组合components
- ✅ Prompt设计合理
- 🌟 加分：添加错误处理、流式输出、缓存

### 题目2：实现一个自定义Tool

**要求：**
```
实现一个天气查询Tool，要求：
1. 继承BaseTool
2. 实现同步和异步方法
3. 添加错误处理
4. 定义清晰的描述
```

**参考答案：**

```python
from langchain.tools import BaseTool
from pydantic import Field
from typing import Optional, Type
from pydantic import BaseModel
import aiohttp
import requests

# 定义输入schema
class WeatherInput(BaseModel):
    city: str = Field(description="城市名称，例如：北京、上海")
    
class WeatherTool(BaseTool):
    name = "weather_query"
    description = """
    查询指定城市的当前天气信息。
    输入应该是城市名称（中文或英文都可以）。
    返回温度、天气状况、风力等信息。
    例如输入"北京"会返回北京当前的天气情况。
    """
    args_schema: Type[BaseModel] = WeatherInput
    
    # 可选：工具特定配置
    api_key: str = Field(default="", description="API密钥")
    api_url: str = "https://api.weatherapi.com/v1/current.json"
    
    def _run(self, city: str) -> str:
        """同步执行"""
        try:
            # 调用天气API（示例）
            params = {
                "key": self.api_key,
                "q": city,
                "lang": "zh"
            }
            
            response = requests.get(
                self.api_url,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._format_weather_data(data)
            else:
                return f"查询失败：{response.status_code}"
                
        except requests.Timeout:
            return "查询超时，请稍后重试"
        except requests.RequestException as e:
            return f"网络错误：{str(e)}"
        except Exception as e:
            return f"未知错误：{str(e)}"
    
    async def _arun(self, city: str) -> str:
        """异步执行"""
        try:
            params = {
                "key": self.api_key,
                "q": city,
                "lang": "zh"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_weather_data(data)
                    else:
                        return f"查询失败：{response.status}"
                        
        except asyncio.TimeoutError:
            return "查询超时，请稍后重试"
        except aiohttp.ClientError as e:
            return f"网络错误：{str(e)}"
        except Exception as e:
            return f"未知错误：{str(e)}"
    
    def _format_weather_data(self, data: dict) -> str:
        """格式化天气数据"""
        try:
            location = data["location"]["name"]
            temp_c = data["current"]["temp_c"]
            condition = data["current"]["condition"]["text"]
            humidity = data["current"]["humidity"]
            wind_kph = data["current"]["wind_kph"]
            
            return f"""
{location}的天气情况：
- 温度：{temp_c}°C
- 天气：{condition}
- 湿度：{humidity}%
- 风速：{wind_kph} km/h
            """.strip()
        except KeyError:
            return "天气数据格式错误"

# 使用
if __name__ == "__main__":
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    # 创建tool
    weather_tool = WeatherTool(api_key="your-api-key")
    tools = [weather_tool]
    
    # 创建agent
    llm = ChatOpenAI(model="gpt-4")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    # 测试
    result = agent_executor.invoke({"input": "北京的天气怎么样？"})
    print(result["output"])
```

**评分要点：**
- ✅ 正确继承BaseTool
- ✅ 实现_run和_arun方法
- ✅ 定义args_schema
- ✅ 错误处理完善
- ✅ 清晰的description
- 🌟 加分：超时处理、重试机制、结果验证

---

## 行为面试准备

### 1. 为什么对LLM和LangChain感兴趣？

**好的回答：**
```
我对LLM和LangChain感兴趣主要有三个原因：

第一，技术革命性。大语言模型代表了AI发展的重要方向，它们展现出惊人的
理解和生成能力。我相信LLM会深刻改变软件开发和人机交互的方式。

第二，实用价值高。LangChain提供了将LLM能力转化为实际应用的完整框架。
通过学习LangChain，我能够快速构建有价值的AI应用，解决真实的业务问题。

第三，持续学习机会。这个领域发展非常快，新技术、新方法不断涌现，
这激励我保持学习热情。同时LangChain的开源社区活跃，我也希望能够参与贡献。

具体到LangChain，我特别欣赏它的设计理念：
- Runnable抽象优雅，让组件组合变得简单
- LCEL表达式让代码更简洁易读
- 模块化设计使得系统易于扩展和维护

我期待能将这些技术应用到实际项目中，创造价值。
```

### 2. 学习LangChain遇到的最大挑战是什么？如何克服？

**STAR方法回答：**

```
【Situation - 情境】
在构建RAG系统时，我发现检索的准确率只有60%左右，
远低于预期的80%，很多相关文档没有被检索到。

【Task - 任务】
我需要找出问题原因并提升检索准确率。

【Action - 行动】
我采取了系统性的排查和优化：

1. 分析数据：
   - 收集了100个查询的检索结果
   - 发现问题主要是文本分割不合理，
     很多相关内容被分到了不同chunk

2. 学习研究：
   - 阅读LangChain文档中关于Text Splitter的部分
   - 研究了不同分割策略的优缺点
   - 查看了社区中的最佳实践

3. 实验优化：
   - 调整chunk_size从500到1000
   - 增加overlap从50到200
   - 改用RecursiveCharacterTextSplitter，
     优先按段落分割
   - 尝试了混合检索（BM25 + 向量）

4. 评估迭代：
   - 建立了评估数据集
   - 对比不同方案的效果
   - 最终选择表现最好的组合

【Result - 结果】
通过这些优化，检索准确率从60%提升到了82%，
达到了预期目标。

更重要的是，这个过程让我深入理解了RAG系统的
关键技术点，也学会了系统性解决问题的方法。
```

### 3. 你认为LangChain的优缺点是什么？

**平衡的回答：**

```
【优点】

1. 优雅的抽象设计
   - Runnable接口统一了所有组件
   - LCEL让代码简洁且可读
   - 组合模式提供了极大的灵活性

2. 丰富的生态系统
   - 支持主流的LLM提供商
   - 集成了各种工具和数据源
   - 活跃的社区和大量示例

3. 生产就绪
   - 内置回调和追踪系统
   - 完善的错误处理
   - 与LangSmith集成提供可观测性

4. 快速开发
   - 预构建的chains和agents
   - 减少样板代码
   - 快速原型验证

【缺点/限制】

1. 学习曲线
   - 概念抽象层次高，初学者需要时间理解
   - 文档虽然丰富但有时不够详细
   - 版本更新快，API可能变化

2. 性能开销
   - 抽象层带来一定的性能损失
   - 对于简单任务可能过度设计

3. 调试复杂性
   - 链式调用深时，调试困难
   - 错误信息有时不够明确

4. 控制粒度
   - 高层抽象有时限制了灵活性
   - 复杂场景可能需要LangGraph

【改进建议】
对于复杂的生产应用，我会：
- 使用LangGraph获得更精确的控制
- 添加详细的日志和监控
- 进行充分的性能测试和优化
- 保持对新版本特性的关注

总体来说，LangChain的优点远大于缺点，
是目前最成熟的LLM应用开发框架之一。
```

---

## 最后的建议

### 面试前准备

1. **复习核心概念**
   - 重新阅读学习指南
   - 回顾代码示例
   - 准备1-2个深入的项目故事

2. **准备演示**
   - 准备一个可以展示的项目
   - 能够现场解释代码
   - 准备讨论技术选择和权衡

3. **了解公司**
   - 研究公司的AI产品
   - 了解技术栈
   - 准备相关问题

4. **模拟面试**
   - 找朋友练习
   - 录制自己的回答
   - 优化表达方式

### 面试中注意

1. **展现热情**
   - 表达对AI技术的兴趣
   - 分享学习心得
   - 展示成长意愿

2. **结构化回答**
   - 使用STAR方法
   - 先总后分
   - 重点突出

3. **诚实谦虚**
   - 不懂的承认不懂
   - 展示学习能力
   - 愿意接受指导

4. **主动沟通**
   - 理解问题后再回答
   - 边写边解释
   - 寻求反馈

### 面试后跟进

1. 发送感谢邮件
2. 反思面试表现
3. 补充未掌握的知识
4. 保持积极心态

---

## 总结

通过充分准备，你应该能够：

✅ 自信地介绍LangChain项目经验
✅ 深入回答技术问题
✅ 设计合理的系统方案
✅ 快速实现代码要求
✅ 展现学习能力和潜力

**记住：**
- 技术能力是基础
- 沟通能力同样重要
- 学习态度是关键
- 项目经验是加分项

祝你面试成功，拿到心仪的Offer！🎉

---

*这份指南持续更新中，欢迎反馈和建议。*

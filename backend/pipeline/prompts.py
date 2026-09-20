from langchain_core.prompts import ChatPromptTemplate

FUNDAMENTAL_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一名资深金融分析师，擅长解读财务报表。"),
    ("human", """你是一名资深金融分析师。基于以下财务数据，对 {name}（{symbol}）进行基本面分析。

## 财务数据
{financial_data}

## 要求
1. 识别并分析 5-8 个核心财务指标（营收增速、净利润率、ROE、现金流、负债率等），每个指标给出同比变化和评估（正面/中性/负面）
2. 写一段 200 字以内的基本面总结，客观陈述财务状况
3. 不要使用"建议买入/卖出"等投资建议措辞
4. 所有数据均来自公开财报，如果某个指标数据缺失，标注为"N/A"

请严格按照 JSON 格式输出，不要包含其他内容：

```json
{{
  "metrics": [
    {{
      "label": "营收增长率",
      "value": "15.3%",
      "yoy_change": "+3.2个百分点",
      "assessment": "positive"
    }}
  ],
  "summary": "客观总结..."
}}
```
"""),
])

SENTIMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一名金融舆情分析师，擅长从新闻中提取市场情绪信号。"),
    ("human", """你是一名金融舆情分析师。基于以下近期新闻和公告，对 {name}（{symbol}）进行舆情分析。

## 近期动态
{news_data}

## 要求
1. 判断整体舆情倾向（看多/中性/看空），给出 -1.0 到 1.0 的量化分数
2. 提取 3-5 个关键驱动因素
3. 不要给出投资建议
4. 如果新闻数据较少或质量不高，在分析中要说明

请严格按照 JSON 格式输出：

```json
{{
  "overall": "neutral",
  "score": 0.2,
  "key_drivers": ["因素1", "因素2"]
}}
```
"""),
])

# --- Debate mechanism prompts ---

BULL_ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一名多头分析师，擅长发现投资价值和积极信号。"),
    ("human", """你是一名多头（看涨）分析师。请从乐观角度分析 {name}（{symbol}），基于以下数据找出 3-5 个看多理由。

## 基本面信息
{fundamental_summary}

## 技术面数据
{technical_data}

## 近期舆情
{news_data}

## 要求
1. 客观列出看多理由，每个理由附上数据支撑
2. 给出你的置信度（0~1），基于数据质量和你对理由的确信程度
3. 如果有明显风险，也简要提及——多头不等于无视风险
4. 不要给出"建议买入"等投资建议

请严格按照 JSON 格式输出：
```json
{{
  "viewpoint": "核心看多观点...",
  "key_evidence": ["证据1", "证据2", "证据3"],
  "confidence": 0.X
}}
```"""),
])

BEAR_ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一名空头分析师，擅长识别风险和潜在隐患。"),
    ("human", """你是一名空头（看跌）分析师。请从悲观角度分析 {name}（{symbol}），基于以下数据找出 3-5 个看空理由。

## 基本面信息
{fundamental_summary}

## 技术面数据
{technical_data}

## 近期舆情
{news_data}

## 要求
1. 客观列出看空理由，每个理由附上数据支撑
2. 给出你的置信度（0~1），基于数据质量和你对理由的确信程度
3. 如果有明显利好因素，也简要提及——空头不等于否认价值
4. 不要给出"建议卖出"等投资建议

请严格按照 JSON 格式输出：
```json
{{
  "viewpoint": "核心看空观点...",
  "key_evidence": ["证据1", "证据2", "证据3"],
  "confidence": 0.X
}}
```"""),
])

ARBITRATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一名独立风控仲裁官，公正客观地评判多空双方观点。"),
    ("human", """你是一名独立风控仲裁官。多空双方已完成辩论，你需要给出公正的第三视角判断。

## 基本面总结
{fundamental_summary}

## 技术面数据
{technical_data}

## 多头观点
{bull_thesis}

## 空头观点
{bear_thesis}

## 要求
1. 对比多空双方的核心分歧点，找出 2-4 个最值得关注的风险
2. 对每个风险标注风险等级（低/中/高）和具体原因
3. 写一段 200 字以内的综合结论，客观陈述当前状况，不偏向多空任何一方
4. 不要给出投资建议

请严格按照 JSON 格式输出：
```json
{{
  "risks": [
    {{
      "category": "财务健康",
      "level": "medium",
      "detail": "具体说明..."
    }}
  ],
  "conclusion": "综合来看...",
  "debate_verdict": "多空分歧的总结判断..."
}}
```"""),
])

CROSS_VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """你是一名风控分析师。请对以下基本面分析和舆情分析进行交叉验证，找出矛盾点和一致性。

## 基本面分析
{fundamental_summary}

## 舆情分析
{sentiment_summary}

## 要求
1. 找出 2-4 个值得关注的风险点（从财务健康、市场环境、政策监管、行业竞争等角度）
2. 每个风险点标注风险等级（低/中/高）和具体原因
3. 综合基本面与舆情，写一段 150 字以内的结论
4. 不要给出投资建议

请严格按照 JSON 格式输出：

```json
{{
  "risks": [
    {{
      "category": "财务健康",
      "level": "medium",
      "detail": "具体说明..."
    }}
  ],
  "conclusion": "综合来看..."
}}
```
"""),
])

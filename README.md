# Finance AI Analyst

AI 驱动的上市公司财报 + 舆情综合分析平台。

输入 A 股或美股代码，自动拉取最新财报与新闻数据，通过四阶段 LLM 分析管线生成结构化报告。

## 功能

- **基本面分析** — 提取核心财务指标（营收增速、ROE、利润率、负债率等），逐项评估
- **技术面指标** — MA5/20/60、MACD、RSI(14)、KDJ 由本地 pandas 计算，不经过 LLM
- **舆情分析** — 汇总近期新闻，量化市场情绪（-1.0 ~ 1.0），提取关键驱动因素
- **多空辩论** — 多头 / 空头分析师并行论证，各自给出置信度
- **风控仲裁** — 独立第三方视角对比多空分歧，标注风险等级并给出综合结论
- **历史存储** — 分析报告持久化到 JSON，支持按股票代码精确检索历史
- **语义检索** — 报告向量化存入 ChromaDB，支持用自然语言跨股票检索历史报告

## 架构

```
用户 → Streamlit 前端 → FastAPI 后端 → LangChain 分析管线
                              ↓
                    akshare / yfinance (数据)
                              ↓
       JSON 文件存储 (历史) + ChromaDB 向量索引 (语义检索)
```

四阶段分析管线：

```
Step 1: 基本面 + 技术面  →  基本面指标由 LLM 提取；MA/MACD/RSI/KDJ 由 pandas 本地计算
Step 2: 舆情分析        →  LLM 量化市场情绪（可选，无新闻时跳过）
Step 3: 多空辩论        →  多头 / 空头分析师并行输出论点与置信度
Step 4: 风控仲裁        →  LLM 对比多空分歧，识别 2-4 个风险并给出结论
```

每个 LLM 阶段都是一条 LCEL 链（`ChatPromptTemplate | ChatOpenAI | JsonOutputParser`），
Step 3 的多空两路用 `RunnableParallel` 并行执行；网络错误与 JSON 解析失败共用一套
指数退避重试（最多 3 次尝试）。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key（支持 OpenAI / DeepSeek / 阿里云百炼 等兼容接口）

# 3. 启动后端 (终端 1)
python -m backend.main

# 4. 启动前端 (终端 2)
python -m streamlit run frontend/app.py --server.port 8501
```

浏览器打开 `http://localhost:8501`，输入股票代码即可开始分析。

## 支持的 LLM

任何兼容 OpenAI 接口的 API 均可使用，在 `.env` 中配置：

```env
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # 阿里云百炼
LLM_API_KEY=sk-your-key
LLM_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v3   # 语义检索用的向量化模型，需与上者同一家服务
```

## 语义检索

报告会按 `fundamental_summary + conclusion + 风险点` 等关键文本向量化，存入 ChromaDB
的 `analysis_reports` collection（余弦距离）。可以跨股票用自然语言检索历史报告：

```bash
curl "http://localhost:8000/api/v1/search?q=高风险的消费类股票分析&limit=5"
curl "http://localhost:8000/api/v1/search?q=现金流质量&symbol=600519"
```

返回每条的 `symbol` / `name` / `timestamp` / `score`（余弦相似度，越大越相似）/ `text`。

向量索引由 JSON 文件派生：写入报告时双写（JSON 为准，索引失败只告警不影响落盘），
启动时比对 id 差集自动补齐缺失的向量，embedding 模型变更则清空重建。

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | Streamlit |
| 后端 | FastAPI + Uvicorn |
| AI 框架 | LangChain (LCEL) + langchain-openai / OpenAI SDK |
| 数据 | akshare (A股) / yfinance (美股) |
| 存储 | JSON 文件 + ChromaDB 向量存储与相似度检索 |

## 示例

- A 股：`600519`（贵州茅台）、`000001`（平安银行）
- 美股：`AAPL`（苹果）、`TSLA`（特斯拉）

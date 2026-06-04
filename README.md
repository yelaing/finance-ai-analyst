# Finance AI Analyst

AI 驱动的上市公司财报 + 舆情综合分析平台。

输入 A 股或美股代码，自动拉取最新财报与新闻数据，通过三步 LLM 分析管线生成结构化报告。

## 功能

- **基本面分析** — 提取核心财务指标（营收增速、ROE、利润率、负债率等），逐项评估
- **舆情分析** — 汇总近期新闻，量化市场情绪（-1.0 ~ 1.0），提取关键驱动因素
- **风险交叉验证** — 综合基本面与舆情，识别矛盾点与潜在风险
- **历史存储** — 分析报告持久化，支持按股票代码检索历史

## 架构

```
用户 → Streamlit 前端 → FastAPI 后端 → LangChain 分析管线
                              ↓
                    akshare / yfinance (数据)
                              ↓
                      JSON 文件存储 (历史)
```

三步分析管线：

```
Step 1: 基本面分析  →  LLM 提取 5-8 个核心指标
Step 2: 舆情分析    →  LLM 量化市场情绪
Step 3: 交叉验证    →  LLM 识别风险 + 生成结论
```

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
```

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | Streamlit |
| 后端 | FastAPI + Uvicorn |
| AI 框架 | LangChain + OpenAI SDK |
| 数据 | akshare (A股) / yfinance (美股) |
| 存储 | JSON 文件持久化 |

## 示例

- A 股：`600519`（贵州茅台）、`000001`（平安银行）
- 美股：`AAPL`（苹果）、`TSLA`（特斯拉）

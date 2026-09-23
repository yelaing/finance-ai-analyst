"""公共测试夹具。

原则：只替换**外部边界**（LLM 链、embed_texts、akshare / yfinance），
routes、中间件、异常处理器、四阶段编排、双写与差集自愈逻辑一律真实执行。

另外：凡是构造 Settings 的测试都传 `_env_file=None`，否则会读到开发者本机的
.env，断言随环境漂移。
"""

import math
import re
import zlib
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.schemas.models import (
    AnalysisReport,
    DebateThesis,
    MetricItem,
    RiskItem,
    SentimentSummary,
)

_EMB_DIM = 48


def _tokens(text: str) -> list[str]:
    # 中文按单字、英文数字按词切分：字面重叠多的文本会得到更接近的向量
    return re.findall(r"[a-z0-9]+|[一-鿿]", text.lower())


def fake_embed(texts: list[str]) -> list[list[float]]:
    """确定性的假向量，不发网络请求。

    仅用于验证检索**链路**（排序、过滤、维度、相似度换算），不代表真实语义质量。
    """
    vectors = []
    for text in texts:
        vec = [0.0] * _EMB_DIM
        for tok in _tokens(text):
            vec[zlib.crc32(tok.encode()) % _EMB_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


@pytest.fixture
def stub_embed(monkeypatch):
    """替换向量化调用，返回假向量。

    注意 patch 目标必须是**使用方模块**：chroma_store 用
    `from backend.store.embeddings import embed_texts` 导入，函数对象在导入时
    就绑定进了它自己的命名空间，改 `backend.store.embeddings.embed_texts`
    对它无效。
    """
    import backend.store.chroma_store as chroma_store

    monkeypatch.setattr(chroma_store, "embed_texts", fake_embed)
    return fake_embed


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """隔离的数据目录：JSON 报告文件与 ChromaDB 都在临时目录里，绝不碰真实 data/。"""
    data_dir = tmp_path / "data"
    (data_dir / "chroma").mkdir(parents=True)
    return data_dir


@pytest.fixture
def store(tmp_data_dir: Path, stub_embed):
    from backend.store.chroma_store import ReportStore

    return ReportStore(
        data_file=str(tmp_data_dir / "reports.json"),
        chroma_dir=str(tmp_data_dir / "chroma"),
    )


def build_report(
    symbol: str = "600519",
    name: str = "贵州茅台",
    *,
    timestamp: datetime | None = None,
    with_sentiment: bool = True,
    with_debate: bool = True,
    summary: str = "营收稳健，现金流充沛。",
    conclusion: str = "综合来看基本面稳健，增速放缓。",
) -> AnalysisReport:
    """构造一份结构完整的报告，字段都可调以便覆盖各分支。"""
    return AnalysisReport(
        symbol=symbol,
        name=name,
        timestamp=timestamp or datetime(2026, 9, 24, 10, 0, 0),
        fundamental_metrics=[
            MetricItem(label="营收增速", value="6.3%", yoy_change="-3.2pct", assessment="neutral")
        ],
        fundamental_summary=summary,
        sentiment=(
            SentimentSummary(overall="neutral", score=0.1, key_drivers=["渠道压力"])
            if with_sentiment
            else None
        ),
        risks=[RiskItem(category="增长可持续性", level="high", detail="增速回落")],
        conclusion=conclusion,
        sources=["同花顺财务摘要"],
        bull_thesis=(
            DebateThesis(
                analyst="bull",
                viewpoint="盈利质量优异",
                key_evidence=["毛利 89%"],
                confidence=0.7,
            )
            if with_debate
            else None
        ),
        bear_thesis=(
            DebateThesis(
                analyst="bear",
                viewpoint="增速换挡",
                key_evidence=["净利增速 1.5%"],
                confidence=0.6,
            )
            if with_debate
            else None
        ),
        debate_verdict="分歧在于增速是否结构性下移。" if with_debate else None,
    )


@pytest.fixture
def report_factory():
    return build_report


@pytest.fixture
def client(store, monkeypatch):
    """绑定隔离 store 的 TestClient。

    routes 直接调用 get_store()（不是 Depends），所以替换模块里的这一层。
    这是替换协作对象，不是伪造内部行为。
    """
    import backend.api.routes as routes
    from backend.main import app

    monkeypatch.setattr(routes, "get_store", lambda: store)
    return TestClient(app)

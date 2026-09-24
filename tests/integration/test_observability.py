"""可观测性端点：/metrics、/health、/health/ready，以及指标真的被填上。

走真实的中间件与路由 —— 指标埋点在中间件里，用假 app 测不出接线问题。
"""

import re
from datetime import datetime

import pytest

import backend.main as main_module
from backend.api.health import DependencyStatus
from tests.conftest import build_report

pytestmark = pytest.mark.integration


@pytest.fixture
def deps(monkeypatch):
    """替换依赖探测，避免测试打真实网络 / 真实数据库。"""

    def install(statuses):
        monkeypatch.setattr(main_module, "check_all", lambda: statuses)

    return install


def metrics_text(client) -> str:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    return resp.text


def sample(text: str, name: str, **labels) -> float | None:
    """从 Prometheus 文本里取一条样本。只按名字+标签精确匹配。"""
    pattern = re.compile(rf"^{re.escape(name)}\{{(.*?)\}} (\S+)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        body = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
        if all(body.get(key) == value for key, value in labels.items()):
            return float(match.group(2))
    return None


# ---------- /metrics ----------


def test_metrics_endpoint_returns_prometheus_text(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "# HELP" in resp.text and "# TYPE" in resp.text


def test_metrics_is_not_part_of_the_business_api(client):
    """指标端点不该出现在 OpenAPI 里 —— 它不是业务接口。"""
    spec = client.get("/openapi.json").json()
    assert "/metrics" not in spec["paths"]


def test_requests_are_counted_with_route_template(client):
    client.get("/health")

    text = metrics_text(client)
    assert sample(text, "http_requests_total", method="GET", path="/health", status="200") == 1
    assert sample(text, "http_request_duration_seconds_count", method="GET", path="/health") == 1


def test_path_label_uses_route_template_not_raw_url(client, store):
    """**基数守卫**：/export/{symbol} 必须按模板聚合成一条序列。

    用原始 URL 当标签的话，每分析一个新股票代码就会长出一条独立时间序列 ——
    几百个股票就能把 Prometheus 拖垮。这条用例就是钉住这一点。
    """
    store.save(build_report(symbol="600519", timestamp=datetime(2026, 9, 24, 9, 0, 0)))
    store.save(build_report(symbol="AAPL", name="Apple", timestamp=datetime(2026, 9, 24, 10, 0, 0)))
    client.get("/api/v1/export/600519")
    client.get("/api/v1/export/AAPL")  # 两个不同 symbol，仍必须是同一条序列

    text = metrics_text(client)
    templated = sample(
        text, "http_requests_total", method="GET", path="/api/v1/export/{symbol}", status="200"
    )
    assert templated == 2  # 两个不同 symbol 聚合成一条序列
    # 原始 URL 形态绝不能出现
    assert 'path="/api/v1/export/600519"' not in text
    assert 'path="/api/v1/export/AAPL"' not in text


def test_unmatched_requests_are_bucketed_not_labelled_by_url(client):
    """随便打一个不存在的路径，标签也必须是固定值，不能让攻击者用随机 URL 撑爆基数。"""
    client.get("/api/v1/nonexistent")
    client.get("/api/v1/another-made-up-path")

    text = metrics_text(client)
    assert sample(text, "http_requests_total", method="GET", path="unmatched", status="404") == 2


def test_metrics_include_llm_and_cache_families(client):
    text = metrics_text(client)
    for family in (
        "llm_calls_total",
        "llm_tokens_total",
        "cache_operations_total",
        "data_fetch_seconds",
        "analysis_runs_total",
        "analysis_degraded_stages_total",
    ):
        assert f"# TYPE {family}" in text, f"缺少指标族 {family}"


# 缓存的 hit/miss 计数由 test_cache / test_pipeline 覆盖：
# 这里走不通 —— store 的测试替身把 `embed_texts` 整个换掉了，而缓存逻辑就在真实的
# embed_texts 内部，于是那段代码根本没被执行（替身把被测代码也一起替掉了）。


# ---------- /health 与 /health/ready ----------


def test_health_reports_dependencies_and_stays_200(client, deps):
    deps(
        [DependencyStatus("chromadb", True, "12 条向量"), DependencyStatus("llm", True, "5 个模型")]
    )

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert [d["name"] for d in body["dependencies"]] == ["chromadb", "llm"]
    assert all(d["ok"] for d in body["dependencies"])


def test_health_stays_200_even_when_dependency_is_down(client, deps):
    """存活探针不能因为上游不可用就失败 —— 那会让编排重启一个健康的容器。"""
    deps([DependencyStatus("llm", False, "APITimeoutError: boom")])

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_ready_returns_200_when_all_dependencies_ok(client, deps):
    deps([DependencyStatus("chromadb", True, "12 条向量"), DependencyStatus("llm", True, "ok")])

    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_ready_returns_503_when_a_dependency_is_down(client, deps):
    deps([DependencyStatus("chromadb", True, "12 条向量"), DependencyStatus("llm", False, "炸了")])

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert [d["ok"] for d in body["dependencies"]] == [True, False]


def test_health_endpoints_are_counted_separately(client, deps):
    deps([DependencyStatus("chromadb", True, "ok")])
    client.get("/health")
    client.get("/health/ready")

    text = metrics_text(client)
    assert sample(text, "http_requests_total", path="/health", status="200") is not None
    assert sample(text, "http_requests_total", path="/health/ready", status="200") is not None

"""API 集成测试：走真实的路由 / 中间件 / 异常处理器，只在 store 与 LLM 边界替换。"""

import logging
from datetime import datetime

import pytest

import backend.api.routes as routes
from backend.core.errors import InvalidSymbolError, UnsupportedMarketError
from tests.conftest import build_report

pytestmark = pytest.mark.integration

ERROR_PATHS = [
    ("POST", "/api/v1/analyze", {"json": {"symbol": "!!!bad", "market": "auto"}}, "INVALID_SYMBOL"),
    (
        "POST",
        "/api/v1/analyze",
        {"json": {"symbol": "600519", "market": "mars"}},
        "UNSUPPORTED_MARKET",
    ),
    ("GET", "/api/v1/export/NOPE", {}, "REPORT_NOT_FOUND"),
    ("GET", "/api/v1/nonexistent", {}, "NOT_FOUND"),
    ("GET", "/api/v1/search", {}, "VALIDATION_ERROR"),
    ("GET", "/api/v1/search?q=x&limit=50", {}, "VALIDATION_ERROR"),
    ("GET", "/api/v1/history", {}, "VALIDATION_ERROR"),
    ("POST", "/api/v1/analyze", {"json": {"market": "auto"}}, "VALIDATION_ERROR"),
]


# ---------- 基础探针 ----------


def test_lifespan_applies_network_env(monkeypatch):
    """网络环境变量在**服务启动时**应用，而不是 import 时。

    放 import 时会给任何导入 backend.main 的一方留下全局副作用 ——
    实测那会污染测试里「默认值为 None」的断言（收集阶段就改了进程环境）。
    """
    import os

    from fastapi.testclient import TestClient

    import backend.main as main_module
    from backend.config import Settings

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        main_module,
        "settings",
        Settings(_env_file=None, http_proxy="http://proxy.example:8080", no_proxy="example.com"),
    )

    with TestClient(main_module.app):
        assert os.environ["HTTP_PROXY"] == "http://proxy.example:8080"
        assert os.environ["NO_PROXY"] == "example.com"


def test_network_env_is_not_applied_at_import_time():
    """结构断言：apply_network_env 不得出现在模块顶层。

    放顶层就是 import 副作用 —— 测试收集阶段就会执行，污染后续断言；也让任何
    导入 backend.main 的一方被动改掉自己的进程环境。必须留在 lifespan 里。
    用 AST 而不是行号，避免断言随排版失效。
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "backend" / "main.py").read_text(encoding="utf-8"))

    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", "") == "apply_network_env"
    ]
    assert top_level_calls == [], "apply_network_env 被放在模块顶层了，应移入 lifespan"


def test_health_is_liveness_and_reports_dependencies(client, monkeypatch):
    """存活探针**始终 200**（依赖不可用也不改状态码 —— 否则编排会去重启一个
    本来健康的容器），但会把各依赖状态报出来供人看。"""
    import backend.main as main_module
    from backend.api.health import DependencyStatus

    monkeypatch.setattr(
        main_module,
        "check_all",
        lambda: [
            DependencyStatus("chromadb", True, "12 条向量"),
            DependencyStatus("llm", False, "APITimeoutError: boom"),
        ],
    )

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert [d["name"] for d in body["dependencies"]] == ["chromadb", "llm"]


def test_every_response_carries_request_id(client):
    request_id = client.get("/health").headers["X-Request-ID"]
    assert len(request_id) == 8


def test_caller_supplied_request_id_is_propagated(client):
    resp = client.get("/health", headers={"X-Request-ID": "feedface"})
    assert resp.headers["X-Request-ID"] == "feedface"


def test_request_ids_differ_between_requests(client):
    assert (
        client.get("/health").headers["X-Request-ID"]
        != client.get("/health").headers["X-Request-ID"]
    )


# ---------- 统一错误响应 ----------


@pytest.mark.parametrize(("method", "path", "kwargs", "code"), ERROR_PATHS)
def test_error_responses_share_one_shape(client, method, path, kwargs, code):
    body = client.request(method, path, **kwargs).json()
    assert set(body) == {"detail", "code", "request_id"}
    assert body["code"] == code
    assert isinstance(body["detail"], str) and body["detail"]
    assert body["request_id"]


def test_invalid_symbol_message_is_user_actionable(client):
    resp = client.post("/api/v1/analyze", json={"symbol": "!!!bad", "market": "auto"})
    assert resp.status_code == 400
    assert "无法自动识别" in resp.json()["detail"]


def test_validation_error_flattens_field_path(client):
    body = client.get("/api/v1/search").json()
    assert body["code"] == "VALIDATION_ERROR"
    assert "q" in body["detail"]


def test_unhandled_exception_becomes_generic_500(client):
    def boom(symbol, market="auto", include_sentiment=True):
        raise RuntimeError("内部口令是 hunter2")

    client.app.dependency_overrides = {}  # 明确不用依赖注入
    original = routes.run_analysis
    routes.run_analysis = boom
    try:
        resp = client.post("/api/v1/analyze", json={"symbol": "999999", "market": "a_share"})
    finally:
        routes.run_analysis = original

    body = resp.json()
    assert resp.status_code == 500
    assert body["code"] == "INTERNAL_ERROR"
    assert body["detail"] == "分析失败，请稍后重试"
    assert "hunter2" not in resp.text
    assert "Traceback" not in resp.text


def test_unhandled_exception_is_logged_with_traceback(client, caplog):
    def boom(symbol, market="auto", include_sentiment=True):
        raise RuntimeError("内部口令是 hunter2")

    original = routes.run_analysis
    routes.run_analysis = boom
    try:
        with caplog.at_level(logging.ERROR):
            client.post("/api/v1/analyze", json={"symbol": "999999", "market": "a_share"})
    finally:
        routes.run_analysis = original

    assert "hunter2" in caplog.text
    assert "Traceback" in caplog.text


# ---------- /api/v1/history ----------


def test_middleware_backstops_exceptions_that_bypass_handlers(client, store, monkeypatch, caplog):
    """路由里没有 try/except 的异常绕过统一处理器，由中间件兜底成 500。

    这条路径是 request_id 设计的关键：中间件的 except 分支必须仍能拿到它。
    """

    def boom(*args, **kwargs):
        raise RuntimeError("store 内部炸了")

    monkeypatch.setattr(store, "get_history", boom)
    with caplog.at_level(logging.ERROR):
        resp = client.get("/api/v1/history", params={"symbol": "600519"})

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "INTERNAL_ERROR"
    assert body["detail"] == "服务器内部错误"
    assert body["request_id"]
    assert resp.headers["X-Request-ID"] == body["request_id"]
    assert "store 内部炸了" in caplog.text


def test_history_lists_saved_reports(client, store):
    store.save(build_report(timestamp=datetime(2026, 9, 24, 9, 0, 0)))
    store.save(build_report(symbol="AAPL", name="Apple", timestamp=datetime(2026, 9, 24, 10, 0, 0)))

    body = client.get("/api/v1/history", params={"symbol": "600519"}).json()
    assert body["total"] == 1
    item = body["items"][0]
    assert set(item) == {"id", "symbol", "name", "timestamp", "summary"}
    assert item["symbol"] == "600519"
    assert item["summary"] == "同花顺财务摘要"


def test_history_respects_limit(client, store):
    for minute in range(4):
        store.save(build_report(timestamp=datetime(2026, 9, 24, 9, minute, 0)))
    body = client.get("/api/v1/history", params={"symbol": "600519", "limit": 2}).json()
    assert body["total"] == 2


def test_history_for_unknown_symbol_is_empty_not_404(client):
    body = client.get("/api/v1/history", params={"symbol": "NOPE"}).json()
    assert body == {"items": [], "total": 0}


# ---------- /api/v1/search ----------


def test_search_returns_scored_hits(client, store):
    store.save(
        build_report(
            symbol="002594",
            name="比亚迪",
            summary="新能源汽车与动力电池",
            timestamp=datetime(2026, 9, 24, 9, 0, 0),
        )
    )
    body = client.get("/api/v1/search", params={"q": "新能源汽车与动力电池", "limit": 3}).json()

    assert body["total"] == 1
    hit = body["items"][0]
    assert set(hit) == {"id", "symbol", "name", "timestamp", "score", "text"}
    assert hit["symbol"] == "002594"
    assert 0 < hit["score"] <= 1


def test_search_on_empty_index_is_empty(client):
    assert client.get("/api/v1/search", params={"q": "随便"}).json() == {"items": [], "total": 0}


def test_search_symbol_filter_yields_nothing_for_unknown(client, store):
    store.save(build_report())
    body = client.get("/api/v1/search", params={"q": "基本面", "symbol": "NOPE"}).json()
    assert body["total"] == 0


# ---------- /api/v1/export ----------


def test_export_returns_markdown_attachment(client, store):
    store.save(build_report())
    resp = client.get("/api/v1/export/600519")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert "600519_report.md" in resp.headers["content-disposition"]
    assert "# 贵州茅台（600519）分析报告" in resp.text


def test_export_missing_report_returns_404(client):
    resp = client.get("/api/v1/export/NOPE")
    assert resp.status_code == 404
    assert resp.json()["code"] == "REPORT_NOT_FOUND"
    assert resp.json()["detail"] == "未找到该股票的分析报告"


# ---------- /api/v1/analyze ----------


def test_analyze_runs_pipeline_and_persists_it(client, store, monkeypatch):
    calls = []

    def fake_run(symbol, market="auto", include_sentiment=True):
        calls.append((symbol, market, include_sentiment))
        return build_report(timestamp=datetime(2026, 9, 24, 10, 0, 0))

    monkeypatch.setattr(routes, "run_analysis", fake_run)
    resp = client.post("/api/v1/analyze", json={"symbol": "600519", "market": "a_share"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["from_cache"] is False
    assert body["report"]["symbol"] == "600519"
    assert calls == [("600519", "a_share", True)]

    # 落盘了才算真跑通，用另一个端点验证
    assert client.get("/api/v1/history", params={"symbol": "600519"}).json()["total"] == 1


def test_analyze_returns_cache_hit_for_todays_report(client, store, monkeypatch):
    store.save(build_report(timestamp=datetime.now()))

    def must_not_run(*args, **kwargs):
        raise AssertionError("命中缓存时不该重新分析")

    monkeypatch.setattr(routes, "run_analysis", must_not_run)
    body = client.post("/api/v1/analyze", json={"symbol": "600519"}).json()

    assert body["from_cache"] is True
    assert body["cached_at"] is not None


def test_analyze_ignores_stale_cache_and_reanalyzes(client, store, monkeypatch):
    store.save(build_report(timestamp=datetime(2026, 1, 1, 9, 0, 0)))
    monkeypatch.setattr(
        routes,
        "run_analysis",
        lambda *a, **k: build_report(timestamp=datetime.now(), conclusion="新分析"),
    )
    body = client.post("/api/v1/analyze", json={"symbol": "600519"}).json()

    assert body["from_cache"] is False
    assert body["report"]["conclusion"] == "新分析"


def test_analyze_does_not_persist_when_pipeline_fails(client, store, monkeypatch):
    def boom(*args, **kwargs):
        raise InvalidSymbolError("无法自动识别")

    monkeypatch.setattr(routes, "run_analysis", boom)
    resp = client.post("/api/v1/analyze", json={"symbol": "600519"})

    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_SYMBOL"
    assert client.get("/api/v1/history", params={"symbol": "600519"}).json()["total"] == 0


def test_analyze_maps_legacy_value_error_to_400(client, monkeypatch):
    """存量代码抛出的 ValueError 仍按 400 处理，维持改造前的契约。"""

    def boom(*args, **kwargs):
        raise ValueError("某个参数不合法")

    monkeypatch.setattr(routes, "run_analysis", boom)
    resp = client.post("/api/v1/analyze", json={"symbol": "600519"})

    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_REQUEST"
    assert resp.json()["detail"] == "某个参数不合法"


def test_analyze_propagates_market_error_verbatim(client, monkeypatch):
    def boom(*args, **kwargs):
        raise UnsupportedMarketError("不支持的市场类型: mars")

    monkeypatch.setattr(routes, "run_analysis", boom)
    resp = client.post("/api/v1/analyze", json={"symbol": "600519", "market": "a_share"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "UNSUPPORTED_MARKET"
    assert resp.json()["detail"] == "不支持的市场类型: mars"

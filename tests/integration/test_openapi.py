"""OpenAPI 文档的契约。

这些断言看起来像在测文档，实际是在钉住**对外契约**：错误码集合、响应内容类型、
示例是否可用。它们一旦漂移，生成出来的客户端就会按错误的形状实现。
"""

import pytest

pytestmark = pytest.mark.integration

BUSINESS_ENDPOINTS = {
    ("/api/v1/analyze", "post"): {"200", "400", "422", "500"},
    ("/api/v1/history", "get"): {"200", "422", "500"},
    ("/api/v1/search", "get"): {"200", "422", "500"},
    ("/api/v1/export/{symbol}", "get"): {"200", "404", "422", "500"},
}


@pytest.fixture
def spec(client) -> dict:
    return client.get("/openapi.json").json()


def test_docs_is_reachable(client):
    assert client.get("/docs").status_code == 200


@pytest.mark.parametrize(("path", "method"), sorted(BUSINESS_ENDPOINTS))
def test_each_endpoint_declares_its_error_responses(spec, path, method):
    declared = set(spec["paths"][path][method]["responses"])
    assert declared == BUSINESS_ENDPOINTS[(path, method)]


@pytest.mark.parametrize(("path", "method"), sorted(BUSINESS_ENDPOINTS))
def test_error_responses_reference_the_shared_model(spec, path, method):
    """错误体必须是同一个 ErrorResponse，而不是各端点各写一份形状。"""
    responses = spec["paths"][path][method]["responses"]
    for code in ("400", "404", "422", "500"):
        if code not in responses:
            continue
        schema = responses[code]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/ErrorResponse"}, f"{path} {code}"


def test_error_response_model_has_description_and_example(spec):
    model = spec["components"]["schemas"]["ErrorResponse"]
    assert set(model["properties"]) == {"detail", "code", "request_id"}
    assert model["examples"], "错误体没有示例，/docs 里看不出实际形状"


@pytest.mark.parametrize(
    "name", ["AnalysisRequest", "HistoryResponse", "SearchResponse", "AnalysisReport"]
)
def test_request_and_response_models_carry_examples(spec, name):
    model = spec["components"]["schemas"][name]
    has_example = bool(model.get("examples")) or any(
        prop.get("examples") for prop in model.get("properties", {}).values()
    )
    assert has_example, f"{name} 没有可用示例"


def test_export_documents_only_markdown(spec):
    """这个端点从不返回 JSON。

    不设 response_class 时 FastAPI 会给 200 补一个默认 application/json，
    那是错的 —— 生成出来的客户端会以为能解析 JSON。
    """
    content = spec["paths"]["/api/v1/export/{symbol}"]["get"]["responses"]["200"]["content"]
    assert list(content) == ["text/markdown"]
    assert content["text/markdown"]["schema"] == {"type": "string"}


def test_health_endpoints_have_typed_responses(spec):
    health = spec["paths"]["/health"]["get"]["responses"]
    assert health["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HealthResponse"
    }

    ready = spec["paths"]["/health/ready"]["get"]["responses"]
    assert set(ready) == {"200", "503"}
    assert ready["503"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ReadyResponse"
    }


def test_status_fields_are_enumerated(spec):
    """status 用 Literal 声明，/docs 才能列出全部可能取值，而不是"任意字符串"。"""
    health = spec["components"]["schemas"]["HealthResponse"]["properties"]["status"]
    assert set(health["enum"]) == {"ok", "degraded"}
    ready = spec["components"]["schemas"]["ReadyResponse"]["properties"]["status"]
    assert set(ready["enum"]) == {"ready", "not_ready"}


def test_tags_have_descriptions(spec):
    tags = {tag["name"]: tag["description"] for tag in spec.get("tags", [])}
    assert set(tags) == {"analysis", "history", "system"}
    assert all(description for description in tags.values())


def test_every_operation_has_summary_and_description(spec):
    """逐个端点检查，避免新增端点时漏写文档。"""
    missing = []
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            if not operation.get("summary") or not operation.get("description"):
                missing.append(f"{method.upper()} {path}")
    assert not missing, f"这些端点缺 summary 或 description：{missing}"


def test_query_parameters_are_documented(spec):
    """参数说明不能只有 key 名。"""
    for path, method in (("/api/v1/history", "get"), ("/api/v1/search", "get")):
        params = spec["paths"][path][method].get("parameters", [])
        assert params, f"{path} 没有声明参数"
        for param in params:
            assert param.get("description"), f"{path} 的参数 {param['name']} 缺说明"


def test_metrics_stays_out_of_the_business_api(spec):
    assert "/metrics" not in spec["paths"]


@pytest.mark.parametrize(
    "name", ["AnalysisRequest", "AnalysisReport", "HistoryResponse", "SearchResponse"]
)
def test_declared_examples_are_schema_valid(spec, name):
    """示例必须能通过模型校验。

    写错的示例比没有示例更糟：/docs 里会展示一个根本不合法的请求体/响应体。
    """
    from backend.schemas import models as schemas

    model = getattr(schemas, name)
    for example in spec["components"]["schemas"][name].get("examples", []):
        model.model_validate(example)

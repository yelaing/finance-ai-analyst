import pytest
from pydantic import SecretStr, ValidationError

from backend.config import Settings, get_settings


def make(**env) -> Settings:
    """不读 .env，只按传入项构造 —— 否则断言会依赖开发者本机的 .env。"""
    return Settings(_env_file=None, **env)


def test_defaults(monkeypatch):
    # 显式清掉代理变量：否则断言会依赖跑测试的机器/CI 是否设了它们
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)

    s = make()
    assert s.env == "dev"
    assert s.is_prod is False
    assert s.log_format is None
    assert s.backend_port == 8000
    assert s.cors_origins == ["*"]
    assert s.http_proxy is None
    assert s.no_proxy is None
    assert s.llm_timeout == 30.0


def test_prod_derives_json_log_format():
    s = make(env="prod")
    assert s.is_prod is True
    assert s.resolved_log_format == "json"


def test_dev_derives_text_log_format():
    assert make(env="dev").resolved_log_format == "text"


@pytest.mark.parametrize(
    ("env", "log_format", "expected"),
    [("prod", "text", "text"), ("dev", "json", "json")],
)
def test_explicit_log_format_beats_env_derivation(env, log_format, expected):
    assert make(env=env, log_format=log_format).resolved_log_format == expected


def test_api_key_is_secret_and_masked():
    s = make(llm_api_key="sk-super-secret-value")
    assert isinstance(s.llm_api_key, SecretStr)
    assert s.llm_api_key.get_secret_value() == "sk-super-secret-value"
    # 关键：把 settings 打进日志时不能泄漏明文
    assert "sk-super-secret-value" not in repr(s)
    assert "sk-super-secret-value" not in str(s)


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen-max")
    monkeypatch.setenv("BACKEND_PORT", "9999")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1080")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = make()
    assert s.llm_model == "qwen-max"
    assert s.backend_port == 9999
    assert s.http_proxy == "http://127.0.0.1:1080"
    assert s.log_level == "DEBUG"


def test_cors_origins_accepts_json_array(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", '["http://a", "http://b"]')
    assert make().cors_origins == ["http://a", "http://b"]


def test_unknown_env_value_rejected():
    with pytest.raises(ValidationError):
        make(env="staging")


def test_unknown_env_var_is_ignored(monkeypatch):
    """extra='ignore' —— 多余的变量不该让启动失败（容器里常注入一堆无关变量）。"""
    monkeypatch.setenv("SOME_UNRELATED_VAR", "whatever")
    assert make().env == "dev"


def test_get_settings_is_cached():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()

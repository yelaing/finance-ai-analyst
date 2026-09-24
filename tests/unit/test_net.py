"""启动时应用网络环境变量。

重点是「只在启动时」：requests / httpx 都在请求时读这些变量，运行期修改它们在
多线程下就是竞态（历史上 technical.py 为了 A 股绕过代理就那么干过）。
"""

import logging
import os

import pytest

from backend.config import Settings
from backend.core.net import apply_network_env

PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")


@pytest.fixture
def clean_env(monkeypatch):
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)


def test_applies_configured_values(clean_env):
    apply_network_env(
        Settings(_env_file=None, http_proxy="http://proxy:8080", no_proxy="a.com,b.com")
    )

    assert os.environ["HTTP_PROXY"] == "http://proxy:8080"
    assert os.environ["HTTPS_PROXY"] == "http://proxy:8080"
    assert os.environ["NO_PROXY"] == "a.com,b.com"


def test_blank_settings_leave_environment_untouched(clean_env, monkeypatch):
    """留空表示「不干预」——交给环境原有的配置（比如系统代理），不能写成空字符串。"""
    monkeypatch.setenv("HTTP_PROXY", "http://from-system:1")

    apply_network_env(Settings(_env_file=None, http_proxy=None, no_proxy=None))

    assert os.environ["HTTP_PROXY"] == "http://from-system:1"
    assert "NO_PROXY" not in os.environ


def test_does_not_log_credential_bearing_values(clean_env, caplog):
    """代理 URL 可能形如 http://user:pass@host，值不进 INFO 日志。"""
    settings = Settings(_env_file=None, http_proxy="http://user:s3cret@proxy:8080")

    with caplog.at_level(logging.INFO):
        apply_network_env(settings)

    assert "HTTP_PROXY" in caplog.text
    assert "s3cret" not in caplog.text


def test_is_idempotent(clean_env):
    settings = Settings(_env_file=None, no_proxy="a.com")

    apply_network_env(settings)
    apply_network_env(settings)

    assert os.environ["NO_PROXY"] == "a.com"


def test_no_log_when_nothing_configured(clean_env, caplog):
    with caplog.at_level(logging.INFO):
        apply_network_env(Settings(_env_file=None))

    assert "已应用网络环境变量" not in caplog.text

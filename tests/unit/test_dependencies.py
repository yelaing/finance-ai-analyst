"""守护性测试：代码里直接 import 的第三方包，必须在 requirements 中声明。

背景：cachetools 被坑过一次 —— 本机由旧版 streamlit 的传递依赖带入，而 CI 装到
新版 streamlit 后它不再被依赖，于是 `from cachetools import TTLCache` 直接失败。
这类问题只在干净环境暴露，本地永远绿；所以用静态检查提前挡住。

不引入 packaging 来解析依赖名（那会让「检查依赖的测试」自己多一个依赖）。
"""

import ast
import pathlib
import re
import sys
from importlib.metadata import packages_distributions

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "backend"
REQUIREMENT_FILES = ("requirements.txt", "requirements-dev.txt")


def imported_top_level_modules() -> set[str]:
    modules: set[str] = set()
    for path in BACKEND.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def declared_distributions() -> set[str]:
    names: set[str] = set()
    for filename in REQUIREMENT_FILES:
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r")):
                continue
            names.add(re.split(r"[<>=!~\[;]", line, maxsplit=1)[0].strip().lower())
    return names


def test_every_imported_distribution_is_declared():
    declared = declared_distributions()
    mapping = packages_distributions()
    external = sorted(
        module
        for module in imported_top_level_modules()
        if module not in sys.stdlib_module_names and module != "backend"
    )
    assert external, "没有解析到任何第三方 import，静态分析大概坏了"

    unmapped = []
    undeclared = {}
    for module in external:
        distributions = mapping.get(module)
        if not distributions:
            unmapped.append(module)
            continue
        if not any(dist.lower() in declared for dist in distributions):
            undeclared[module] = distributions

    assert not unmapped, f"这些模块名无法映射到任何发行包：{unmapped}"
    assert not undeclared, (
        f"这些包被直接 import 却没在 requirements 里声明（本地靠传递依赖侥幸可用）：{undeclared}"
    )


def test_requirements_files_are_readable():
    """避免上一条测试因为文件路径写错而「空集合通过」这种假绿。"""
    for filename in REQUIREMENT_FILES:
        assert (ROOT / filename).is_file(), f"{filename} 不存在"
    assert len(declared_distributions()) >= 10

"""采集真实 A 股报告，作为检索/生成评测的固定快照。

为什么单独做采集：评测要能复现。每次跑评测都重新拉数据、重新跑四阶段流水线，
既贵又会让报告 ID、数值和标签漂移。这里把真实数据抓一次、落成
`tests/fixtures/eval_reports/`，评测只读快照；换数据时显式重跑本脚本并复核标签。

用法：

    python tests/eval_collect.py                 # 采到 30 份
    python tests/eval_collect.py --only 600519   # 只采一只，做烟测
    python tests/eval_collect.py --force         # 重采并覆盖同 symbol 的旧记录

数据源全部走项目已有的 fetch_stock_data（东方财富不可达时自动退到巨潮/同花顺），
报告走项目已有的四阶段流水线；没有一条数据是手工编的。
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "eval_reports"
REPORTS_FILE = DATA_DIR / "reports.json"
INPUTS_DIR = DATA_DIR / "inputs"
MANIFEST_FILE = DATA_DIR / "manifest.json"

# 按行业交错排列，前 30 个如果都健康，四个行业各 7 家，另有 2 家其他行业。
# 后面的公司是备选：某个行业的数据源失败时，继续往后补，不硬凑。
UNIVERSE: list[tuple[str, str]] = [
    ("600519", "白酒"),
    ("601318", "金融"),
    ("002594", "新能源"),
    ("600276", "医药"),
    ("000858", "白酒"),
    ("600036", "金融"),
    ("300750", "新能源"),
    ("300760", "医药"),
    ("000568", "白酒"),
    ("601166", "金融"),
    ("601012", "新能源"),
    ("603259", "医药"),
    ("600809", "白酒"),
    ("600030", "金融"),
    ("002459", "新能源"),
    ("000538", "医药"),
    ("002304", "白酒"),
    ("601398", "金融"),
    ("688223", "新能源"),
    ("600436", "医药"),
    ("603369", "白酒"),
    ("601288", "金融"),
    ("300274", "新能源"),
    ("300122", "医药"),
    ("000596", "白酒"),
    ("601601", "金融"),
    ("002129", "新能源"),
    ("600196", "医药"),
    ("000333", "家电"),
    ("600887", "食品饮料"),
    ("601088", "煤炭"),
    ("600019", "钢铁"),
    ("600941", "通信"),
    ("000651", "家电"),
    ("600900", "公用事业"),
    ("601899", "有色"),
    ("601988", "金融"),
    ("300014", "新能源"),
    ("000661", "医药"),
    ("600585", "建材"),
]

DEFAULT_TARGET = 30

logger = logging.getLogger("eval_collect")


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload) -> None:
    """先写临时文件再替换，采集被中断时不会留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_failures(manifest: dict) -> dict[str, dict]:
    """manifest 里 failures 落盘是列表（方便人读），内存里用 dict（方便按 symbol 增删）。"""
    raw = manifest.get("failures", {})
    if isinstance(raw, list):
        return {item["symbol"]: item for item in raw if item.get("symbol")}
    return dict(raw)


def _pending_symbols(
    universe: list[tuple[str, str]],
    completed: set[str],
    only: bool,
    force: bool,
) -> list[tuple[str, str]]:
    """算出这一轮真正要采集的代码。

    `--only` 是重采指定标的的逃生口：即使候选池已经采满，
    也必须按 force 语义处理这几只，不能被 target 提前截断。
    """
    return [(symbol, category) for symbol, category in universe if force or symbol not in completed]


def _latest_by_symbol(records: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for record in records:
        symbol = record.get("symbol")
        if not symbol:
            continue
        current = latest.get(symbol)
        if current is None or str(record.get("timestamp", "")) > str(current.get("timestamp", "")):
            latest[symbol] = record
    return latest


def _parse_only(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _collect_one(symbol: str, category: str) -> dict:
    """抓真实数据、跑四阶段流水线，返回报告记录与原始输入。"""
    from backend.pipeline.analyzer import run_analysis
    from backend.pipeline.data_fetcher import _is_healthy, fetch_stock_data
    from backend.store.chroma_store import build_record

    fetched_at = datetime.now().isoformat(timespec="seconds")
    fetched = fetch_stock_data(symbol, "a_share")
    if not _is_healthy("a_share", fetched):
        raise RuntimeError(f"必需数据源未全部成功：{fetched.sources}")
    if not fetched.financial_history_text:
        raise RuntimeError(f"财务摘要没有可用近三年数据：{fetched.sources}")

    report = run_analysis(symbol, "a_share", include_sentiment=True)
    if not report.financial_history:
        raise RuntimeError("报告没有携带近三年财务快照")
    record = build_record(report)

    raw_input = {
        "symbol": symbol,
        "name": report.name,
        "category": category,
        "fetched_at": fetched_at,
        "sources": fetched.sources,
        "financial_text": fetched.financial_text,
        "financial_history_text": fetched.financial_history_text,
        "news_text": fetched.news_text,
        "report_id": record["id"],
        "report_timestamp": record["timestamp"],
    }
    return record, raw_input


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="采集真实 A 股评测快照")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="目标报告数")
    parser.add_argument("--only", help="只采这些代码，逗号分隔，例如 600519,000858")
    parser.add_argument("--force", action="store_true", help="已存在的 symbol 也重新采集")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # 巨潮公告接口内部用 tqdm；采集日志里不需要进度条
    os.environ.setdefault("TQDM_DISABLE", "1")

    # 代理必须在 OpenAI client / akshare 请求之前写进环境变量
    from backend.config import get_settings
    from backend.core.net import apply_network_env

    apply_network_env(get_settings())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    universe = UNIVERSE
    only = _parse_only(args.only)
    if only:
        by_symbol = dict(UNIVERSE)
        missing = [symbol for symbol in only if symbol not in by_symbol]
        if missing:
            parser.error(f"这些代码不在候选池里：{missing}")
        universe = [(symbol, by_symbol[symbol]) for symbol in only]
        target = len(only)
    else:
        target = args.target

    records = _read_json(REPORTS_FILE, [])
    manifest = _read_json(MANIFEST_FILE, {"failures": {}})
    failures = _load_failures(manifest)
    collected = _latest_by_symbol(records)
    universe_symbols = {symbol for symbol, _ in universe}

    # 只统计候选池内的报告，避免历史遗留的其他 symbol 把目标数占满
    completed = {symbol for symbol in collected if symbol in universe_symbols}

    logger.info("已有 %d/%d 份候选池报告，目标 %d", len(completed), len(universe), target)
    for symbol, category in _pending_symbols(universe, completed, bool(only), args.force):
        if not only and len(completed) >= target:
            break
        logger.info("[start] %s %s", symbol, category)
        started = time.perf_counter()
        try:
            record, raw_input = _collect_one(symbol, category)
        except Exception as exc:
            failures[symbol] = {
                "symbol": symbol,
                "category": category,
                "reason": f"{type(exc).__name__}: {exc}",
                "at": datetime.now().isoformat(timespec="seconds"),
            }
            logger.warning("[fail] %s %s: %s", symbol, category, exc)
            _write_json(MANIFEST_FILE, _build_manifest(records, failures))
            continue

        # 一个 symbol 只保留最新一份，避免重采后索引里出现重复文档
        records = [item for item in records if item.get("symbol") != symbol]
        records.append(record)
        collected[symbol] = record
        completed.add(symbol)
        failures.pop(symbol, None)

        _write_json(REPORTS_FILE, records)
        _write_json(INPUTS_DIR / f"{symbol}.json", raw_input)
        _write_json(MANIFEST_FILE, _build_manifest(records, failures))

        elapsed = time.perf_counter() - started
        logger.info(
            "[ok] %s %s -> %s（%.1fs，累计 %d/%d）",
            symbol,
            raw_input["name"],
            record["id"],
            elapsed,
            len(completed),
            target,
        )

    counts: dict[str, int] = {}
    for symbol, category in universe:
        if symbol in completed:
            counts[category] = counts.get(category, 0) + 1
    logger.info("完成 %d/%d；分行业：%s", len(completed), target, counts)
    if len(completed) < target:
        logger.error("没有达到目标数，失败的候选见 %s", MANIFEST_FILE)
        return 1
    return 0


def _build_manifest(records: list[dict], failures: dict[str, dict]) -> dict:
    latest = _latest_by_symbol(records)
    universe_symbols = {symbol for symbol, _ in UNIVERSE}
    symbols = [
        {
            "symbol": symbol,
            "name": latest[symbol].get("name"),
            "category": next(category for item, category in UNIVERSE if item == symbol),
            "report_id": latest[symbol]["id"],
            "timestamp": latest[symbol]["timestamp"],
            "sources": latest[symbol].get("sources", []),
        }
        for symbol, _ in UNIVERSE
        if symbol in latest and symbol in universe_symbols
    ]
    try:
        import akshare

        akshare_version = getattr(akshare, "__version__", "unknown")
    except Exception:
        akshare_version = "unknown"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "tests/eval_collect.py",
        "snapshot_note": "真实 A 股数据；报告由现有四阶段流水线生成，未人工改写",
        "akshare_version": akshare_version,
        "symbols": symbols,
        "failures": cases_sorted(failures),
    }


def cases_sorted(failures: dict[str, dict]) -> list[dict]:
    return [failures[symbol] for symbol in sorted(failures)]


if __name__ == "__main__":
    sys.exit(main())

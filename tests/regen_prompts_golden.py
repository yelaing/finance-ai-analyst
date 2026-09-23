"""重新生成 prompt golden 基线。

只在**有意修改 prompt** 之后运行，并把生成结果一起提交，让改动进 code review：

    python tests/regen_prompts_golden.py

非预期地需要重跑，往往说明 prompt 被误改了 —— 先看清 diff 再决定。
"""

import json
import sys
from pathlib import Path

# 直接 `python tests/regen_prompts_golden.py` 时 sys.path[0] 是 tests/，
# 需要把仓库根补上才能 import tests.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.unit.test_prompts import TEMPLATES, render  # noqa: E402

TARGET = Path(__file__).resolve().parent / "fixtures" / "prompts_expected.json"


def main() -> None:
    payload = {name: render(name) for name in TEMPLATES}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"已写入 {TARGET}（{len(payload)} 个模板）")


if __name__ == "__main__":
    main()

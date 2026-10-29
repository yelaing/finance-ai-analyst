ifeq ($(OS),Windows_NT)
PYTHON := py -3
else
PYTHON := python3
endif

.PHONY: test eval eval-retrieval eval-generation eval-data

test:
	$(PYTHON) -m pytest

# 一条命令跑完：检索 80 条 + 生成 30 条，读固定快照，不重新采集数据。
eval:
	$(PYTHON) tests/eval_suite.py

eval-retrieval:
	$(PYTHON) tests/eval_retrieval.py

eval-generation:
	$(PYTHON) tests/eval_generation.py

# 重新拉真实数据并跑四阶段流水线；会真实调用 LLM，耗时且产生费用。
eval-data:
	$(PYTHON) tests/eval_collect.py

PYTHON ?= .venv/bin/python

.PHONY: test check

test:
	$(PYTHON) -m unittest discover -s tests -v

check:
	$(PYTHON) -m compileall -q src tests scripts train_llm.py

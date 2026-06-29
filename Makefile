.PHONY: init start

PYTHON ?= python3

init:
	$(PYTHON) -m pip install -r requirements.txt

start:
	$(PYTHON) app.py

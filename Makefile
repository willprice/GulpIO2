export PYTHONPATH := $(PWD)/src
export PYTHONNOUSERSITE=1

.PHONY: test
test:
	python -m pytest test/unit
	python test/integration/system_tests.py

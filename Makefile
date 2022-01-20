install: requirements.txt
	pip install --upgrade pip build &&\
		pip install -r requirements.txt

build:
	python -m build

test:
	python -m pytest -vv --cov

format:
	black *.py

lint:
	pylint --disable=R,C,E1101,W0221 src tests

clean:
	rm -rf __pycache__ .coverage dist

all: install lint test build

.PHONY: lint clean all
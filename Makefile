PYTHON?=python3
VENV_NAME?=venv
SRC?=src

.PHONY: setup check clean help

setup:
	@if [ ! -d "$(VENV_NAME)" ]; then \
	  $(PYTHON) -m venv $(VENV_NAME); \
	fi
	$(VENV_NAME)/bin/pip install --upgrade pip
	@if [ -f requirements.txt ]; then \
	  $(VENV_NAME)/bin/pip install -r requirements.txt; \
	fi
	@if [ -f requirements-dev.txt ]; then \
	  $(VENV_NAME)/bin/pip install -r requirements-dev.txt; \
	fi

check:
	-$(VENV_NAME)/bin/python utils/end_of_file_fixer.py $(SRC)/* evaluation/*
	-$(VENV_NAME)/bin/python utils/trailing_whitespace_fixer.py $(SRC)/* evaluation/*
	@if [ -f requirements.txt ]; then \
		$(VENV_NAME)/bin/python utils/requirements_txt_fixer.py requirements.txt; \
	fi
	@if [ -f requirements-dev.txt ]; then \
		$(VENV_NAME)/bin/python utils/requirements_txt_fixer.py requirements-dev.txt; \
	fi

	$(VENV_NAME)/bin/isort --profile black ./$(SRC) ./evaluation
	$(VENV_NAME)/bin/black ./$(SRC) ./evaluation
	$(VENV_NAME)/bin/flake8 --ignore=E501 ./$(SRC) ./evaluation
	$(VENV_NAME)/bin/mypy --config-file mypy.ini ./$(SRC)

	@echo "\033[1;32m✨ All checks passed successfully! ✨\033[0m"

clean:
	@find ./src ./evaluation -type d -name "__pycache__" -exec rm -r {} +

help:
	@echo "\033[1;36mAvailable commands :\033[0m"
	@echo "	\033[1;32mmake setup\033[0m    : Create the virtual environment and install dependencies."
	@echo "	\033[1;32mmake check\033[0m    : Check and format code (quality, style, mypy, etc.)."
	@echo "	\033[1;32mmake clean\033[0m    : Remove all __pycache__ directories."
	@echo "	\033[1;32mmake help\033[0m     : Show this help message."

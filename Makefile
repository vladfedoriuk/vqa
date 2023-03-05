 # We want bash behaviour in all shell invocations
SHELL := bash
# Run each target in a separate shell
.ONESHELL:
 # Fail on error inside any functions or subshells
.SHELLFLAGS := -eu -o pipefail -c
 # Remove partially created files on error
.DELETE_ON_ERROR:
 # Warn when an undefined variable is referenced
MAKEFLAGS += --warn-undefined-variables
# Disable built-in rules
MAKEFLAGS += --no-builtin-rules
# A catalog of requirements files
REQUIREMENTS?=requirements

requirements-base: # Compile base requirements
	python -m piptools compile \
	-v \
	--resolver backtracking \
	--output-file=requirements/base.txt \
	pyproject.toml

requirements-dev: requirements-base # Compile dev requirements
	python -m piptools compile \
	-v \
	--resolver backtracking \
	--extra=dev \
	--output-file=requirements/dev.txt \
	pyproject.toml

requirements: requirements-base requirements-dev
.PHONY: requirements

install-base:  # Install the app locally
	python -m pip install -r $(REQUIREMENTS)/base.txt .
.PHONY: install-base

install-dev: install-base # Install the app locally with dev dependencies
	python -m pip install \
		-r $(REQUIREMENTS)/dev.txt \
		--editable .
.PHONY: install-dev

init-dev: install-dev # Install the app locally with dev dependencies and install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg
.PHONY: init-dev

.DEFAULT_GOAL := init-dev # Default goal is init-dev

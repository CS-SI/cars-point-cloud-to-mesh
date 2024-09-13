# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# Dependencies : python3 venv
# Some Makefile global variables can be set in make command line
# Recall: .PHONY  defines special targets not associated with files

############### GLOBAL VARIABLES ######################
.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
# Exemple: VENV="other-venv/" make install
ifndef VENV
	VENV = "venv"
endif

# Browser definition
define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

# Set Docker build version
# Exemple: DOCKER_BUILD_VERSION="0.1.0" make docker
ifndef DOCKER_BUILD_VERSION
	DOCKER_BUILD_VERSION = "dev"
endif

# Check Docker
CHECK_DOCKER = $(shell docker -v)

# Python global variables definition
PYTHON_VERSION_MIN = 3.8

PYTHON=$(shell command -v python3)

PYTHON_VERSION_CUR=$(shell $(PYTHON) -c 'import sys; print("%d.%d"% sys.version_info[0:2])')
PYTHON_VERSION_OK=$(shell $(PYTHON) -c 'import sys; cur_ver = sys.version_info[0:2]; min_ver = tuple(map(int, "$(PYTHON_VERSION_MIN)".split("."))); print(int(cur_ver >= min_ver))')

############### Check python version supported ############

ifeq (, $(PYTHON))
    $(error "PYTHON=$(PYTHON) not found in $(PATH)")
endif

ifeq ($(PYTHON_VERSION_OK), 0)
    $(error "Requires python version >= $(PYTHON_VERSION_MIN). Current version is $(PYTHON_VERSION_CUR)")
endif


################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | sort

## Install section

.PHONY: git
git: ## init local git repository if not present
	@test -d .git/ || git init .

# Check python install in VENV
CHECK_NUMPY = $(shell ${VENV}/bin/python -m pip list|grep numpy)
CHECK_FIONA = $(shell ${VENV}/bin/python -m pip list|grep Fiona)
CHECK_RASTERIO = $(shell ${VENV}/bin/python -m pip list|grep rasterio)
CHECK_TBB = $(shell ${VENV}/bin/python -m pip list|grep tbb)
CHECK_NUMBA = $(shell ${VENV}/bin/python -m pip list|grep numba)


.PHONY: venv
venv: ## create virtualenv in "venv" dir if not exists
	@test -d ${VENV} || python3 -m venv ${VENV}
	@${VENV}/bin/python -m pip install --upgrade pip setuptools wheel # no check to upgrade each time
	@touch ${VENV}/bin/activate

.PHONY: install-deps
install-deps: venv ## install python libs
	@[ "${CHECK_NUMPY}" ] ||${VENV}/bin/python -m pip install --upgrade cython numpy
	@[ "${CHECK_NUMBA}" ] ||${VENV}/bin/python -m pip install --upgrade numba


.PHONY: install-deps-gdal
install-deps-gdal: install-deps ## create an healthy python environment for GDAL/ proj
	@[ "${CHECK_FIONA}" ] ||${VENV}/bin/python -m pip install --no-binary fiona fiona
	@[ "${CHECK_RASTERIO}" ] ||${VENV}/bin/python -m pip install --no-binary rasterio rasterio



.PHONY: install
install: venv git  ## install the package in dev mode in virtualenv
	@test -f ${VENV}/bin/cars_point_cloud_to_mesh || echo "Install cars_point_cloud_to_mesh package from local directory"
	@test -f ${VENV}/bin/cars_point_cloud_to_mesh || ${VENV}/bin/python -m pip install -e .[dev,docs,notebook]
	@echo "patch CARS 0.8.0"
	@patch --forward -p1 ${VENV}/lib/python3.8/site-packages/cars/applications/point_cloud_fusion/pc_tif_tools.py ./tests/cars_patch_data/patch_file_cars_pc_tif_tools
	@test -f .git/hooks/pre-commit || echo "Install pre-commit"
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit
	@chmod +x ${VENV}/bin/register-python-argcomplete
	@echo "cars_point_cloud_to_mesh installed in dev mode in virtualenv ${VENV} with documentation"
	@echo " cars_point_cloud_to_mesh venv usage : source ${VENV}/bin/activate; cars_point_cloud_to_mesh -h"

.PHONY: install-dev
install-dev: install-deps ## install cars in dev editable mode (pip install -e .) without recompiling otb remote modules, rasterio, fiona
	@test -f ${VENV}/bin/cars || ${VENV}/bin/pip install -e .[dev,docs,notebook]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${VENV}/bin/pre-commit install -t pre-push

## Test section
	
.PHONY: test
test: install ## run tests and coverage quickly with the default Python (source venv before)
	@${VENV}/bin/pytest -o log_cli=true --cov-config=.coveragerc --cov --cov-report=term-missing

.PHONY: test-all
test-all: install ## run tests on every Python version with tox (source venv before)
	@${VENV}/bin/tox -r -p auto  ## recreate venv (-r) and parallel mode (-p auto)
	
.PHONY: coverage
coverage: install ## check code coverage quickly with the default Python
	@${VENV}/bin/coverage run --source cars_point_cloud_to_mesh -m pytest
	@${VENV}/bin/coverage report -m
	@${VENV}/bin/coverage html
	$(BROWSER) htmlcov/index.html

## Code quality, linting section

### Format with isort and black

.PHONY: format
format: install format/isort format/black  ## run black and isort formatting (depends install)

.PHONY: format/isort
format/isort: install  ## run isort formatting (depends install)
	@echo "+ $@"
	@${VENV}/bin/isort cars_point_cloud_to_mesh tests

.PHONY: format/black
format/black: install  ## run black formatting (depends install)
	@echo "+ $@"
	@${VENV}/bin/black cars_point_cloud_to_mesh tests

### Check code quality and linting : isort, black, flake8, pylint

.PHONY: lint
lint: install lint/isort lint/black lint/flake8 lint/pylint ## check code quality and linting (source venv before)

.PHONY: lint/isort
lint/isort: ## check imports style with isort
	@echo "+ $@"
	@${VENV}/bin/isort --check cars_point_cloud_to_mesh tests
	
.PHONY: lint/black
lint/black: ## check global style with black
	@echo "+ $@"
	@${VENV}/bin/black --check cars_point_cloud_to_mesh tests

.PHONY: lint/flake8
lint/flake8: ## check linting with flake8
	@echo "+ $@"
	@${VENV}/bin/flake8 cars_point_cloud_to_mesh tests

.PHONY: lint/pylint
lint/pylint: ## check linting with pylint
	@echo "+ $@"
	@set -o pipefail; ${VENV}/bin/pylint cars_point_cloud_to_mesh tests --rcfile=.pylintrc --output-format=parseable | tee pylint-report.txt # pipefail to propagate pylint exit code in bash

## Documentation section

.PHONY: docs
docs: install ## generate Sphinx HTML documentation, including API docs
	@${VENV}/bin/sphinx-build -M clean docs/source/ docs/build
	@${VENV}/bin/sphinx-build -M html docs/source/ docs/build -W --keep-going
	$(BROWSER) docs/build/html/index.html

## Notebook section

.PHONY: notebook
notebook: install ## Install Jupyter notebook kernel with venv
	@echo "Install Jupyter Kernel and launch Jupyter notebooks environment"
	@${VENV}/bin/python -m ipykernel install --sys-prefix --name=cars_point_cloud_to_mesh$(VENV) --display-name=cars_point_cloud_to_mesh$(VERSION)
	@echo " --> After virtualenv activation, please use following command to launch local jupyter notebook to open Notebooks:"
	@echo "jupyter notebook"

.PHONY: notebook-clean-output ## Clean Jupyter notebooks outputs
notebook-clean-output:
	@echo "Clean Jupyter notebooks"
	@${VENV}/bin/jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb

## Docker section

docker: git ## Build docker image (and check Dockerfile)
	@[ "${CHECK_DOCKER}" ] || ( echo ">> docker not found"; exit 1 )
	@echo "Check Dockerfile with hadolint"
	@docker pull hadolint/hadolint
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@echo "Build Docker image cars_point_cloud_to_mesh"
	@docker build -t cs group/cars_point_cloud_to_mesh:${DOCKER_BUILD_VERSION} -t cs group/cars_point_cloud_to_mesh:latest . -f Dockerfile

## Release section
	
.PHONY: dist
dist: clean install ## clean, install, builds source and wheel package
	@${VENV}/bin/python -m pip install --upgrade build
	@${VENV}/bin/python -m build
	ls -l dist

.PHONY: release
release: dist ## package and upload a release
	@${VENV}/bin/twine check dist/*
	@${VENV}/bin/twine upload dist/* --verbose ##  update your .pypirc accordingly

## Clean section

.PHONY: clean
clean: clean-venv clean-build clean-precommit clean-pyc clean-test clean-lint clean-docs clean-notebook ## clean all targets (except docker)

.PHONY: clean-venv
clean-venv: ## clean venv
	@echo "+ $@"
	@rm -rf ${VENV}

.PHONY: clean-build
clean-build: ## remove build artifacts
	@echo "+ $@"
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-precommit
clean-precommit: ## clean precommit hooks in .git/hooks
	@rm -f .git/hooks/pre-commit
	@rm -f .git/hooks/pre-push

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	@echo "+ $@"
	@find . -type f -name "*.py[co]" -exec rm -fr {} +
	@find . -type d -name "__pycache__" -exec rm -fr {} +
	@find . -name '*~' -exec rm -fr {} +

.PHONY: clean-test
clean-test: ## remove test, logging and coverage artifacts
	@echo "+ $@"
	@rm -fr .tox/
	@rm -f .coverage
	@rm -rf .coverage.*
	@rm -rf coverage.xml
	@rm -fr htmlcov/
	@rm -fr .pytest_cache
	@rm -f pytest-report.xml
	@rm -f debug.log

.PHONY: clean-lint
clean-lint: ## remove linting artifacts
	@echo "+ $@"
	@rm -f pylint-report.txt
	@rm -f pylint-report.xml
	@rm -rf .mypy_cache/

.PHONY: clean-docs
clean-docs: ## clean builded documentations
	@echo "+ $@"
	@rm -rf docs/build/
	@rm -rf docs/source/api_reference/
	@rm -rf docs/source/apidoc/

.PHONY: clean-notebook
clean-notebook: ## clean notebooks cache
	@echo "+ $@"
	@find . -type d -name ".ipynb_checkpoints" -exec rm -fr {} +

.PHONY: clean-docker
clean-docker: ## clean created docker images
		@echo "+ $@"
		@echo "Clean Docker image cars_point_cloud_to_mesh"
		@docker image rm cs group/cars_point_cloud_to_mesh:${DOCKER_BUILD_VERSION}
		@docker image rm cs group/cars_point_cloud_to_mesh:latest

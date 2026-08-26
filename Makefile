.PHONY: setup setup-uv install-deepclean install-deepclean-cluster lint clean

## Create the conda env (virgo-deepclean) from pyproject.toml and offer to
## fetch deepclean-prod. See scripts/setup_env.sh.
setup:
	scripts/setup_env.sh

## Same, but with uv instead of conda (creates .venv/, no conda required).
setup-uv:
	uv sync

## Install deepclean_prod for local/dev use (modern torch, --no-deps).
install-deepclean:
	scripts/install_deepclean_prod.sh --mode lite

## Install deepclean_prod the upstream way, for a LIGO/Virgo Linux cluster.
install-deepclean-cluster:
	scripts/install_deepclean_prod.sh --mode cluster

## Run the pre-commit checks (black, trailing whitespace, ...) on all files.
lint:
	pre-commit run --all-files

## Remove Python/Jupyter cache clutter (__pycache__, .ipynb_checkpoints).
clean:
	find . -type d -name "__pycache__" -not -path "./config/virgo-deepclean-prod/*" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -not -path "./config/virgo-deepclean-prod/*" -exec rm -rf {} +

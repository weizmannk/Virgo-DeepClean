#!/usr/bin/env bash
#
# One-shot environment setup for Virgo-DeepClean:
#   1. create a conda environment (just the Python interpreter)
#   2. install this project's dependencies from pyproject.toml
#   3. install pre-commit hooks
#   4. fetch + install the upstream DeepClean-prod pipeline
#
# pyproject.toml is the single source of truth for dependencies (also used
# by `uv sync`, see README.md) — conda here only provides the interpreter.
#
# Usage:
#   scripts/setup_env.sh [env_name]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-virgo-deepclean}"

if ! command -v conda &> /dev/null; then
    echo "conda not found. Install Miniconda/Anaconda first, or use" \
         "'uv sync' (see README.md) as a pip-based alternative." >&2
    exit 1
fi

if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "Creating conda environment '${ENV_NAME}' (python=3.10)..."
    conda create -y -n "${ENV_NAME}" python=3.10
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Installing dependencies from pyproject.toml..."
pip install "${REPO_ROOT}"

echo "Installing pre-commit hooks..."
pip install pre-commit
pre-commit install || true

read -r -p "Fetch and install the upstream deepclean-prod pipeline now? [y/N] " reply
if [[ "${reply}" =~ ^[Yy]$ ]]; then
    "${REPO_ROOT}/scripts/install_deepclean_prod.sh"
else
    echo "Skipped. Run scripts/install_deepclean_prod.sh manually when ready."
fi

echo "Environment '${ENV_NAME}' is ready. Activate it with: conda activate ${ENV_NAME}"

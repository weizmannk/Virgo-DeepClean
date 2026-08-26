#!/usr/bin/env bash
#
# Fetch and install the original DeepClean-prod pipeline that this repository
# builds on. DeepClean-prod is not published on PyPI, so it must be cloned
# from its upstream repository and installed.
#
# Upstream : https://git.ligo.org/tri.nguyen/deepclean-prod
# Authors  : Tri Nguyen, Michael Coughlin, Rich Ormiston, Rana Adhikari,
#            Gabriele Vajente, Erik Katsavounidis (maintained with Muhammed Saleem)
#
# Cloning requires a LIGO.ORG account with access to git.ligo.org.
#
# Two install modes are supported, because deepclean-prod's own
# environment.yml/requirements.txt are a ~2020 snapshot (Python 3.7,
# CUDA 10.2, torch==1.0.0) that only resolves on the platform it was built
# for:
#
#   --mode cluster   Recreate deepclean-prod's own conda environment from
#                     environment.yml and `pip install .` inside it, exactly
#                     as upstream documents. Use this on LIGO/Virgo Linux
#                     clusters (matching linux-64 + Python 3.7 + CUDA stack),
#                     where those pinned builds still resolve.
#
#   --mode lite       Install a modern torch into your *current* environment,
#   (default)         then `pip install -e . --no-deps` so the stale
#                     torch==1.0.0 pin is never resolved. Use this on a local
#                     machine / laptop (e.g. macOS) where the frozen 2020
#                     conda/CUDA stack cannot be reproduced.
#
# Usage:
#   scripts/install_deepclean_prod.sh [--mode cluster|lite] [source]
#
#   source  Optional path to a local deepclean-prod copy (directory or .zip),
#           used instead of cloning from git.ligo.org.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/config/virgo-deepclean-prod"
UPSTREAM_URL="https://git.ligo.org/tri.nguyen/deepclean-prod.git"

MODE="lite"
SOURCE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --mode=*)
            MODE="${1#--mode=}"
            shift
            ;;
        *)
            SOURCE="$1"
            shift
            ;;
    esac
done

if [ "${MODE}" != "cluster" ] && [ "${MODE}" != "lite" ]; then
    echo "Unknown --mode '${MODE}' (expected 'cluster' or 'lite')" >&2
    exit 1
fi

if [ -d "${DEST}/.git" ] || [ -f "${DEST}/setup.py" ]; then
    echo "deepclean-prod already present at ${DEST}, skipping fetch."
elif [ -n "${SOURCE}" ]; then
    echo "Copying local deepclean-prod from ${SOURCE} to ${DEST}..."
    rm -rf "${DEST}"
    if [ -f "${SOURCE}" ] && [[ "${SOURCE}" == *.zip ]]; then
        mkdir -p "${DEST}"
        unzip -q "${SOURCE}" -d "${REPO_ROOT}/config"
        # zip extracts to config/deepclean-prod ; move contents into DEST
        rsync -a "${REPO_ROOT}/config/deepclean-prod/" "${DEST}/"
        rm -rf "${REPO_ROOT}/config/deepclean-prod"
    else
        cp -r "${SOURCE}" "${DEST}"
    fi
else
    echo "Cloning deepclean-prod from ${UPSTREAM_URL}..."
    echo "(requires a LIGO.ORG account with access to git.ligo.org)"
    git clone "${UPSTREAM_URL}" "${DEST}"
fi

if [ "${MODE}" = "cluster" ]; then
    ENV_NAME="$(grep '^name:' "${DEST}/environment.yml" | awk '{print $2}')"
    echo "Recreating deepclean-prod's own conda environment '${ENV_NAME}' from environment.yml..."
    ( cd "${DEST}" && conda env create -f environment.yml )
    echo "Installing deepclean-prod into '${ENV_NAME}'..."
    conda run -n "${ENV_NAME}" pip install "${DEST}"
    echo "Done. Activate with: conda activate ${ENV_NAME}"
else
    # `uv sync`-created venvs don't ship a `pip` binary; fall back to `uv pip`
    # (it installs into the currently active venv) when there's no `pip`.
    if command -v pip &> /dev/null; then
        PIP=(pip)
    elif command -v uv &> /dev/null; then
        PIP=(uv pip)
    else
        echo "Neither 'pip' nor 'uv' found in PATH." >&2
        exit 1
    fi

    echo "Installing a modern torch into the current environment..."
    "${PIP[@]}" install torch

    # deepclean-prod's setup.py does `import pkg_resources` at build time.
    # setuptools >= 81 dropped pkg_resources, so pin an older setuptools and
    # build against it directly (--no-build-isolation) instead of letting an
    # isolated build env resolve the latest setuptools again.
    echo "Pinning a setuptools version that still ships pkg_resources..."
    "${PIP[@]}" install "setuptools<81"

    # deepclean_prod/logger.py hardcodes the pre-3.6 matplotlib style name
    # 'seaborn-colorblind', renamed to 'seaborn-v0_8-colorblind' upstream.
    # Patch the local vendored copy so import doesn't crash on a modern
    # matplotlib.
    LOGGER_FILE="${DEST}/deepclean_prod/logger.py"
    if [ -f "${LOGGER_FILE}" ] && grep -q "'seaborn-colorblind'" "${LOGGER_FILE}"; then
        echo "Patching deprecated matplotlib style name in deepclean_prod/logger.py..."
        sed -i.bak "s/'seaborn-colorblind'/'seaborn-v0_8-colorblind'/" "${LOGGER_FILE}"
        rm -f "${LOGGER_FILE}.bak"
    fi

    echo "Installing deepclean-prod (editable, --no-deps to keep the modern torch)..."
    "${PIP[@]}" install --no-build-isolation -e "${DEST}" --no-deps
    echo "Done. 'import deepclean_prod' should now work in this environment."
fi

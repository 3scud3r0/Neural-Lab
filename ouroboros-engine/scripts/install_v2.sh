#!/usr/bin/env bash
# Instala fontes upstream locais e o projeto, com verificação TLS preservada.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
python_command="${OUROBOROS_PYTHON:-python3}"
"$python_command" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), "Use Python 3.11 ou 3.12"'
venv_path="${OUROBOROS_VENV:-$project_root/.venv}"
if [[ ! -x "$venv_path/bin/python" ]]; then
    "$python_command" -m venv "$venv_path"
fi
python_path="$venv_path/bin/python"
"$python_path" -m pip install 'setuptools>=69' wheel
"$python_path" -m pip install 'torch==2.6.0' --index-url "${OUROBOROS_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
"$python_path" -m pip install --no-build-isolation -r requirements.txt
"$python_path" -m pip install --no-build-isolation -e .
"$python_path" scripts/run_v2.py doctor
printf '\nInstalação concluída. Python: %s\n' "$python_path"

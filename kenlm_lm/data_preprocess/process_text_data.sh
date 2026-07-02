#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lm_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${lm_root}/../.." && pwd)"
ROOT="${ROOT:-${repo_root}}"

python3 "${script_dir}/scripts/process_text_data.py" \
  --input-dir "${INPUT_DIR:-${ROOT}/data/ENX/control_text/control_expanded}" \
  --output-dir "${OUTPUT_DIR:-${ROOT}/data/ENX/control_text/control_expanded_normalized}" \
  --case "${CASE:-upper}" \
  --replace ' \$\$' ''

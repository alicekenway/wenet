#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lm_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${lm_root}/../.." && pwd)"
ROOT="${ROOT:-${repo_root}}"

LM_TEXT="${LM_TEXT:-${ROOT}/data/ENX/merge_b1/enx.txt}" \
OUT_DIR="${OUT_DIR:-${lm_root}/models}" \
REPORT_DIR="${REPORT_DIR:-${lm_root}/reports}" \
ORDER="${ORDER:-3}" \
PRUNE="${PRUNE:-0 0 1}" \
"${lm_root}/tools/train_kenlm_arpa.sh"

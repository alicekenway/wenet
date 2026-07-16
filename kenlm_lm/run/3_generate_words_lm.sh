#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lm_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${lm_root}/../.." && pwd)"
ROOT="${ROOT:-${repo_root}}"

"${lm_root}/tools/generate_words_from_lm.py" \
  --lm-text "${LM_TEXT:-${ROOT}/data/ENX/merge_b1/enx.txt}" \
  --output "${WORDS_OUT:-${lm_root}/data/words.txt}" \
  --report "${WORDS_REPORT:-${lm_root}/reports/words_report.json}"

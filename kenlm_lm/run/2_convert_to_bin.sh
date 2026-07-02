#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lm_root="$(cd "${script_dir}/.." && pwd)"

ARPA="${ARPA:-${lm_root}/models/wenetspeech_char_3gram.arpa}" \
BIN="${BIN:-${lm_root}/models/wenetspeech_char_3gram.bin}" \
REPORT_DIR="${REPORT_DIR:-${lm_root}/reports}" \
"${lm_root}/tools/build_kenlm_binary.sh"

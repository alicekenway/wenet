#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lm_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${lm_root}/../.." && pwd)"
ROOT="${ROOT:-${repo_root}}"

"${lm_root}/tools/generate_lexicon_for_am.py" \
  --words "${WORDS:-${lm_root}/data/words.txt}" \
  --tokens "${TOKENS:-${ROOT}/model/sherpa-onnx-en-wenet-gigaspeech_int8/tokens.txt}" \
  --output "${LEXICON_OUT:-${lm_root}/data/lexicon.txt}" \
  --report "${LEXICON_REPORT:-${lm_root}/reports/lexicon_report.json}" \
  --tokenization "${TOKENIZATION:-bpe}" \
  --allow-rejected \
  --ignore-case

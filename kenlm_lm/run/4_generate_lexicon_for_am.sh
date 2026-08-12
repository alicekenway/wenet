#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lm_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${lm_root}/../.." && pwd)"
ROOT="${ROOT:-${repo_root}}"
PYTHON="${PYTHON:-python3}"

command=("${PYTHON}" "${lm_root}/tools/generate_lexicon_for_am.py" \
  --words "${WORDS:-${lm_root}/data/words.txt}" \
  --tokens "${TOKENS:-${ROOT}/model/sherpa-onnx-en-wenet-gigaspeech_int8/tokens.txt}" \
  --output "${LEXICON_OUT:-${lm_root}/data/lexicon.txt}" \
  --report "${LEXICON_REPORT:-${lm_root}/reports/lexicon_report.json}" \
  --tokenization "${TOKENIZATION:-bpe}" \
  --bpe-max-spellings "${BPE_MAX_SPELLINGS:-3}" \
  --allow-rejected \
  --ignore-case)

if [[ -n "${SENTENCEPIECE_MODEL:-}" ]]; then
  command+=(--sentencepiece-model "${SENTENCEPIECE_MODEL}")
fi
if [[ "${BPE_ADD_CHARACTER_FALLBACK:-false}" == "true" ]]; then
  command+=(--bpe-add-character-fallback)
fi

"${command[@]}"

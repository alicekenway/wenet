#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lm_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${lm_root}/../.." && pwd)"

ROOT="${ROOT:-${repo_root}}"
SDK_ROOT="${SDK_ROOT:-${ROOT}/wenet/SDK/0.0.3}"
KENLM_BIN="${KENLM_BIN:-${SDK_ROOT}/third_party/kenlm-install/bin}"
LM_TEXT="${LM_TEXT:-${ROOT}/LM/wenet_lm/training/preprocess_data/wenetspeech_lm_char.txt}"
OUT_DIR="${OUT_DIR:-${lm_root}/models}"
REPORT_DIR="${REPORT_DIR:-${lm_root}/reports}"
ORDER="${ORDER:-4}"
PRUNE="${PRUNE:-0 0 1 1}"
ARPA="${ARPA:-${OUT_DIR}/wenetspeech_char_${ORDER}gram.arpa}"

mkdir -p "${OUT_DIR}" "${REPORT_DIR}"

if [[ ! -x "${KENLM_BIN}/lmplz" ]]; then
  echo "lmplz not found or not executable: ${KENLM_BIN}/lmplz" >&2
  exit 1
fi
if [[ ! -f "${LM_TEXT}" ]]; then
  echo "LM text not found: ${LM_TEXT}" >&2
  exit 1
fi

{
  echo "date: $(date -Iseconds)"
  echo "lmplz: ${KENLM_BIN}/lmplz"
  echo "lm_text: ${LM_TEXT}"
  echo "order: ${ORDER}"
  echo "prune: ${PRUNE}"
  echo "arpa: ${ARPA}"
  echo "command:"
  echo "  ${KENLM_BIN}/lmplz -o ${ORDER} --prune ${PRUNE} --text ${LM_TEXT} --arpa ${ARPA}"
} > "${REPORT_DIR}/train_kenlm_arpa.command.txt"

"${KENLM_BIN}/lmplz" \
  -o "${ORDER}" \
  --prune ${PRUNE} \
  --text "${LM_TEXT}" \
  --arpa "${ARPA}"

ls -lh "${ARPA}" | tee "${REPORT_DIR}/train_kenlm_arpa.size.txt"

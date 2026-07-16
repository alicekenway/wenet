#!/usr/bin/env bash
# Convert an ARPA language model to KenLM's binary format.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lm_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${lm_root}/../.." && pwd)"

ROOT="${ROOT:-${repo_root}}"
SDK_ROOT="${SDK_ROOT:-${ROOT}/wenet/SDK/0.0.6}"
KENLM_BIN="${KENLM_BIN:-${SDK_ROOT}/third_party/kenlm-install/bin}"
ARPA="${ARPA:-${lm_root}/models/wenetspeech_char_4gram.arpa}"
BIN="${BIN:-${lm_root}/models/lm.bin}"
REPORT_DIR="${REPORT_DIR:-${lm_root}/reports}"

mkdir -p "$(dirname "${BIN}")" "${REPORT_DIR}"

if [[ ! -x "${KENLM_BIN}/build_binary" ]]; then
  echo "build_binary not found or not executable: ${KENLM_BIN}/build_binary" >&2
  exit 1
fi
if [[ ! -f "${ARPA}" ]]; then
  echo "ARPA not found: ${ARPA}" >&2
  exit 1
fi

{
  echo "date: $(date -Iseconds)"
  echo "build_binary: ${KENLM_BIN}/build_binary"
  echo "arpa: ${ARPA}"
  echo "bin: ${BIN}"
  echo "command:"
  echo "  ${KENLM_BIN}/build_binary ${ARPA} ${BIN}"
} > "${REPORT_DIR}/convert_kenlm_binary.command.txt"

"${KENLM_BIN}/build_binary" "${ARPA}" "${BIN}"
ls -lh "${BIN}" | tee "${REPORT_DIR}/convert_kenlm_binary.size.txt"

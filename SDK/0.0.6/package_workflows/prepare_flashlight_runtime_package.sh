#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jinyang_wang/Dev/ASR/ASR_wenet
AM_DIR="${AM_DIR:-${ROOT}/model/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30}"
LM_DIR="${LM_DIR:-${ROOT}/LM/kenlm_lm}"
OUT_DIR="${OUT_DIR:-${ROOT}/test/0.0.6/model_flashlight}"
MAIN_LM_BIN="${MAIN_LM_BIN:-${LM_DIR}/models/lm.bin}"
# Set CONTACT_LM_BIN and CONTACT_LM_METADATA to enable pattern-bias runtime
# contact decoding.  The metadata is generated beside the handmade ARPA.
CONTACT_LM_BIN="${CONTACT_LM_BIN:-}"
CONTACT_LM_METADATA="${CONTACT_LM_METADATA:-}"
CONTACT_LM_MODE="${CONTACT_LM_MODE:-pattern_bias_v1}"
# These are the main-LM dictionary and static lexicon. Pattern words must be
# representable there; when contact mode is enabled the staged words.txt also
# receives the virtual <CONTACT> label, but no static lexicon pronunciation.
WORDS_FILE="${WORDS_FILE:-${LM_DIR}/data/words.txt}"
LEXICON_FILE="${LEXICON_FILE:-${LM_DIR}/data/lexicon.txt}"
MAPPING="${MAPPING:-}"
BEAM_SIZE="${BEAM_SIZE:-50}"
BEAM_SIZE_TOKEN="${BEAM_SIZE_TOKEN:-20}"
BEAM_THRESHOLD="${BEAM_THRESHOLD:-25}"
LM_WEIGHT="${LM_WEIGHT:-1.5}"
WORD_SCORE="${WORD_SCORE:--0.5}"
UNK_SCORE="${UNK_SCORE:--5.0}"
SIL_SCORE="${SIL_SCORE:-0.0}"
LOG_ADD="${LOG_ADD:-false}"
ALLOW_UNK="${ALLOW_UNK:-true}"
SMEARING="${SMEARING:-max}"
NBEST="${NBEST:-1}"
FEATURE_TYPE="${FEATURE_TYPE:-whisper}"
BLANK_TOKEN="${BLANK_TOKEN:-<blk>}"
SIL_TOKEN="${SIL_TOKEN:-▁}"
UNK_WORD="${UNK_WORD:-<unk>}"
CONTACT_CLASS_WORD="${CONTACT_CLASS_WORD:-<CONTACT>}"
CONTACT_LM_WEIGHT="${CONTACT_LM_WEIGHT:-1.5}"
CONTACT_LM_ACCUMULATION_FACTOR="${CONTACT_LM_ACCUMULATION_FACTOR:-0.5}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"

case "${FEATURE_TYPE}" in
  kaldi|whisper) ;;
  *)
    echo "FEATURE_TYPE must be 'kaldi' or 'whisper', got: ${FEATURE_TYPE}" >&2
    exit 1
    ;;
esac

case "${SAMPLE_RATE}" in
  ''|*[!0-9]*)
    echo "SAMPLE_RATE must be a positive integer, got: ${SAMPLE_RATE}" >&2
    exit 1
    ;;
esac

mkdir -p "${OUT_DIR}"

require_file() {
  local file="$1"
  if [[ ! -e "${file}" && -L "${file}" ]]; then
    echo "required file is a broken symlink: ${file} -> $(readlink "${file}")" >&2
    exit 1
  fi
  if [[ ! -f "${file}" ]]; then
    echo "required file missing: ${file}" >&2
    exit 1
  fi
}

copy_runtime_file() {
  local src="$1"
  local dst="$2"
  if [[ -e "${dst}" || -L "${dst}" ]]; then
    rm -f -- "${dst}"
  fi
  cp -aL "${src}" "${dst}"
}

is_positive_finite_number() {
  local value="$1"
  # Keep the emitted manifest valid JSON as well as requiring a positive,
  # finite numeric value.  In particular, reject +1, .5, 1., and 01.
  if [[ ! "${value}" =~ ^(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
    return 1
  fi
  awk -v value="${value}" 'BEGIN { exit !(value == value && value > 0) }'
}

is_unit_interval_number() {
  local value="$1"
  if [[ ! "${value}" =~ ^(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
    return 1
  fi
  awk -v value="${value}" 'BEGIN { exit !(value == value && value >= 0 && value <= 1) }'
}

has_first_column_word() {
  local file="$1"
  local word="$2"
  awk -v word="${word}" '$1 == word { found = 1 } END { exit !found }' \
    "${file}"
}

append_word_if_missing() {
  local file="$1"
  local word="$2"
  if has_first_column_word "${file}" "${word}"; then
    return
  fi
  local next_id
  next_id="$(awk 'BEGIN { max = -1 } $2 ~ /^[0-9]+$/ && $2 > max { max = $2 } END { print max + 1 }' "${file}")"
  printf '%s %s\n' "${word}" "${next_id}" >> "${file}"
}

for file in "${AM_DIR}/model.onnx" "${AM_DIR}/tokens.txt" \
            "${MAIN_LM_BIN}" "${WORDS_FILE}" "${LEXICON_FILE}"; do
  require_file "${file}"
done
if [[ -n "${CONTACT_LM_BIN}" ]]; then
  require_file "${CONTACT_LM_BIN}"
  require_file "${CONTACT_LM_METADATA}"
  if [[ "${CONTACT_LM_MODE}" != "pattern_bias_v1" ]]; then
    echo "CONTACT_LM_MODE must be pattern_bias_v1, got: ${CONTACT_LM_MODE}" >&2
    exit 1
  fi
  if ! is_positive_finite_number "${CONTACT_LM_WEIGHT}"; then
    echo "CONTACT_LM_WEIGHT must be a finite number > 0, got: ${CONTACT_LM_WEIGHT}" >&2
    exit 1
  fi
  if ! is_unit_interval_number "${CONTACT_LM_ACCUMULATION_FACTOR}"; then
    echo "CONTACT_LM_ACCUMULATION_FACTOR must be finite and between 0 and 1, got: ${CONTACT_LM_ACCUMULATION_FACTOR}" >&2
    exit 1
  fi
fi

copy_runtime_file "${AM_DIR}/model.onnx" "${OUT_DIR}/model.onnx"
copy_runtime_file "${AM_DIR}/tokens.txt" "${OUT_DIR}/tokens.txt"
copy_runtime_file "${MAIN_LM_BIN}" "${OUT_DIR}/lm.bin"
copy_runtime_file "${WORDS_FILE}" "${OUT_DIR}/words.txt"
copy_runtime_file "${LEXICON_FILE}" "${OUT_DIR}/lexicon.txt"

if [[ -n "${MAPPING}" ]]; then
  require_file "${MAPPING}"
  copy_runtime_file "${MAPPING}" "${OUT_DIR}/output_mapping.txt"
else
  : > "${OUT_DIR}/output_mapping.txt"
fi

contact_enabled=false
if [[ -n "${CONTACT_LM_BIN}" ]]; then
  contact_enabled=true
  if [[ "${SMEARING,,}" != "max" ]]; then
    echo "runtime contact packages require SMEARING=max" >&2
    exit 1
  fi
  # words.txt is shared by both decoders.  Add the class label only to this
  # staged copy; it must not be added to the main source dictionary, lexicon,
  # or AM token table.
  append_word_if_missing "${OUT_DIR}/words.txt" "${CONTACT_CLASS_WORD}"
  if has_first_column_word "${OUT_DIR}/tokens.txt" "${CONTACT_CLASS_WORD}"; then
    echo "CONTACT_CLASS_WORD must not be present in tokens.txt: ${CONTACT_CLASS_WORD}" >&2
    exit 1
  fi
  if has_first_column_word "${OUT_DIR}/lexicon.txt" "${CONTACT_CLASS_WORD}"; then
    echo "CONTACT_CLASS_WORD must not have a static lexicon pronunciation: ${CONTACT_CLASS_WORD}" >&2
    exit 1
  fi
  copy_runtime_file "${CONTACT_LM_BIN}" "${OUT_DIR}/contact_lm.bin"
  copy_runtime_file "${CONTACT_LM_METADATA}" "${OUT_DIR}/contact_lm.meta.json"
else
  # Do not leave a stale contact LM behind when reusing an output directory.
  rm -f -- "${OUT_DIR}/contact_lm.bin"
  rm -f -- "${OUT_DIR}/contact_lm.meta.json"
fi

{
cat <<JSON
{
  "decoder_type": "flashlight_lexicon_kenlm",
  "model_path": "model.onnx",
  "tokens": "tokens.txt",
  "words": "words.txt",
  "lexicon": "lexicon.txt",
  "lm": "lm.bin",
JSON
if [[ "${contact_enabled}" == true ]]; then
cat <<JSON
  "contact_lm": "contact_lm.bin",
  "contact_lm_weight": ${CONTACT_LM_WEIGHT},
  "contact_lm_mode": "${CONTACT_LM_MODE}",
  "contact_lm_metadata": "contact_lm.meta.json",
  "contact_lm_accumulation_factor": ${CONTACT_LM_ACCUMULATION_FACTOR},
  "contact_class_word": "${CONTACT_CLASS_WORD}",
JSON
fi
cat <<JSON
  "mapping": "output_mapping.txt",
  "feature_type": "${FEATURE_TYPE}",
  "blank_token": "${BLANK_TOKEN}",
  "sil_token": "${SIL_TOKEN}",
  "unk_word": "${UNK_WORD}",
  "sample_rate": ${SAMPLE_RATE},
  "beam_size": ${BEAM_SIZE},
  "beam_size_token": ${BEAM_SIZE_TOKEN},
  "beam_threshold": ${BEAM_THRESHOLD},
  "lm_weight": ${LM_WEIGHT},
  "word_score": ${WORD_SCORE},
  "unk_score": ${UNK_SCORE},
  "sil_score": ${SIL_SCORE},
  "log_add": ${LOG_ADD},
  "allow_unk": ${ALLOW_UNK},
  "smearing": "${SMEARING}",
  "nbest": ${NBEST}
}
JSON
} > "${OUT_DIR}/sdk_model.json"

checksum_files=(
  "${OUT_DIR}/model.onnx"
  "${OUT_DIR}/tokens.txt"
  "${OUT_DIR}/words.txt"
  "${OUT_DIR}/lexicon.txt"
  "${OUT_DIR}/lm.bin"
  "${OUT_DIR}/output_mapping.txt"
)
if [[ "${contact_enabled}" == true ]]; then
  checksum_files+=("${OUT_DIR}/contact_lm.bin" "${OUT_DIR}/contact_lm.meta.json")
fi
sha256sum "${checksum_files[@]}" > "${OUT_DIR}/checksums.sha256"

echo "prepared package: ${OUT_DIR}"
if [[ "${contact_enabled}" == true ]]; then
  echo "runtime contacts: enabled (contact_lm_weight=${CONTACT_LM_WEIGHT}, accumulation_factor=${CONTACT_LM_ACCUMULATION_FACTOR})"
else
  echo "runtime contacts: not configured"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jinyang_wang/Dev/ASR/ASR_wenet
AM_DIR="${AM_DIR:-${ROOT}/model/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30}"
LM_DIR="${LM_DIR:-${ROOT}/LM/kenlm_lm}"
OUT_DIR="${OUT_DIR:-${ROOT}/test/0.0.3/model_flashlight}"
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

for file in "${AM_DIR}/model.onnx" "${AM_DIR}/tokens.txt" \
            "${LM_DIR}/models/lm.bin" "${LM_DIR}/data/words.txt" \
            "${LM_DIR}/data/lexicon.txt"; do
  require_file "${file}"
done

copy_runtime_file "${AM_DIR}/model.onnx" "${OUT_DIR}/model.onnx"
copy_runtime_file "${AM_DIR}/tokens.txt" "${OUT_DIR}/tokens.txt"
copy_runtime_file "${LM_DIR}/models/lm.bin" "${OUT_DIR}/lm.bin"
copy_runtime_file "${LM_DIR}/data/words.txt" "${OUT_DIR}/words.txt"
copy_runtime_file "${LM_DIR}/data/lexicon.txt" "${OUT_DIR}/lexicon.txt"

if [[ -n "${MAPPING}" ]]; then
  require_file "${MAPPING}"
  copy_runtime_file "${MAPPING}" "${OUT_DIR}/output_mapping.txt"
else
  : > "${OUT_DIR}/output_mapping.txt"
fi

cat > "${OUT_DIR}/sdk_model.json" <<JSON
{
  "decoder_type": "flashlight_lexicon_kenlm",
  "model_path": "model.onnx",
  "tokens": "tokens.txt",
  "words": "words.txt",
  "lexicon": "lexicon.txt",
  "lm": "lm.bin",
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

sha256sum "${OUT_DIR}/model.onnx" "${OUT_DIR}/tokens.txt" \
  "${OUT_DIR}/words.txt" "${OUT_DIR}/lexicon.txt" "${OUT_DIR}/lm.bin" \
  "${OUT_DIR}/output_mapping.txt" > "${OUT_DIR}/checksums.sha256"

echo "prepared package: ${OUT_DIR}"

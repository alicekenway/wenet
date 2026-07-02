#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lm_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${lm_root}/../.." && pwd)"
ROOT="${ROOT:-${repo_root}}"

LM_WEIGHT="${LM_WEIGHT:-0.5}" WORD_SCORE="${WORD_SCORE:-0.0}" \
BEAM_SIZE="${BEAM_SIZE:-50}" BEAM_SIZE_TOKEN="${BEAM_SIZE_TOKEN:-15}" \
FEATURE_TYPE="${FEATURE_TYPE:-kaldi}" BLANK_TOKEN="${BLANK_TOKEN:-<blank>}" \
AM_DIR="${AM_DIR:-${ROOT}/model/sherpa-onnx-en-wenet-gigaspeech_int8}" \
LM_DIR="${LM_DIR:-${lm_root}}" \
OUT_DIR="${OUT_DIR:-${ROOT}/test/0.0.3/sherpa-onnx-en-wenet-gigaspeech_int8/package}" \
"${SDK_ROOT:-${ROOT}/wenet/SDK/0.0.3}/scripts/prepare_flashlight_runtime_package.sh"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SDK_BUILD_DIR MODEL_DIR WAV_PATH OUT_DIR" >&2
  exit 2
fi

sdk_build_dir=$1
model_dir=$2
wav_path=$3
out_dir=$4

mkdir -p "${out_dir}"

"${sdk_build_dir}/print_build_info" | tee "${out_dir}/build_info.json"
"${sdk_build_dir}/inspect_package" --model_dir "${model_dir}" \
  | tee "${out_dir}/inspect_package.txt"
"${sdk_build_dir}/asr_stream_file" \
  --model_dir "${model_dir}" \
  --wav "${wav_path}" \
  --chunk_ms 100 \
  --print_partial true \
  | tee "${out_dir}/decode_wfst.txt"

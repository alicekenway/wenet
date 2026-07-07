#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SDK_TOOLS_DIR MODEL_DIR WAV_PATH OUT_DIR" >&2
  exit 2
fi

sdk_tools_dir=$1
model_dir=$2
wav_path=$3
out_dir=$4

mkdir -p "${out_dir}"

"${sdk_tools_dir}/print_sdk_info" | tee "${out_dir}/sdk_info.json"
"${sdk_tools_dir}/inspect_package" --model_dir "${model_dir}" \
  | tee "${out_dir}/inspect_package.txt"
"${sdk_tools_dir}/asr_stream_file" \
  --model_dir "${model_dir}" \
  --wav "${wav_path}" \
  --chunk_ms 100 \
  --print_partial true \
  | tee "${out_dir}/decode_wfst.txt"

#!/usr/bin/env bash
set -euo pipefail

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(cd "${sdk_dir}/../../.." && pwd)

build_dir=${1:-"${sdk_dir}/build"}
model_dir=${2:-"${repo_root}/model/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30"}
data_dir=${3:-"${repo_root}/data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000"}
out_dir=${4:-"${repo_root}/test/0.0.2/zipformer_smoke"}

mkdir -p "${out_dir}"

graph_dir="${out_dir}/token_loop_graph"
"${sdk_dir}/scripts/generate_sherpa_token_loop_fst.py" \
  --tokens "${model_dir}/tokens.txt" \
  --out-dir "${graph_dir}"

run_one() {
  local wav=$1
  local name=$2
  local log="${out_dir}/${name}.log"
  "${build_dir}/zipformer_ctc_wfst_main" \
    --model "${model_dir}/model.onnx" \
    --tokens "${model_dir}/tokens.txt" \
    --fst "${graph_dir}/TLG.fst" \
    --words "${graph_dir}/words.txt" \
    --wav "${wav}" \
    --print-greedy true \
    --print-wfst true \
    > "${log}"
  grep -q '^greedy text: .\+' "${log}"
  grep -q '^WFST text: .\+' "${log}"
  if grep -Eq '[<#]|▁' "${log}"; then
    echo "unexpected special symbol in ${log}" >&2
    return 1
  fi
  cat "${log}"
}

run_one "${model_dir}/test_wavs/0.wav" official_0
run_one "${model_dir}/test_wavs/1.wav" official_1

if [[ -d "${data_dir}/wav" ]]; then
  mapfile -t dataset_wavs < <(find "${data_dir}/wav" -maxdepth 1 -name '*.wav' | sort)
  for wav in "${dataset_wavs[@]:0:3}"; do
    stem=$(basename "${wav}" .wav)
    run_one "${wav}" "dataset_${stem}"
  done
fi

echo "zipformer smoke test passed"

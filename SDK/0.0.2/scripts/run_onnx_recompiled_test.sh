#!/usr/bin/env bash
set -euo pipefail

sdk_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

onnx_dir="${1:-/home/jinyang_wang/Dev/ASR/ASR_wenet/model/wenet_efficient_conformer_aishell_v2_onnx_1}"
out_root="${2:-${sdk_root}/test_runs/onnx_recompiled}"

asr_root="${ASR_ROOT:-/home/jinyang_wang/Dev/ASR/ASR_wenet}"
build_dir="${BUILD_DIR:-${sdk_root}/build}"
metadata="${METADATA:-${asr_root}/data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000/metadata.jsonl}"
wav_root="${WAV_ROOT:-${asr_root}/data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000/wav}"
am_units="${AM_UNITS:-${asr_root}/model/wenet_efficient_conformer_aishell_v2/words.txt}"
tlg_dir="${TLG_DIR:-${asr_root}/LM/wenet_lm/tlg/lang_test}"
chunk_size="${CHUNK_SIZE:-16}"
num_left_chunks="${NUM_DECODING_LEFT_CHUNKS:-16}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
usage: $(basename "$0") [ONNX_DIR] [OUT_ROOT]

Defaults:
  ONNX_DIR=${onnx_dir}
  OUT_ROOT=${out_root}

Environment overrides:
  ASR_ROOT, BUILD_DIR, METADATA, WAV_ROOT, AM_UNITS, TLG_DIR,
  CHUNK_SIZE, NUM_DECODING_LEFT_CHUNKS
EOF
  exit 0
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
}

require_file "${onnx_dir}/encoder.onnx"
require_file "${onnx_dir}/ctc.onnx"
require_file "${onnx_dir}/decoder.onnx"
require_file "${build_dir}/asr_batch_decode"
require_file "${metadata}"
require_file "${am_units}"
require_file "${tlg_dir}/units.txt"
require_file "${tlg_dir}/words.txt"
require_file "${tlg_dir}/TLG.fst"

mkdir -p "${out_root}/data" "${out_root}/models" "${out_root}/results"

all_scp="${out_root}/data/all.wav.scp"
smoke_scp="${out_root}/data/smoke10.wav.scp"

python3 - "${metadata}" "${wav_root}" "${all_scp}" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
wav_root = Path(sys.argv[2])
out_path = Path(sys.argv[3])

with metadata.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key") or row.get("id") or row.get("utt") or row.get("utt_id")
        wav = (
            row.get("wav")
            or row.get("audio_filepath")
            or row.get("audiofile_path")
            or row.get("audio")
            or row.get("path")
        )
        if wav:
            wav_path = Path(wav)
            if not wav_path.is_absolute():
                wav_path = wav_root / wav_path.name
        elif key:
            wav_path = wav_root / f"{key}.wav"
        else:
            raise SystemExit(f"Cannot infer key/wav from row: {row}")
        if key is None:
            key = wav_path.stem
        fout.write(f"{key} {wav_path}\n")
PY

head -n 10 "${all_scp}" > "${smoke_scp}"

write_package() {
  local mode="$1"
  local model_dir="$2"
  mkdir -p "${model_dir}"
  ln -sfn "${onnx_dir}/encoder.onnx" "${model_dir}/encoder.onnx"
  ln -sfn "${onnx_dir}/ctc.onnx" "${model_dir}/ctc.onnx"
  ln -sfn "${onnx_dir}/decoder.onnx" "${model_dir}/decoder.onnx"

  if [[ "${mode}" == "wfst" ]]; then
    ln -sfn "${tlg_dir}/units.txt" "${model_dir}/units.txt"
    ln -sfn "${tlg_dir}/words.txt" "${model_dir}/words.txt"
    ln -sfn "${tlg_dir}/TLG.fst" "${model_dir}/TLG.fst"
    cat > "${model_dir}/sdk_model.json" <<JSON
{
  "sdk_model_version": 1,
  "backend": "wenet_onnxruntime_static_wenet_dynamic_ort",
  "audio": {"sample_rate": 16000},
  "wenet": {
    "onnx_dir": ".",
    "unit_path": "units.txt",
    "dict_path": "words.txt",
    "fst_path": "TLG.fst"
  },
  "decode": {
    "chunk_size": ${chunk_size},
    "num_left_chunks": ${num_left_chunks},
    "nbest": 1
  },
  "runtime": {"enable_continuous_decoding": true},
  "postprocess": {"language_type": "chs", "enable_timestamp": false}
}
JSON
  else
    ln -sfn "${am_units}" "${model_dir}/units.txt"
    rm -f "${model_dir}/words.txt" "${model_dir}/TLG.fst"
    cat > "${model_dir}/sdk_model.json" <<JSON
{
  "sdk_model_version": 1,
  "backend": "wenet_onnxruntime_static_wenet_dynamic_ort",
  "audio": {"sample_rate": 16000},
  "wenet": {
    "onnx_dir": ".",
    "unit_path": "units.txt"
  },
  "decode": {
    "chunk_size": ${chunk_size},
    "num_left_chunks": ${num_left_chunks},
    "nbest": 1
  },
  "runtime": {"enable_continuous_decoding": true},
  "postprocess": {"language_type": "chs", "enable_timestamp": false}
}
JSON
  fi
}

count_non_empty() {
  local result="$1"
  python3 - "$result" <<'PY'
import sys
from pathlib import Path

rows = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
non_empty = 0
for row in rows:
    parts = row.split(maxsplit=1)
    if len(parts) > 1 and parts[1].strip():
        non_empty += 1
print(non_empty)
PY
}

run_decode() {
  local name="$1"
  local model_dir="$2"
  local wav_scp="$3"
  local result="$4"
  local log="$5"
  echo "Running ${name}: ${wav_scp}"
  "${build_dir}/asr_batch_decode" \
    --model_dir "${model_dir}" \
    --wav_scp "${wav_scp}" \
    --result "${result}" \
    2>&1 | tee "${log}"
}

no_lm_model="${out_root}/models/no_lm_greedy"
wfst_model="${out_root}/models/wfst_beam"
write_package "no_lm" "${no_lm_model}"
write_package "wfst" "${wfst_model}"

echo "ONNX dir: ${onnx_dir}"
echo "Output root: ${out_root}"
echo "All wav.scp: ${all_scp}"
echo "Smoke wav.scp: ${smoke_scp}"
echo "No-LM model: ${no_lm_model}"
echo "WFST model: ${wfst_model}"

"${build_dir}/inspect_package" --model_dir "${no_lm_model}" > "${out_root}/results/inspect_no_lm.txt"
"${build_dir}/inspect_package" --model_dir "${wfst_model}" > "${out_root}/results/inspect_wfst.txt"

no_lm_smoke="${out_root}/results/no_lm_greedy_smoke10.txt"
wfst_smoke="${out_root}/results/wfst_beam_smoke10.txt"
run_decode "no_lm_greedy_smoke10" "${no_lm_model}" "${smoke_scp}" "${no_lm_smoke}" "${out_root}/results/no_lm_greedy_smoke10.log"
run_decode "wfst_beam_smoke10" "${wfst_model}" "${smoke_scp}" "${wfst_smoke}" "${out_root}/results/wfst_beam_smoke10.log"

no_lm_smoke_non_empty="$(count_non_empty "${no_lm_smoke}")"
wfst_smoke_non_empty="$(count_non_empty "${wfst_smoke}")"
echo "no_lm_greedy_smoke10_non_empty=${no_lm_smoke_non_empty}"
echo "wfst_beam_smoke10_non_empty=${wfst_smoke_non_empty}"

if (( no_lm_smoke_non_empty > 0 )); then
  run_decode "no_lm_greedy_all" "${no_lm_model}" "${all_scp}" \
    "${out_root}/results/no_lm_greedy_all.txt" \
    "${out_root}/results/no_lm_greedy_all.log"
  echo "no_lm_greedy_all_non_empty=$(count_non_empty "${out_root}/results/no_lm_greedy_all.txt")"
else
  echo "Skipping no_lm_greedy_all because smoke test was empty"
fi

if (( wfst_smoke_non_empty > 0 )); then
  run_decode "wfst_beam_all" "${wfst_model}" "${all_scp}" \
    "${out_root}/results/wfst_beam_all.txt" \
    "${out_root}/results/wfst_beam_all.log"
  echo "wfst_beam_all_non_empty=$(count_non_empty "${out_root}/results/wfst_beam_all.txt")"
else
  echo "Skipping wfst_beam_all because smoke test was empty"
fi

echo "Done. Results are under ${out_root}/results"

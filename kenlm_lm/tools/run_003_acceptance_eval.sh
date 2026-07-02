#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lm_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${lm_root}/../.." && pwd)"

ROOT="${ROOT:-${repo_root}}"
SDK_ROOT="${SDK_ROOT:-${ROOT}/wenet/SDK/0.0.3}"
BUILD_DIR="${SDK_ROOT}/build"
PACKAGE_DIR="${PACKAGE_DIR:-${ROOT}/test/0.0.3/model_flashlight}"
METADATA="${METADATA:-${ROOT}/test/0.0.3/metadata.sample100.jsonl}"
OUT_DIR="${OUT_DIR:-${ROOT}/test/0.0.3/acceptance}"
WAV_ROOT="${WAV_ROOT:-${ROOT}/data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000}"
LIMIT="${LIMIT:-100}"
BEAM_SIZE="${BEAM_SIZE:-50}"
BEAM_SIZE_TOKEN="${BEAM_SIZE_TOKEN:-20}"
BEAM_THRESHOLD="${BEAM_THRESHOLD:-25}"
LM_WEIGHT="${LM_WEIGHT:-1.5}"
WORD_SCORE="${WORD_SCORE:--0.5}"
UNK_SCORE="${UNK_SCORE:--5.0}"
SIL_SCORE="${SIL_SCORE:-0.0}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${BUILD_DIR}/zipformer_ctc_flashlight_main" ]]; then
  echo "decoder not built: ${BUILD_DIR}/zipformer_ctc_flashlight_main" >&2
  exit 1
fi
if [[ ! -f "${METADATA}" ]]; then
  echo "sample metadata not found: ${METADATA}" >&2
  exit 1
fi

python3 - "$METADATA" "$OUT_DIR/wav_refs.tsv" "$LIMIT" "$WAV_ROOT" <<'PY'
import json
import sys
from pathlib import Path

metadata, out, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
base = Path(sys.argv[4])
rows = []
for line in metadata.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    ref = item.get("text") or item.get("sentence") or item.get("transcript")
    if ref is None:
        raise SystemExit(f"cannot find ref text in metadata row: {item}")
    wav = (
        item.get("audio_filepath")
        or item.get("audiofile_path")
        or item.get("wav")
        or item.get("path")
    )
    if wav is None and "file_name" in item:
        wav = str(base / item["file_name"])
    if wav is None:
        wav = item.get("audio", {}).get("path")
    if wav is None:
        raise SystemExit(f"cannot find wav path in metadata row: {item}")
    wav_path = Path(wav)
    if not wav_path.is_absolute():
        wav_path = base / wav_path
    rows.append((str(wav_path), "".join(str(ref).split())))
rows = rows[:limit]
out.write_text(
    "".join(f"{i}\t{wav}\t{ref}\n" for i, (wav, ref) in enumerate(rows)),
    encoding="utf-8",
)
print(f"wrote {len(rows)} wav/ref rows to {out}")
PY

: > "${OUT_DIR}/transcripts.txt"
: > "${OUT_DIR}/decode.log"

while IFS=$'\t' read -r utt wav ref; do
  log="${OUT_DIR}/${utt}.log"
  "${BUILD_DIR}/zipformer_ctc_flashlight_main" \
    --model "${PACKAGE_DIR}/model.onnx" \
    --tokens "${PACKAGE_DIR}/tokens.txt" \
    --words "${PACKAGE_DIR}/words.txt" \
    --lexicon "${PACKAGE_DIR}/lexicon.txt" \
    --lm "${PACKAGE_DIR}/lm.bin" \
    --mapping "${PACKAGE_DIR}/output_mapping.txt" \
    --wav "${wav}" \
    --nbest 1 \
    --beam_size "${BEAM_SIZE}" \
    --beam_size_token "${BEAM_SIZE_TOKEN}" \
    --beam_threshold "${BEAM_THRESHOLD}" \
    --lm_weight "${LM_WEIGHT}" \
    --word_score "${WORD_SCORE}" \
    --unk_score "${UNK_SCORE}" \
    --sil_score "${SIL_SCORE}" \
    > "${log}" 2>&1 || {
      echo "${utt} failed; see ${log}" | tee -a "${OUT_DIR}/decode.log"
      exit 1
    }
  greedy=$(grep -a -m1 '^greedy text:' "${log}" | sed 's/^greedy text: //')
  mapped=$(grep -a -m1 '^hyp 0 mapped text:' "${log}" | sed 's/^hyp 0 mapped text: //')
  raw=$(grep -a -m1 '^hyp 0 raw words:' "${log}" | sed 's/^hyp 0 raw words: //')
  search_rtf=$(grep -a -m1 '^search RTF:' "${log}" | awk '{print $3}')
  echo -e "${utt}\t${ref}\t${greedy}\t${mapped}\t${raw}\t${search_rtf}" >> "${OUT_DIR}/transcripts.txt"
done < "${OUT_DIR}/wav_refs.tsv"

python3 "${lm_root}/tools/score_acceptance_results.py" \
  --metadata "${METADATA}" \
  --wav-root "${WAV_ROOT}" \
  --log-dir "${OUT_DIR}" \
  --output "${OUT_DIR}/eval.tsv" \
  --summary "${OUT_DIR}/summary.txt" \
  --limit "${LIMIT}"

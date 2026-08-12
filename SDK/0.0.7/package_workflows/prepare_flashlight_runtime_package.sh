#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jinyang_wang/Dev/ASR/ASR_wenet
OUT_DIR="${OUT_DIR:-${ROOT}/test/0.0.7/package}"
AM_MODEL="${AM_MODEL:-${ROOT}/model/sdk006_export/model.int8.onnx}"
TOKENS_FILE="${TOKENS_FILE:-${ROOT}/model/sdk006_export/tokens.txt}"
LM_SEARCH_JSON="${LM_SEARCH_JSON:?set LM_SEARCH_JSON to the multi-LM JSON file}"
LM_BIN_DIR="${LM_BIN_DIR:-$(dirname "${LM_SEARCH_JSON}")}"
WORDS_FILE="${WORDS_FILE:?set WORDS_FILE to the shared words.txt}"
LEXICON_FILE="${LEXICON_FILE:?set LEXICON_FILE to the static lexicon.txt}"
MAPPING="${MAPPING:-}"
FINAL_MAPPING="${FINAL_MAPPING:-}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.0}"
BEAM_SIZE="${BEAM_SIZE:-50}"
BEAM_SIZE_TOKEN="${BEAM_SIZE_TOKEN:-20}"
BEAM_THRESHOLD="${BEAM_THRESHOLD:-25}"
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

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file missing: $1" >&2
    exit 1
  fi
}

copy_runtime_file() {
  local src="$1" dst="$2"
  if [[ -e "${dst}" || -L "${dst}" ]]; then rm -f -- "${dst}"; fi
  cp -aL -- "${src}" "${dst}"
}

append_word_if_missing() {
  local file="$1" word="$2"
  if awk -v word="${word}" '$1 == word { found=1 } END { exit !found }' "${file}"; then
    return
  fi
  local next_id
  next_id="$(awk 'BEGIN {m=-1} $2 ~ /^[0-9]+$/ && $2>m {m=$2} END {print m+1}' "${file}")"
  printf '%s %s\n' "${word}" "${next_id}" >> "${file}"
}

for file in "${AM_MODEL}" "${TOKENS_FILE}" "${LM_SEARCH_JSON}" \
            "${WORDS_FILE}" "${LEXICON_FILE}"; do
  require_file "${file}"
done
case "${FEATURE_TYPE}" in kaldi|whisper) ;; *) echo "invalid FEATURE_TYPE" >&2; exit 1;; esac
if [[ "${SMEARING,,}" != max ]]; then
  echo "0.0.7 multi-slot packages require SMEARING=max" >&2
  exit 1
fi

# Use a full validator that also emits deterministic filename and slot records.
mapfile -t records < <(python3 - "${LM_SEARCH_JSON}" <<'PY'
import json, pathlib, sys
def no_dupes(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError(f"duplicate key: {k}")
        out[k]=v
    return out
p=pathlib.Path(sys.argv[1])
obj=json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=no_dupes)
if not isinstance(obj,dict) or not obj: raise SystemExit("lm_search must be a non-empty object")
for name,cfg in sorted(obj.items()):
    q=pathlib.Path(name)
    if q.name != name or q.suffix != ".bin": raise SystemExit(f"invalid LM basename: {name}")
    if not isinstance(cfg,dict) or cfg.get("type") not in ("ngram","bias"):
        raise SystemExit(f"invalid LM config: {name}")
    print("LM\t"+name)
    if cfg["type"]=="bias":
        for slot in cfg.get("slots",[]): print("SLOT\t"+slot)
PY
)

mkdir -p "${OUT_DIR}"
copy_runtime_file "${AM_MODEL}" "${OUT_DIR}/model.onnx"
copy_runtime_file "${TOKENS_FILE}" "${OUT_DIR}/tokens.txt"
copy_runtime_file "${WORDS_FILE}" "${OUT_DIR}/words.txt"
copy_runtime_file "${LEXICON_FILE}" "${OUT_DIR}/lexicon.txt"
copy_runtime_file "${LM_SEARCH_JSON}" "${OUT_DIR}/lm_search.json"
if [[ -n "${MAPPING}" ]]; then
  require_file "${MAPPING}"
  copy_runtime_file "${MAPPING}" "${OUT_DIR}/output_mapping.txt"
else
  : > "${OUT_DIR}/output_mapping.txt"
fi
if [[ -n "${FINAL_MAPPING}" ]]; then
  require_file "${FINAL_MAPPING}"
  copy_runtime_file "${FINAL_MAPPING}" "${OUT_DIR}/final_output_mapping.txt"
else
  rm -f -- "${OUT_DIR}/final_output_mapping.txt"
fi

lm_outputs=()
for record in "${records[@]}"; do
  kind="${record%%$'\t'*}"
  value="${record#*$'\t'}"
  if [[ "${kind}" == LM ]]; then
    require_file "${LM_BIN_DIR}/${value}"
    copy_runtime_file "${LM_BIN_DIR}/${value}" "${OUT_DIR}/${value}"
    lm_outputs+=("${value}")
  elif [[ "${kind}" == SLOT ]]; then
    append_word_if_missing "${OUT_DIR}/words.txt" "${value}"
    if awk -v word="${value}" '$1 == word {found=1} END {exit !found}' "${OUT_DIR}/tokens.txt" ||
       awk -v word="${value}" '$1 == word {found=1} END {exit !found}' "${OUT_DIR}/lexicon.txt"; then
      echo "slot token must not occur in tokens.txt or lexicon.txt: ${value}" >&2
      exit 1
    fi
  fi
done

python3 - "${OUT_DIR}/sdk_model.json" "${LENGTH_PENALTY}" <<PY
import json,sys
p=sys.argv[1]
manifest={
  "decoder_type":"flashlight_lexicon_kenlm", "model_path":"model.onnx",
  "tokens":"tokens.txt", "words":"words.txt", "lexicon":"lexicon.txt",
  "lm_search":"lm_search.json", "mapping":"output_mapping.txt",
  "feature_type":"${FEATURE_TYPE}", "blank_token":"${BLANK_TOKEN}",
  "sil_token":"${SIL_TOKEN}", "unk_word":"${UNK_WORD}",
  "sample_rate":${SAMPLE_RATE}, "beam_size":${BEAM_SIZE},
  "beam_size_token":${BEAM_SIZE_TOKEN}, "beam_threshold":${BEAM_THRESHOLD},
  "length_penalty":float(sys.argv[2]), "unk_score":${UNK_SCORE},
  "sil_score":${SIL_SCORE}, "log_add":"${LOG_ADD}".lower()=="true",
  "allow_unk":"${ALLOW_UNK}".lower()=="true",
  "smearing":"${SMEARING}", "nbest":${NBEST}
}
if "${FINAL_MAPPING}":
  manifest["final_mapping"]="final_output_mapping.txt"
open(p,"w",encoding="utf-8").write(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
PY

checksum_files=("sdk_model.json" "model.onnx" "tokens.txt" "words.txt"
  "lexicon.txt" "lm_search.json" "output_mapping.txt" "${lm_outputs[@]}")
if [[ -n "${FINAL_MAPPING}" ]]; then
  checksum_files+=("final_output_mapping.txt")
fi
(
  cd "${OUT_DIR}"
  sha256sum "${checksum_files[@]}" > checksums.sha256
)
echo "prepared 0.0.7 multi-LM package: ${OUT_DIR}"

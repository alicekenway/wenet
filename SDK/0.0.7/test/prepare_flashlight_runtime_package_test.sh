#!/usr/bin/env bash
set -euo pipefail
sdk_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/asr-sdk-007-package.XXXXXX")"
trap 'rm -rf -- "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/input"
touch "${tmp_dir}/input/model.onnx" "${tmp_dir}/input/tokens.txt" \
      "${tmp_dir}/input/one.bin" "${tmp_dir}/input/two.bin" \
      "${tmp_dir}/input/bias.bin"
printf '<unk> 0\nhello 1\n' > "${tmp_dir}/input/words.txt"
printf 'hello H\n' > "${tmp_dir}/input/lexicon.txt"
cat > "${tmp_dir}/input/lm_search.json" <<'JSON'
{
  "one.bin":{"type":"ngram","weight":0.5,"clip":true,
             "clip_lower":0,"clip_upper":5},
  "two.bin":{"type":"ngram","weight":0.8,"clip":false},
  "bias.bin":{"type":"bias","weight":1.5,
              "contact_lm_accumulation_factor":0.5,
              "slots":["<CONTACT>","<APP>"]}
}
JSON
OUT_DIR="${tmp_dir}/out" AM_MODEL="${tmp_dir}/input/model.onnx" \
TOKENS_FILE="${tmp_dir}/input/tokens.txt" \
LM_SEARCH_JSON="${tmp_dir}/input/lm_search.json" \
LM_BIN_DIR="${tmp_dir}/input" WORDS_FILE="${tmp_dir}/input/words.txt" \
LEXICON_FILE="${tmp_dir}/input/lexicon.txt" \
bash "${sdk_root}/package_workflows/prepare_flashlight_runtime_package.sh"
python3 - "${tmp_dir}/out" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
m=json.loads((p/'sdk_model.json').read_text())
assert m['lm_search']=='lm_search.json' and m['length_penalty']==0
assert not any(k in m for k in ('lm','lm_weight','contact_lm','contact_lm_metadata','word_score'))
for name in ('one.bin','two.bin','bias.bin','lm_search.json'): assert (p/name).is_file()
words=(p/'words.txt').read_text()
assert '<CONTACT>' in words and '<APP>' in words
checks=(p/'checksums.sha256').read_text()
assert 'sdk_model.json' in checks and 'bias.bin' in checks
assert 'lm_search.json' in checks and 'meta.json' not in checks
PY

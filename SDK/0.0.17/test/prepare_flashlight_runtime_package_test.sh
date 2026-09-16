#!/usr/bin/env bash
set -euo pipefail
sdk_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/asr-sdk-009-package.XXXXXX")"
trap 'rm -rf -- "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/input"
touch "${tmp_dir}/input/model.onnx"
touch "${tmp_dir}/input/sentencepiece.model"
touch "${tmp_dir}/input/en_itn_tagger.fst" \
      "${tmp_dir}/input/en_itn_verbalizer.fst"
printf '<blk> 0\n▁ 1\nH 2\n#0 3\n' > "${tmp_dir}/input/tokens.txt"
printf '<unk> 0\nhello 1\nworld 2\n<CONTACT> 3\n<APP> 4\n' > "${tmp_dir}/input/words.txt"
printf 'hello ▁ H\n' > "${tmp_dir}/input/lexicon.txt"
printf 'hello world -> world hello\n' > "${tmp_dir}/input/mapping.txt"
for lm in one.bin two.bin bias.bin; do
cat > "${tmp_dir}/input/${lm}" <<'ARPA'
\data\
ngram 1=7
ngram 2=1

\1-grams:
-1	<unk>	0
-1	<s>	0
-1	</s>	0
-0.2	hello	0
-0.3	world	0
-0.5	<CONTACT>	0
-0.5	<APP>	0

\2-grams:
-0.1	<s> hello

\end\
ARPA
done
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
MAPPING="${tmp_dir}/input/mapping.txt" \
bash "${sdk_root}/package_workflows/prepare_flashlight_runtime_package.sh"
python3 - "${tmp_dir}/out" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
m=json.loads((p/'sdk_model.json').read_text())
assert m['lm_search']=='lm_search.json' and m['length_penalty']==0
assert m['decoder_type']=='flashlight_compact_lexicon_kenlm'
assert m['lexicon']=='lexicon.bin' and m['lexicon_format']=='compact_trie_v1'
assert not any(k in m for k in ('lm','lm_weight','contact_lm','contact_lm_metadata','word_score'))
assert not any(k.startswith('itn_') for k in m)
for name in ('one.bin','two.bin','bias.bin','lm_search.json'): assert (p/name).is_file()
assert (p/'lexicon.bin').is_file() and not (p/'lexicon.txt').exists()
words=(p/'words.txt').read_text()
assert '<CONTACT>' in words and '<APP>' in words
checks=(p/'checksums.sha256').read_text()
assert 'sdk_model.json' in checks and 'bias.bin' in checks
assert 'lm_search.json' in checks and 'meta.json' not in checks
assert (p/'output_mapping.txt').read_text() == 'hello world -> world hello\n'
PY

OUT_DIR="${tmp_dir}/out-itn" AM_MODEL="${tmp_dir}/input/model.onnx" \
SENTENCEPIECE_MODEL="${tmp_dir}/input/sentencepiece.model" \
TOKENS_FILE="${tmp_dir}/input/tokens.txt" \
LM_SEARCH_JSON="${tmp_dir}/input/lm_search.json" \
LM_BIN_DIR="${tmp_dir}/input" WORDS_FILE="${tmp_dir}/input/words.txt" \
LEXICON_FILE="${tmp_dir}/input/lexicon.txt" \
ITN_LANGUAGE=en ITN_TAGGER="${tmp_dir}/input/en_itn_tagger.fst" \
ITN_VERBALIZER="${tmp_dir}/input/en_itn_verbalizer.fst" \
bash "${sdk_root}/package_workflows/prepare_flashlight_runtime_package.sh"
python3 - "${tmp_dir}/out-itn" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
m=json.loads((p/'sdk_model.json').read_text())
assert m['itn_language']=='en'
assert m['sentencepiece_model']=='sentencepiece.model'
assert (p/'sentencepiece.model').is_file()
assert 'sentencepiece.model' in (p/'checksums.sha256').read_text()
assert m['itn_tagger']=='en_itn_tagger.fst'
assert m['itn_verbalizer']=='en_itn_verbalizer.fst'
checks=(p/'checksums.sha256').read_text()
assert 'en_itn_tagger.fst' in checks and 'en_itn_verbalizer.fst' in checks
PY

cat > "${tmp_dir}/input/conflicting_mapping.txt" <<'EOF'
hello -> world
hello world -> world
EOF
if OUT_DIR="${tmp_dir}/conflict-out" \
   AM_MODEL="${tmp_dir}/input/model.onnx" \
   TOKENS_FILE="${tmp_dir}/input/tokens.txt" \
   LM_SEARCH_JSON="${tmp_dir}/input/lm_search.json" \
   LM_BIN_DIR="${tmp_dir}/input" WORDS_FILE="${tmp_dir}/input/words.txt" \
   LEXICON_FILE="${tmp_dir}/input/lexicon.txt" \
   MAPPING="${tmp_dir}/input/conflicting_mapping.txt" \
   bash "${sdk_root}/package_workflows/prepare_flashlight_runtime_package.sh" \
   2>"${tmp_dir}/prefix-error.txt"; then
  echo "packaging accepted conflicting pre-LM mapping prefixes" >&2
  exit 1
fi
grep -Fq "is a strict prefix" "${tmp_dir}/prefix-error.txt"

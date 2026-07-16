#!/usr/bin/env bash
set -euo pipefail

sdk_root="$(cd "$(dirname "$0")/.." && pwd)"
package_script="${sdk_root}/package_workflows/prepare_flashlight_runtime_package.sh"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/asr-sdk-package-test.XXXXXX")"
trap 'rm -rf -- "${tmp_dir}"' EXIT

mkdir -p "${tmp_dir}/am" "${tmp_dir}/resources"
: > "${tmp_dir}/am/model.onnx"
: > "${tmp_dir}/resources/main.lm"
: > "${tmp_dir}/resources/contact.lm"
cat > "${tmp_dir}/resources/contact.meta.json" <<'EOF'
{
  "format": "pattern_bias_v1",
  "max_bonus": 9
}
EOF

cat > "${tmp_dir}/am/tokens.txt" <<'EOF'
<blk> 0
▁ 1
EOF
cat > "${tmp_dir}/resources/words.txt" <<'EOF'
<unk> 0
hello 1
EOF
cat > "${tmp_dir}/resources/lexicon.txt" <<'EOF'
hello ▁hello
EOF

run_package() {
  local out_dir="$1"
  shift
  AM_DIR="${tmp_dir}/am" \
    MAIN_LM_BIN="${tmp_dir}/resources/main.lm" \
    WORDS_FILE="${tmp_dir}/resources/words.txt" \
    LEXICON_FILE="${tmp_dir}/resources/lexicon.txt" \
    OUT_DIR="${out_dir}" \
    "$@" "${package_script}"
}

main_only_out="${tmp_dir}/main-only"
run_package "${main_only_out}" env
if grep -Eq '"contact_lm"|"contact_lm_weight"|"contact_lm_mode"|"contact_lm_metadata"|"contact_class_word"' \
    "${main_only_out}/sdk_model.json"; then
  echo "main-only package unexpectedly declares contact resources" >&2
  exit 1
fi
if [[ -e "${main_only_out}/contact_lm.bin" ]]; then
  echo "main-only package unexpectedly copied a contact LM" >&2
  exit 1
fi
if [[ "$(wc -l < "${main_only_out}/checksums.sha256")" != "6" ]]; then
  echo "main-only package checksum list is incomplete" >&2
  exit 1
fi

contact_out="${tmp_dir}/contact"
run_package "${contact_out}" env \
  CONTACT_LM_BIN="${tmp_dir}/resources/contact.lm" \
  CONTACT_LM_METADATA="${tmp_dir}/resources/contact.meta.json" \
  CONTACT_LM_WEIGHT=2.25
grep -Fqx '  "contact_lm": "contact_lm.bin",' \
  "${contact_out}/sdk_model.json"
grep -Fqx '  "contact_lm_weight": 2.25,' \
  "${contact_out}/sdk_model.json"
grep -Fqx '  "contact_lm_mode": "pattern_bias_v1",' \
  "${contact_out}/sdk_model.json"
grep -Fqx '  "contact_lm_metadata": "contact_lm.meta.json",' \
  "${contact_out}/sdk_model.json"
grep -Fqx '  "contact_lm_accumulation_factor": 0.5,' \
  "${contact_out}/sdk_model.json"
grep -Fqx '  "contact_class_word": "<CONTACT>",' \
  "${contact_out}/sdk_model.json"
test -f "${contact_out}/contact_lm.bin"
test -f "${contact_out}/contact_lm.meta.json"
grep -Fqx '<CONTACT> 2' "${contact_out}/words.txt"
grep -Fq 'contact_lm.bin' "${contact_out}/checksums.sha256"
grep -Fq 'contact_lm.meta.json' "${contact_out}/checksums.sha256"
if [[ "$(wc -l < "${contact_out}/checksums.sha256")" != "8" ]]; then
  echo "contact package checksum list is incomplete" >&2
  exit 1
fi

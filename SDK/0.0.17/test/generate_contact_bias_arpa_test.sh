#!/usr/bin/env bash
set -euo pipefail

sdk_root="$(cd "$(dirname "$0")/.." && pwd)"
tool="${sdk_root}/../../kenlm_lm/tools/generate_contact_bias_arpa.py"
build_binary="${sdk_root}/third_party/kenlm-install/bin/build_binary"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/asr-sdk-contact-arpa.XXXXXX")"
trap 'rm -rf -- "${tmp_dir}"' EXIT

cat > "${tmp_dir}/rules.tsv" <<'EOF'
call <CONTACT>	9
MESSAGE TO <CONTACT>	7.5
<CONTACT>	5
EOF

python3 "${tool}" \
  --rules "${tmp_dir}/rules.tsv" \
  --arpa "${tmp_dir}/contact.arpa" \
  --metadata "${tmp_dir}/contact.meta.json"

grep -Fq -- $'-9\tCALL <CONTACT>' "${tmp_dir}/contact.arpa"
grep -Fq -- $'-7.5\tMESSAGE TO <CONTACT>' "${tmp_dir}/contact.arpa"
grep -Fq -- $'-5\t<s> <CONTACT>' "${tmp_dir}/contact.arpa"
python3 -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["format"] == "pattern_bias_v1"
assert data["max_bonus"] == 9
assert data["standalone_rule"] is True
' "${tmp_dir}/contact.meta.json"

"${build_binary}" "${tmp_dir}/contact.arpa" "${tmp_dir}/contact.bin"
test -s "${tmp_dir}/contact.bin"

printf 'CALL <CONTACT> LATER\t9\n' > "${tmp_dir}/invalid.tsv"
if python3 "${tool}" --rules "${tmp_dir}/invalid.tsv" \
    --arpa "${tmp_dir}/bad.arpa" --metadata "${tmp_dir}/bad.json"; then
  echo "generator accepted a non-trailing <CONTACT>" >&2
  exit 1
fi

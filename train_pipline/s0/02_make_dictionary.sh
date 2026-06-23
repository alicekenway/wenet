#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "${script_dir}/_common.sh"

train_set=train_l
data_dir=data
dict=data/dict/lang_char.txt
space_symbol="$(printf '\342\226\201')"

help_message="
Usage: $0 [options]

Purpose:
  Build the character dictionary used by the Conformer/U2++ recipe.

Options:
  --train-set NAME               Prepared training set name, for example train_l.
  --data-dir DIR                 WeNet data directory from 01_prepare_data.sh.
  --dict FILE                    Output dictionary path.

Inputs:
  --data-dir/--train-set/text

Outputs:
  --dict
"

. tools/parse_options.sh || exit 1

input_text="${data_dir}/${train_set}/text"

print_section "Inputs"
print_kv "input_text" "${input_text}"

print_section "Outputs"
print_kv "dict" "${dict}"

require_file "${input_text}"
mkdir -p "$(dirname "${dict}")"

{
  echo "<blank> 0"
  echo "<unk> 1"
  echo "<sos/eos> 2"
  echo "${space_symbol} 3"
} > "${dict}"

tools/text2token.py -s 1 -n 1 --space "${space_symbol}" "${input_text}" \
  | cut -f 2- -d" " \
  | tr " " "\n" \
  | sort \
  | uniq \
  | grep -a -v -e '^\s*$' \
  | grep -v "${space_symbol}" \
  | awk '{print $0 " " NR+3}' >> "${dict}"

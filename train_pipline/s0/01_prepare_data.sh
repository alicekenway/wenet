#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "${script_dir}/_common.sh"

train_subset=L
wenetspeech_data_dir=/ssd/nfs07/binbinzhang/wenetspeech
data_dir=data
prep_stage=1
prefix=

help_message="
Usage: $0 [options]

Purpose:
  Convert the downloaded WenetSpeech metadata into WeNet data directories.

Options:
  --train-subset L|M|S|W         Training subset to prepare.
  --wenetspeech-data-dir DIR     Input WenetSpeech directory.
  --data-dir DIR                 Output WeNet data directory.
  --prep-stage INT               Stage passed to local/wenetspeech_data_prep.sh.
  --prefix NAME                  Optional prefix passed to local/wenetspeech_data_prep.sh.

Inputs:
  DIR/WenetSpeech.json and DIR/audio from --wenetspeech-data-dir.

Outputs:
  --data-dir/train_<subset>, --data-dir/dev, --data-dir/test_net, --data-dir/test_meeting.
"

. tools/parse_options.sh || exit 1

train_set=$(resolve_train_set "${train_subset}")

print_section "Inputs"
print_kv "wenetspeech_data_dir" "${wenetspeech_data_dir}"
print_kv "metadata" "${wenetspeech_data_dir}/WenetSpeech.json"
print_kv "audio" "${wenetspeech_data_dir}/audio"
print_kv "train_subset" "${train_subset}"

print_section "Outputs"
print_kv "data_dir" "${data_dir}"
print_kv "train_set" "${data_dir}/${prefix:+${prefix}_}${train_set}"
print_kv "dev_set" "${data_dir}/${prefix:+${prefix}_}dev"
print_kv "test_net" "${data_dir}/${prefix:+${prefix}_}test_net"
print_kv "test_meeting" "${data_dir}/${prefix:+${prefix}_}test_meeting"

require_file "${wenetspeech_data_dir}/WenetSpeech.json"
require_dir "${wenetspeech_data_dir}/audio"

local/wenetspeech_data_prep.sh \
  --train-subset "${train_subset}" \
  --stage "${prep_stage}" \
  --prefix "${prefix}" \
  "${wenetspeech_data_dir}" \
  "${data_dir}"

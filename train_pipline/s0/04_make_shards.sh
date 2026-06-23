#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "${script_dir}/_common.sh"

train_set=train_l
dev_set=dev
test_sets="test_net test_meeting"
data_dir=data
shards_dir=/ssd/nfs06/unified_data/wenetspeech_shards
resample_rate=16000
num_utts_per_shard=1000
num_threads=32

help_message="
Usage: $0 [options]

Purpose:
  Convert prepared wav.scp/text/segments files into shard lists for large-scale training.

Options:
  --train-set NAME               Prepared training set name.
  --dev-set NAME                 Prepared dev set name.
  --test-sets LIST               Space-separated test set names.
  --data-dir DIR                 WeNet data directory.
  --shards-dir DIR               Output shard root directory.
  --resample-rate INT            Audio resample rate.
  --num-utts-per-shard INT       Utterances per shard.
  --num-threads INT              Shard creation thread count.

Inputs:
  For each set: --data-dir/<set>/wav.scp, text, and segments.

Outputs:
  Shard files under --shards-dir/<set> and data lists at --data-dir/<set>/data.list.
"

. tools/parse_options.sh || exit 1

all_sets="${dev_set} ${test_sets} ${train_set}"

print_section "Inputs"
print_kv "data_dir" "${data_dir}"
print_kv "sets" "${all_sets}"

print_section "Outputs"
print_kv "shards_dir" "${shards_dir}"
for set_name in ${all_sets}; do
  print_kv "${set_name} data.list" "${data_dir}/${set_name}/data.list"
done

echo
echo "This stage can require about 1.2T disk space and can take many hours on WenetSpeech L."

for set_name in ${all_sets}; do
  require_file "${data_dir}/${set_name}/wav.scp"
  require_file "${data_dir}/${set_name}/text"
  require_file "${data_dir}/${set_name}/segments"

  dst="${shards_dir}/${set_name}"
  mkdir -p "${dst}"
  tools/make_shard_list.py \
    --resample "${resample_rate}" \
    --num_utts_per_shard "${num_utts_per_shard}" \
    --num_threads "${num_threads}" \
    --segments "${data_dir}/${set_name}/segments" \
    "${data_dir}/${set_name}/wav.scp" \
    "${data_dir}/${set_name}/text" \
    "$(realpath "${dst}")" \
    "${data_dir}/${set_name}/data.list"
done

#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "${script_dir}/_common.sh"

train_set=train_l
data_dir=data
train_config=conf/train_u2++_conformer.yaml
cmvn_sampling_divisor=20
num_workers=16
sampled_wav_scp=
cmvn_out=

help_message="
Usage: $0 [options]

Purpose:
  Sample training wav.scp and compute global CMVN statistics.

Options:
  --train-set NAME               Prepared training set name.
  --data-dir DIR                 WeNet data directory.
  --train-config FILE            Model config used by compute_cmvn_stats.py.
  --cmvn-sampling-divisor INT    20 means sample about 5 percent of wav.scp.
  --num-workers INT              Worker count for CMVN computation.
  --sampled-wav-scp FILE         Optional sampled wav.scp output.
  --cmvn-out FILE                Optional CMVN output path.

Inputs:
  --data-dir/--train-set/wav.scp and --train-config.

Outputs:
  --sampled-wav-scp and --cmvn-out.
"

. tools/parse_options.sh || exit 1

wav_scp="${data_dir}/${train_set}/wav.scp"
sampled_wav_scp=${sampled_wav_scp:-"${data_dir}/${train_set}/wav.scp.sampled"}
cmvn_out=${cmvn_out:-"${data_dir}/${train_set}/global_cmvn"}

print_section "Inputs"
print_kv "wav_scp" "${wav_scp}"
print_kv "train_config" "${train_config}"
print_kv "cmvn_sampling_divisor" "${cmvn_sampling_divisor}"

print_section "Outputs"
print_kv "sampled_wav_scp" "${sampled_wav_scp}"
print_kv "cmvn_out" "${cmvn_out}"

require_file "${wav_scp}"
require_file "${train_config}"

if [ "${cmvn_sampling_divisor}" -le 0 ]; then
  die "--cmvn-sampling-divisor must be greater than 0"
fi

full_size=$(wc -l < "${wav_scp}")
if [ "${full_size}" -le 0 ]; then
  die "empty wav.scp: ${wav_scp}"
fi

sampling_size=$((full_size / cmvn_sampling_divisor))
if [ "${sampling_size}" -lt 1 ]; then
  sampling_size=1
fi

mkdir -p "$(dirname "${sampled_wav_scp}")" "$(dirname "${cmvn_out}")"
shuf -n "${sampling_size}" "${wav_scp}" > "${sampled_wav_scp}"

python3 tools/compute_cmvn_stats.py \
  --num_workers "${num_workers}" \
  --train_config "${train_config}" \
  --in_scp "${sampled_wav_scp}" \
  --out_cmvn "${cmvn_out}"

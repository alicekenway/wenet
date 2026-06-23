#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "${script_dir}/_common.sh"

model_dir=exp/u2pp_conformer
average_num=5
average_mode=step
max_step=88888888
train_yaml=
checkpoint=
output_file=

help_message="
Usage: $0 [options]

Purpose:
  Export a trained checkpoint to WeNet JIT zip format.

Options:
  --model-dir DIR                Trained model directory.
  --average-num INT              Used only for the default checkpoint name.
  --average-mode step|epoch      Used only for the default checkpoint name.
  --max-step INT                 Used only for the default checkpoint name.
  --train-yaml FILE              Input train.yaml. Defaults to --model-dir/train.yaml.
  --checkpoint FILE              Input checkpoint. Defaults to the output from 06_decode_and_score.sh.
  --output-file FILE             Output JIT zip. Defaults to --model-dir/final.zip.

Inputs:
  train.yaml and a checkpoint.

Outputs:
  final.zip or the path set by --output-file.
"

. tools/parse_options.sh || exit 1

train_yaml=${train_yaml:-"${model_dir}/train.yaml"}
checkpoint=${checkpoint:-"${model_dir}/avg${average_num}_mode${average_mode}_max${max_step}.pt"}
output_file=${output_file:-"${model_dir}/final.zip"}

print_section "Inputs"
print_kv "train_yaml" "${train_yaml}"
print_kv "checkpoint" "${checkpoint}"

print_section "Outputs"
print_kv "output_file" "${output_file}"

require_file "${train_yaml}"
require_file "${checkpoint}"
mkdir -p "$(dirname "${output_file}")"

python wenet/bin/export_jit.py \
  --config "${train_yaml}" \
  --checkpoint "${checkpoint}" \
  --output_file "${output_file}"

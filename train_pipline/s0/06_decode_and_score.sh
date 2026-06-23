#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "${script_dir}/_common.sh"

model_dir=exp/u2pp_conformer
data_dir=data
data_type=shard
test_sets="test_net test_meeting"
dev_set=dev
average_checkpoint=true
average_num=5
average_mode=step
max_step=88888888
decode_checkpoint=
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
decoding_chunk_size=
ctc_weight=0.5
reverse_weight=0.0
blank_penalty=0.0
length_penalty=0.0
decode_batch=16
beam_size=10
gpu_list=auto

help_message="
Usage: $0 [options]

Purpose:
  Average checkpoints when requested, run recognition, and compute CER/WER files.

Options:
  --model-dir DIR                Trained model directory.
  --data-dir DIR                 WeNet data directory.
  --data-type raw|shard          Decode data type.
  --test-sets LIST               Space-separated test sets.
  --dev-set NAME                 Dev set name; decoded after test sets.
  --average-checkpoint true|false
  --average-num INT              Number of checkpoints to average.
  --average-mode step|epoch
  --max-step INT                 Maximum step for average_model.py.
  --decode-checkpoint FILE       Checkpoint to decode when not averaging, or override output.
  --decode-modes LIST            Space-separated recognition modes.
  --decoding-chunk-size INT      Optional chunk size. Empty means do not pass this option.
  --ctc-weight FLOAT
  --reverse-weight FLOAT
  --blank-penalty FLOAT
  --length-penalty FLOAT
  --decode-batch INT
  --beam-size INT
  --gpu-list auto|0,1|-1         CUDA_VISIBLE_DEVICES value. -1 means CPU.

Inputs:
  --model-dir/train.yaml, checkpoint, and --data-dir/<set>/data.list/text.

Outputs:
  Averaged checkpoint and decode result directories under --model-dir.
"

. tools/parse_options.sh || exit 1

print_section "Inputs"
print_kv "model_dir" "${model_dir}"
print_kv "train_yaml" "${model_dir}/train.yaml"
print_kv "test_sets" "${test_sets}"
print_kv "dev_set" "${dev_set}"
print_kv "decode_modes" "${decode_modes}"

require_file "${model_dir}/train.yaml"
for set_name in ${test_sets} ${dev_set}; do
  require_file "${data_dir}/${set_name}/data.list"
  require_file "${data_dir}/${set_name}/text"
done

setup_cuda "${gpu_list}"

if [ "${average_checkpoint}" = true ]; then
  decode_checkpoint=${decode_checkpoint:-"${model_dir}/avg${average_num}_mode${average_mode}_max${max_step}.pt"}
  print_section "Checkpoint averaging"
  print_kv "output_checkpoint" "${decode_checkpoint}"
  python wenet/bin/average_model.py \
    --dst_model "${decode_checkpoint}" \
    --src_path "${model_dir}" \
    --num "${average_num}" \
    --mode "${average_mode}" \
    --max_step "${max_step}" \
    --val_best
else
  [ -n "${decode_checkpoint}" ] || die "--decode-checkpoint is required when --average-checkpoint false"
  require_file "${decode_checkpoint}"
fi

print_section "Outputs"
print_kv "decode_checkpoint" "${decode_checkpoint}"
for set_name in ${test_sets} ${dev_set}; do
  base=$(basename "${decode_checkpoint}")
  result_dir="${model_dir}/${set_name}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}"
  print_kv "${set_name} result_dir" "${result_dir}"
done

running=0
device_index=0
for set_name in ${test_sets} ${dev_set}; do
  base=$(basename "${decode_checkpoint}")
  result_dir="${model_dir}/${set_name}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}"
  mkdir -p "${result_dir}"

  device_id="${device_ids[$((device_index % num_gpus))]}"
  echo "Testing ${set_name} on GPU ${device_id}"

  cmd=(
    python
    wenet/bin/recognize.py
    --gpu "${device_id}"
    --modes ${decode_modes}
    --config "${model_dir}/train.yaml"
    --data_type "${data_type}"
    --test_data "${data_dir}/${set_name}/data.list"
    --checkpoint "${decode_checkpoint}"
    --beam_size "${beam_size}"
    --batch_size "${decode_batch}"
    --blank_penalty "${blank_penalty}"
    --length_penalty "${length_penalty}"
    --ctc_weight "${ctc_weight}"
    --reverse_weight "${reverse_weight}"
    --result_dir "${result_dir}"
  )
  if [ -n "${decoding_chunk_size}" ]; then
    cmd+=(--decoding_chunk_size "${decoding_chunk_size}")
  fi

  "${cmd[@]}" &
  running=$((running + 1))
  device_index=$((device_index + 1))
  if [ "${running}" -ge "${num_gpus}" ]; then
    wait
    running=0
  fi
done
wait

for set_name in ${test_sets} ${dev_set}; do
  base=$(basename "${decode_checkpoint}")
  result_dir="${model_dir}/${set_name}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}"
  for mode in ${decode_modes}; do
    python tools/compute-wer.py --char=1 --v=1 \
      "${data_dir}/${set_name}/text" "${result_dir}/${mode}/text" > "${result_dir}/${mode}/wer"
  done
done

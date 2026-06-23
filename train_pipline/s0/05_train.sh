#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "${script_dir}/_common.sh"

train_set=train_l
dev_set=dev
data_dir=data
data_type=shard
train_config=conf/train_u2++_conformer.yaml
checkpoint=
model_dir=exp/u2pp_conformer
tensorboard_dir=tensorboard
num_workers=8
prefetch=10
train_engine=torch_ddp
deepspeed_config=../whisper/conf/ds_stage1.json
deepspeed_save_states="model+optimizer"
host_node_addr=localhost:0
num_nodes=1
rdzv_id=2023
dist_backend=nccl
gpu_list=auto
timeout=1200

help_message="
Usage: $0 [options]

Purpose:
  Train the WenetSpeech s0 model from prepared shard data.

Options:
  --train-set NAME               Training set name.
  --dev-set NAME                 Dev set name.
  --data-dir DIR                 WeNet data directory.
  --data-type raw|shard          Training data type.
  --train-config FILE            Training YAML.
  --checkpoint FILE              Optional checkpoint to initialize/resume from.
  --model-dir DIR                Output model directory.
  --tensorboard-dir DIR          TensorBoard output directory.
  --num-workers INT              Data loader worker count.
  --prefetch INT                 Data prefetch count.
  --train-engine torch_ddp|deepspeed
  --deepspeed-config FILE        DeepSpeed config.
  --deepspeed-save-states VALUE  model+optimizer or model_only.
  --host-node-addr HOST:PORT     torchrun rendezvous endpoint.
  --num-nodes INT                Number of nodes.
  --rdzv-id ID                   torchrun rendezvous id.
  --dist-backend nccl|gloo       Distributed backend.
  --gpu-list auto|0,1|-1         CUDA_VISIBLE_DEVICES value. -1 means CPU.
  --timeout INT                  Distributed timeout.

Inputs:
  --train-config, optional --checkpoint, train/cv data lists.

Outputs:
  --model-dir/train.yaml, checkpoints, logs, and TensorBoard files.
"

. tools/parse_options.sh || exit 1

train_data="${data_dir}/${train_set}/data.list"
cv_data="${data_dir}/${dev_set}/data.list"

print_section "Inputs"
print_kv "train_config" "${train_config}"
print_kv "train_data" "${train_data}"
print_kv "cv_data" "${cv_data}"
print_kv "checkpoint" "${checkpoint:-<none>}"

print_section "Outputs"
print_kv "model_dir" "${model_dir}"
print_kv "train_yaml" "${model_dir}/train.yaml"
print_kv "tensorboard_dir" "${tensorboard_dir}"

require_file "${train_config}"
require_file "${train_data}"
require_file "${cv_data}"
if [ -n "${checkpoint}" ]; then
  require_file "${checkpoint}"
fi

setup_cuda "${gpu_list}"
mkdir -p "${model_dir}"

if [ "${train_engine}" = "deepspeed" ]; then
  print_kv "train_engine" "deepspeed"
else
  print_kv "train_engine" "torch ddp"
fi
print_kv "num_nodes" "${num_nodes}"
print_kv "proc_per_node" "${num_gpus}"

cmd=(
  torchrun
  --nnodes="${num_nodes}"
  --nproc_per_node="${num_gpus}"
  --rdzv_endpoint="${host_node_addr}"
  --rdzv_id="${rdzv_id}"
  --rdzv_backend="c10d"
  wenet/bin/train.py
  --train_engine "${train_engine}"
  --config "${train_config}"
  --data_type "${data_type}"
  --train_data "${train_data}"
  --cv_data "${cv_data}"
  --model_dir "${model_dir}"
  --tensorboard_dir "${tensorboard_dir}"
  --ddp.dist_backend "${dist_backend}"
  --num_workers "${num_workers}"
  --prefetch "${prefetch}"
  --pin_memory
  --timeout "${timeout}"
  --deepspeed_config "${deepspeed_config}"
  --deepspeed.save_states "${deepspeed_save_states}"
)

if [ -n "${checkpoint}" ]; then
  cmd+=(--checkpoint "${checkpoint}")
fi

"${cmd[@]}"

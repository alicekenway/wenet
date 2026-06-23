#!/bin/bash

# Shared setup for the split WenetSpeech s0 pipeline scripts.

pipeline_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wenet_root=$(cd "${pipeline_dir}/../.." && pwd)
recipe_dir=${recipe_dir:-"${wenet_root}/examples/wenetspeech/s0"}

cd "${recipe_dir}" || exit 1
. ./path.sh || exit 1

set -u
set -o pipefail

die() {
  echo "$0: $*" >&2
  exit 1
}

require_file() {
  [ -f "$1" ] || die "missing required file: $1"
}

require_dir() {
  [ -d "$1" ] || die "missing required directory: $1"
}

print_section() {
  echo
  echo "== $* =="
}

print_kv() {
  printf '  %-24s %s\n' "$1:" "$2"
}

resolve_train_set() {
  local subset="$1"
  printf 'train_%s' "$(echo "${subset}" | tr 'A-Z' 'a-z')"
}

setup_cuda() {
  local requested="${1:-auto}"

  if [ "${requested}" = "auto" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
      requested="${CUDA_VISIBLE_DEVICES}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
      local detected
      detected=$(nvidia-smi -L | wc -l)
      if [ "${detected}" -gt 0 ]; then
        requested=$(seq -s, 0 $((detected - 1)))
      else
        requested="-1"
      fi
    else
      requested="-1"
    fi
  fi

  export CUDA_VISIBLE_DEVICES="${requested}"
  IFS=',' read -r -a device_ids <<< "${CUDA_VISIBLE_DEVICES}"
  if [ "${#device_ids[@]}" -eq 0 ]; then
    device_ids=(-1)
  fi
  num_gpus=${#device_ids[@]}

  print_kv "CUDA_VISIBLE_DEVICES" "${CUDA_VISIBLE_DEVICES}"
  print_kv "device_ids" "${device_ids[*]}"
}

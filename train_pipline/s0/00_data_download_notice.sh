#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "${script_dir}/_common.sh"

dataset_url=https://github.com/wenet-e2e/WenetSpeech
wenetspeech_data_dir=/ssd/nfs07/binbinzhang/wenetspeech

help_message="
Usage: $0 [options]

Purpose:
  Show where to get WenetSpeech and which local directory later stages expect.

Options:
  --dataset-url URL              WenetSpeech download/instruction URL.
  --wenetspeech-data-dir DIR     Expected local WenetSpeech dataset directory.

Inputs:
  Remote WenetSpeech dataset source.

Outputs:
  A local dataset directory containing WenetSpeech.json and audio/.
"

. tools/parse_options.sh || exit 1

print_section "Purpose"
echo "Download WenetSpeech manually before running data preparation."

print_section "Inputs"
print_kv "dataset_url" "${dataset_url}"

print_section "Expected outputs"
print_kv "wenetspeech_data_dir" "${wenetspeech_data_dir}"
print_kv "metadata" "${wenetspeech_data_dir}/WenetSpeech.json"
print_kv "audio" "${wenetspeech_data_dir}/audio"

echo
echo "Follow the dataset instructions at ${dataset_url}, then pass the local path to 01_prepare_data.sh with --wenetspeech-data-dir."

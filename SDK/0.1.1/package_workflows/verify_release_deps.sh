#!/usr/bin/env bash
set -euo pipefail

lib=${1:-build/libasr_sdk.so}

if [[ ! -f "${lib}" ]]; then
  echo "FAIL: libasr_sdk.so not found: ${lib}" >&2
  exit 1
fi

ldd "${lib}" | tee /tmp/asr_sdk_ldd.txt
if grep -q 'libwenet' /tmp/asr_sdk_ldd.txt; then
  echo "FAIL: libasr_sdk.so depends on libwenet*.so" >&2
  exit 1
fi
if ! grep -q 'libonnxruntime.so.1' /tmp/asr_sdk_ldd.txt; then
  echo "FAIL: libasr_sdk.so does not depend on libonnxruntime.so.1" >&2
  exit 1
fi
if nm -D "${lib}" | grep -E ' wenet_| _ZN5wenet' >/tmp/asr_sdk_symbols.txt; then
  cat /tmp/asr_sdk_symbols.txt >&2
  echo "FAIL: unexpected WeNet symbols exported" >&2
  exit 1
fi

echo "PASS: dynamic ORT dependency present"
echo "PASS: no dynamic WeNet dependency"
echo "PASS: WeNet symbols hidden"

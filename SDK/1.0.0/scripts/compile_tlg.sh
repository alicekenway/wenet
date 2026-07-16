#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  compile_tlg.sh --from-text TLG.txt TLG.fst

This helper currently supports compiling an existing OpenFst text graph into a
binary TLG.fst for SDK packaging and tests.

Full T/L/G composition from tokens, lexicon, and G.fst requires the offline
Kaldi/OpenFst graph-building toolchain. Keep that process off-device and copy
the final TLG.fst into the model package.
EOF
}

find_fstcompile() {
  if command -v fstcompile >/dev/null 2>&1; then
    command -v fstcompile
    return
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/fstcompile" ]]; then
    printf '%s\n' "${CONDA_PREFIX}/bin/fstcompile"
    return
  fi
  if [[ -x "/home/jinyang_wang/miniforge3/envs/wenet/bin/fstcompile" ]]; then
    printf '%s\n' "/home/jinyang_wang/miniforge3/envs/wenet/bin/fstcompile"
    return
  fi
  return 1
}

if [[ $# -ne 3 || "${1}" != "--from-text" ]]; then
  usage
  exit 2
fi

input_txt="$2"
output_fst="$3"
if [[ ! -f "${input_txt}" ]]; then
  echo "missing input graph text: ${input_txt}" >&2
  exit 1
fi

fstcompile_bin="$(find_fstcompile)" || {
  echo "fstcompile not found; install OpenFst first" >&2
  exit 1
}

"${fstcompile_bin}" "${input_txt}" "${output_fst}"
echo "wrote ${output_fst}"

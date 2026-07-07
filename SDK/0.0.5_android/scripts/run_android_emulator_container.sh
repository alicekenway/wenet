#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-us-docker.pkg.dev/android-emulator-268719/images/30-google-x64-no-metrics:30.1.2}"
NAME="${NAME:-asr-android-emulator-30}"

docker run --rm -d \
  --name "${NAME}" \
  --device /dev/kvm \
  -p 5555:5555 \
  -p 8554:8554 \
  "${IMAGE}"

echo "Container started: ${NAME}"
echo "Connect with: adb connect localhost:5555"

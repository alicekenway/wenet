FROM ubuntu:22.04

ARG ANDROID_CMDLINE_TOOLS_URL=https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
ARG ANDROID_PLATFORM=android-35
ARG ANDROID_BUILD_TOOLS=35.0.0
ARG ANDROID_NDK_VERSION=26.3.11579264

ENV DEBIAN_FRONTEND=noninteractive
ENV ANDROID_HOME=/opt/android-sdk
ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV ANDROID_NDK=/opt/android-sdk/ndk/${ANDROID_NDK_VERSION}
ENV PATH=${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    file \
    git \
    make \
    ninja-build \
    openjdk-17-jdk \
    python3 \
    unzip \
    zip \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${ANDROID_HOME}/cmdline-tools \
  && curl -fsSL "${ANDROID_CMDLINE_TOOLS_URL}" -o /tmp/android-commandline-tools.zip \
  && unzip -q /tmp/android-commandline-tools.zip -d /tmp/android-commandline-tools \
  && mv /tmp/android-commandline-tools/cmdline-tools ${ANDROID_HOME}/cmdline-tools/latest \
  && rm -rf /tmp/android-commandline-tools /tmp/android-commandline-tools.zip

RUN yes | sdkmanager --licenses \
  && sdkmanager \
    "platform-tools" \
    "platforms;${ANDROID_PLATFORM}" \
    "build-tools;${ANDROID_BUILD_TOOLS}" \
    "cmake;3.22.1" \
    "ndk;${ANDROID_NDK_VERSION}"

WORKDIR /work

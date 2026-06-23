# Embedded Porting

Use the C ABI in `include/wenet_sdk/c_api.h` for firmware, JNI, Rust, Python,
or Go bindings. Returned strings are owned by the stream and remain valid until
the next result call on that stream.

Do not run model inference or WFST search inside an audio callback. Copy PCM into
a bounded buffer from the callback and decode on a worker thread.

The `asr_stream_mic` demo uses blocking PortAudio reads when compiled with
`WENETSDK_ENABLE_PORTAUDIO=ON`; decode work is performed outside any audio
callback.

The default SDK build avoids external runtime dependencies. For deployment,
enable and statically package ONNX Runtime and OpenFst as appropriate for the
target.

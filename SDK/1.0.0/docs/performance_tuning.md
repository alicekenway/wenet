# Performance Tuning

Primary knobs:

- `chunk_size`
- `num_left_chunks`
- `blank_skip_thresh`
- `beam`
- `max_active`
- ONNX Runtime thread counts

The `benchmark` tool reports elapsed time, audio duration, chunks, and RTF:

```bash
./build/benchmark --model_dir model_dir --wav sample.wav
```

For real deployments, profile ONNX model latency and WFST active state counts
separately.

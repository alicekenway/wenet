# SDK 0.0.3 Tools

## `asr_package_eval`

`asr_package_eval` evaluates a complete ASR package against a JSONL test set.
It loads the package once, runs inference for every utterance, and writes JSONL
output. Summary calculation is handled by `summarize_asr_package_eval.py`.

Example:

```bash
/home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.0.3/build/asr_package_eval \
  --model_dir /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.3/model_flashlight \
  --metadata /home/jinyang_wang/Dev/ASR/ASR_wenet/data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000/metadata.jsonl \
  --wav_parent /home/jinyang_wang/Dev/ASR/ASR_wenet/data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000 \
  --output_json /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.3/package_eval/output.jsonl

python3 /home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.0.3/tools/summarize_asr_package_eval.py \
  --input_json /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.3/package_eval/output.jsonl \
  --summary /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.3/package_eval/summary.txt
```

Common options:

```text
--model_dir DIR       ASR package directory.
--metadata PATH       JSONL test set.
--wav_parent DIR      Parent directory used to resolve relative wav paths.
--output_json PATH    Output JSONL. Each input row is copied with added fields.
--decode_mode MODE    lm or greedy. Default is lm.
--limit N             Decode only the first N rows.
--num_threads N       SDK inference thread count.
--chunk_ms N          Feed audio in streaming chunks. 0 means whole utterance.
--wav_key KEY         Override wav JSON key.
--text_key KEY        Override reference text JSON key.
```

Decode modes:

```text
lm       Load the package through the SDK and use its configured decoder.
greedy   Load only model.onnx and tokens.txt, run AM inference, then CTC greedy.
```

In `greedy` mode the tool does not load `words.txt`, `lexicon.txt`, or
`lm.bin`. It reads `model_path` and `tokens` from `sdk_model.json` when present,
otherwise it uses `model.onnx` and `tokens.txt` under `--model_dir`. It also
honors `feature_type` from `sdk_model.json`; use `kaldi` for standard
sherpa/icefall Zipformer CTC models and `whisper` for older packages built
against the earlier frontend.

Input JSONL defaults:

- reference text key: `text`, then `sentence`, then `transcript`;
- wav key: `audiofile_path`, then `audio_filepath`, `wav`, `path`,
  `file_name`.

For the Wenetspeech sample metadata, `audiofile_path` is relative, for example
`wav/000000013.wav`, so `--wav_parent` should be the dataset directory that
contains the `wav/` folder.

Output JSONL adds these fields:

```text
hyp          ASR hypothesis.
rtf          Per-utterance decode_sec / audio_sec.
atf          Alias of rtf, kept because some existing scripts use this name.
decode_sec   Wall time spent in SDK decode for this utterance.
audio_sec    Utterance audio duration from the WAV file.
decode_mode  lm or greedy.
error        Only present if WAV reading or decoding failed.
```

## `summarize_asr_package_eval.py`

`summarize_asr_package_eval.py` reads the output JSONL from `asr_package_eval`
and computes WER, SER, aggregate RTF, and wrong-sentence details.

Example:

```bash
python3 /home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.0.3/tools/summarize_asr_package_eval.py \
  --input_json /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.3/package_eval/output.jsonl \
  --summary /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.3/package_eval/summary.txt \
  --detail_json /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.3/package_eval/detail.jsonl
```

The optional `--detail_json` output copies each inference row and adds `wer`,
`ref_units`, and `edits`.

The summary file starts with:

```text
decode_mode
sentence_count
failed_count
ser
wer
rtf
audio_sec
decode_sec
wrong_sentence_count
```

Below that it lists wrong sentences with wav filename, reference, hypothesis,
and per-utterance WER.

WER tokenization policy:

- If the reference or hypothesis contains whitespace, WER is computed over
  whitespace-separated tokens.
- Otherwise, it is computed over UTF-8 characters. This makes Chinese metadata
  usable without a separate tokenizer.

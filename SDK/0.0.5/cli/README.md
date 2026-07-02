# SDK 0.0.5 Tools

## `asr_package_eval`

`asr_package_eval` evaluates a complete ASR package against a JSONL test set.
It loads the package once, runs inference for every utterance, and writes JSONL
output. Summary calculation is handled by `summarize_asr_package_eval.py`.

Example:

```bash
/home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.0.5/build_0_0_5/asr_package_eval \
  --model_dir /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.5/model_flashlight \
  --metadata /home/jinyang_wang/Dev/ASR/ASR_wenet/data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000/metadata.jsonl \
  --wav_parent /home/jinyang_wang/Dev/ASR/ASR_wenet/data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000 \
  --output_json /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.5/package_eval/output.jsonl

python3 /home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.0.5/cli/summarize_asr_package_eval.py \
  --input_json /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.5/package_eval/output.jsonl \
  --summary /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.5/package_eval/summary.txt
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
--debug true|false    In LM mode, collect shallow-fusion debug data.
--debug_log PATH      Write readable LM N-best debug blocks. Defaults next to output_json.
```

Decode modes:

```text
lm       Load the package through the SDK and use lexicon shallow fusion with KenLM.
greedy   Load only model.onnx and tokens.txt, run AM inference, then CTC greedy.
```

In `greedy` mode the tool does not load `words.txt`, `lexicon.txt`, or
`lm.bin`. It reads `model_path` and `tokens` from `sdk_model.json` when present,
otherwise it uses `model.onnx` and `tokens.txt` under `--model_dir`. The
single-ONNX acoustic backend is selected from the model's `model_type` metadata;
0.0.5 supports `zipformer2` and `wenet_ctc`. The tool also honors
`feature_type` from `sdk_model.json`; use `kaldi` for standard sherpa/icefall
Zipformer CTC and exported WeNet CTC models, and `whisper` for older packages
built against the earlier frontend.

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

With `--decode_mode lm --debug true`, debug details are written to a separate
log file instead of the JSONL. The format is one utterance block, then an empty
line:

```text
ref#audio_path#reference
hyp1#hyp text#am score#lm score
hyp2#hyp text#am score#lm score
```

## `summarize_asr_package_eval.py`

`summarize_asr_package_eval.py` reads the output JSONL from `asr_package_eval`
and computes WER or CER, SER, aggregate RTF, and wrong-sentence details.

Example:

```bash
python3 /home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.0.5/cli/summarize_asr_package_eval.py \
  --input_json /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.5/package_eval/output.jsonl \
  --summary /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.5/package_eval/summary.txt \
  --detail_json /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.5/package_eval/detail.jsonl \
  --ref_regex_rules /home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.0.5/cli/ref_text_normalization_rules.tsv
```

The optional `--detail_json` output copies each inference row and adds
`metric`, `error_rate`, `unit_count`, `edits`, `substitutions`,
`insertions`, `deletions`, optional `scored_ref`, and the selected metric field.

`--ref_regex_rules` applies regex replacement rules to reference text before
WER/CER calculation. The original reference is preserved in `text`; the
reference actually used for scoring is written as `scored_ref` in
`detail_json` when it differs from the original. Rule files use one rule per
line:

```text
# regex<TAB>replacement
# one-column rules remove matches
\[[^\]]+\]\s*
\\+contact\b
```

The bundled `ref_text_normalization_rules.tsv` removes common SDS reference
tags such as `[BG]`, `\sp`, `\contact`, `\pf:ah`, and `\unintelligible`.

The summary file starts with:

```text
decode_mode
metric
ref_regex_rules, if configured
sentence_count
failed_count
ser
wer, cer, or error_rate
substitutions
insertions
deletions
substitution_rate
insertion_rate
deletion_rate
rtf
audio_sec
decode_sec
wrong_sentence_count
```

Below that it lists wrong sentences with wav filename, original reference,
optional scored reference, hypothesis, selected metric, per-utterance error
rate, and per-utterance S/I/D counts.

Metric policy:

- Use `--metric wer` for whitespace-separated word error rate.
- Use `--metric cer` for character error rate with whitespace ignored.
- The default `--metric auto` uses WER when either side has whitespace,
  otherwise CER.
- Scoring is case-insensitive; original reference and hypothesis text are still
  preserved in the summary and detail files.

# KenLM/Flashlight Runtime Tools for SDK 0.0.3

This directory keeps the reusable data-processing commands for the SDK 0.0.3
Flashlight + KenLM path.

Run order:

```bash
cd /path/to/ASR_wenet

python3 wenet/kenlm_lm/tools/sample_eval_metadata.py
wenet/kenlm_lm/run/1_train_lm.sh
wenet/kenlm_lm/run/2_convert_to_bin.sh
wenet/kenlm_lm/run/3_build_words_lm.sh
wenet/kenlm_lm/run/4_build_lexicon_for_am.sh
wenet/kenlm_lm/run/6_prepare_runtime.sh
wenet/kenlm_lm/tools/run_003_acceptance_eval.sh
```

Default inputs:

- AM package: `model/sherpa-onnx-en-wenet-gigaspeech_int8`
- LM text: `LM/wenet_lm/training/preprocess_data/wenetspeech_lm_char.txt`
- Eval metadata: `data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000/metadata.jsonl`

Main outputs:

- `wenet/kenlm_lm/models/wenetspeech_char_4gram.arpa`
- `wenet/kenlm_lm/models/lm.bin`
- `wenet/kenlm_lm/data/words.txt`
- `wenet/kenlm_lm/data/lexicon.txt`
- `test/0.0.3/model_flashlight`
- `test/0.0.3/acceptance`
- `test/0.0.3/acceptance/eval.tsv`, with `ref`, `greedy`, `lm`, and CER columns.
- `test/0.0.3/sdk_batch_flashlight_eval.tsv`, when scoring public SDK batch output.

The default KenLM pruning is `0 0 1 1`. Override it like this:

```bash
PRUNE="0 0 1 2" wenet/kenlm_lm/tools/train_kenlm_arpa.sh
```

If output is empty or strange, first check:

- `tokens.txt` IDs match the ONNX output dimension.
- `lexicon.txt` spelling tokens all exist in AM `tokens.txt`.
- `build_lexicon_for_am.py --tokenization` matches the AM token style:
  use `byte` for byte-fallback character models and `bpe` for
  SentencePiece/BPE models with word-start tokens such as `▁THE`.
- Add `--ignore-case` when the LM words and AM tokens use different letter
  casing, for example uppercase LM words with lowercase AM BPE tokens.
- `words.txt` contains the same output units used to train KenLM.
- `output_mapping.txt` is empty or contains valid `source -> target` rules.
- `test/0.0.3/acceptance/*.log` for raw words, mapped words, scores, and RTF.
- `test/0.0.3/acceptance/summary.txt` for greedy-vs-LM aggregate CER.

For English BPE/SentencePiece AM tokens, build the lexicon like this:

```bash
python3 wenet/kenlm_lm/tools/build_lexicon_for_am.py \
  --words wenet/kenlm_lm/data/words.txt \
  --tokens model/sherpa-onnx-en-wenet-gigaspeech_int8/tokens.txt \
  --output wenet/kenlm_lm/data/lexicon.txt \
  --report wenet/kenlm_lm/reports/lexicon_report.json \
  --tokenization bpe \
  --ignore-case \
  --allow-rejected
```

In BPE mode the tool maps each LM word to a token sequence by longest matching
against `tokens.txt`, preferring word-start pieces such as `▁YOU`. For example,
`PASSENGER'S` can map to `▁PASSENGER ' S`. `--allow-rejected` is useful when
the LM contains words the AM token set cannot spell, such as digits or markup;
those words are skipped and listed in the JSON report.

Decoder scoring can be tuned without editing the script:

```bash
LM_WEIGHT=0.5 WORD_SCORE=0.0 BEAM_SIZE=30 BEAM_SIZE_TOKEN=15 \
  wenet/kenlm_lm/tools/run_003_acceptance_eval.sh
```

To make the public SDK package use the same tuned settings, rebuild the package
with the same environment variables:

```bash
LM_WEIGHT=0.5 WORD_SCORE=0.0 BEAM_SIZE=30 BEAM_SIZE_TOKEN=15 \
  wenet/SDK/0.0.3/scripts/prepare_flashlight_runtime_package.sh
```

For standard sherpa/icefall Zipformer CTC models, add `FEATURE_TYPE=kaldi`.
Use `FEATURE_TYPE=whisper` only for packages that were validated with the older
Whisper-style frontend.

Score public SDK batch output against the same sampled references:

```bash
python3 wenet/kenlm_lm/tools/score_sdk_results.py \
  --result test/0.0.3/sdk_batch_flashlight.txt
```

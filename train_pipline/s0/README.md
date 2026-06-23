# Split WenetSpeech s0 Pipeline

This folder splits `examples/wenetspeech/s0/run.sh` into smaller stage scripts.
The folder name is `train_pipline` because that is the path requested.

Each script can be run from any directory, but it changes into
`examples/wenetspeech/s0` before doing work. Because of that, relative paths
such as `data`, `exp/u2pp_conformer`, and `conf/train_u2++_conformer.yaml`
are relative to `examples/wenetspeech/s0`.

Every stage supports `--help` and prints its main inputs and outputs before
running.

## Stage Map

| Script | Original stage | Purpose | Main inputs | Main outputs |
| --- | ---: | --- | --- | --- |
| `00_data_download_notice.sh` | `-1` | Show where the WenetSpeech dataset should come from | WenetSpeech download source | Local dataset directory with `WenetSpeech.json` and `audio/` |
| `01_prepare_data.sh` | `0` | Convert WenetSpeech metadata to WeNet data dirs | `WenetSpeech.json`, `audio/` | `data/train_l`, `data/dev`, `data/test_net`, `data/test_meeting` |
| `02_make_dictionary.sh` | `1` | Build the character dictionary | `data/train_l/text` | `data/dict/lang_char.txt` |
| `03_compute_cmvn.sh` | `2` | Compute global CMVN stats | `data/train_l/wav.scp`, train config | `data/train_l/wav.scp.sampled`, `data/train_l/global_cmvn` |
| `04_make_shards.sh` | `3` | Build shard lists for large-scale training | `wav.scp`, `text`, `segments` for each set | shard files and each set's `data.list` |
| `05_train.sh` | `4` | Train the U2++ Conformer model | train/dev `data.list`, train config | `exp/u2pp_conformer/train.yaml`, checkpoints, logs |
| `06_decode_and_score.sh` | `5` | Average checkpoints, decode, and compute CER/WER | trained model dir, test/dev `data.list` and `text` | averaged checkpoint and decode result dirs |
| `07_export_jit.sh` | `6` | Export a checkpoint to WeNet JIT zip | `train.yaml`, checkpoint | `exp/u2pp_conformer/final.zip` |

## Typical Run

From the WeNet repo root:

```bash
cd /home/jinyang_wang/Dev/ASR/ASR_wenet/wenet
```

First inspect the expected dataset layout:

```bash
bash train_pipline/s0/00_data_download_notice.sh \
  --wenetspeech-data-dir /path/to/wenetspeech
```

Prepare the WeNet data directory:

```bash
bash train_pipline/s0/01_prepare_data.sh \
  --wenetspeech-data-dir /path/to/wenetspeech \
  --train-subset L \
  --data-dir data
```

Build the dictionary:

```bash
bash train_pipline/s0/02_make_dictionary.sh \
  --train-set train_l \
  --data-dir data \
  --dict data/dict/lang_char.txt
```

Compute CMVN:

```bash
bash train_pipline/s0/03_compute_cmvn.sh \
  --train-set train_l \
  --data-dir data \
  --train-config conf/train_u2++_conformer.yaml \
  --cmvn-sampling-divisor 20
```

Create shards. This can require about 1.2T of disk space for WenetSpeech L.

```bash
bash train_pipline/s0/04_make_shards.sh \
  --train-set train_l \
  --dev-set dev \
  --test-sets "test_net test_meeting" \
  --data-dir data \
  --shards-dir /path/to/wenetspeech_shards
```

Train:

```bash
bash train_pipline/s0/05_train.sh \
  --train-set train_l \
  --dev-set dev \
  --data-dir data \
  --train-config conf/train_u2++_conformer.yaml \
  --model-dir exp/u2pp_conformer \
  --gpu-list 0,1,2,3
```

Decode and score:

```bash
bash train_pipline/s0/06_decode_and_score.sh \
  --model-dir exp/u2pp_conformer \
  --data-dir data \
  --test-sets "test_net test_meeting" \
  --dev-set dev \
  --gpu-list 0,1,2,3
```

Export:

```bash
bash train_pipline/s0/07_export_jit.sh \
  --model-dir exp/u2pp_conformer \
  --output-file exp/u2pp_conformer/final.zip
```

## Important Defaults

The default training config is `conf/train_u2++_conformer.yaml`.

The default model output directory is `exp/u2pp_conformer`.

`--gpu-list auto` uses `CUDA_VISIBLE_DEVICES` if it is already set. If it is
not set, the script tries to detect GPUs with `nvidia-smi`. Use `--gpu-list -1`
for CPU-only execution.

The decode stage averages checkpoints by default and writes:

```text
exp/u2pp_conformer/avg5_modestep_max88888888.pt
```

The export stage uses that same checkpoint name by default. Pass
`--checkpoint /path/to/model.pt` if you want to export a different checkpoint.

# Model Package

Expected package layout:

```text
model_dir/
  manifest.json
  config.yaml
  encoder.onnx
  ctc.onnx
  tokens.txt
  words.txt
  TLG.fst
  TLG.txt          # optional source text for small/toy graphs
  global_cmvn        # optional
  checksum.sha256    # optional
```

Validation is performed by `LoadModelMetadata` and
`ValidateModelPackageFiles`. The CLI wrapper is:

```bash
./build/validate_model_package --model_dir model_dir
```

The Python package checker performs the same deployment-oriented file and
checksum checks without requiring a compiled SDK:

```bash
scripts/validate_package.py --model_dir model_dir --require-onnx
```

`tokens.txt` and `words.txt` may use either `SYMBOL ID`, `ID SYMBOL`, or one
symbol per line.

The checked-in `model_example/TLG.fst` is a small OpenFst graph compiled from
`model_example/TLG.txt`.

Incoming PCM may use a different sample rate from `manifest.json`; the SDK
resamples it to the package sample rate before feature extraction.

If `checksum.sha256` is present, each non-comment line is validated in standard
`sha256sum` format:

```text
<64 lowercase hex digest>  relative/path
```

To write a starter manifest/config and optional checksum file:

```bash
scripts/package_model.py --model_dir model_dir --write_checksum
```

For small or prebuilt graph text fixtures, compile OpenFst text format into the
binary graph shipped by the SDK:

```bash
scripts/build_tlg.sh --from-text TLG.txt TLG.fst
```

#!/usr/bin/env python3
"""Shared, standard-library-only dataset runner for SDK 0.0.13 through 0.0.16."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

VERSIONS = ('0.0.13', '0.0.14', '0.0.15', '0.0.16')
FIELDS = {'metadata', 'wav_parent', 'package', 'output_dir',
          'contacts', 'ref_rules', 'mode', 'metric', 'threads', 'itn', 'debug'}


def require_file(path):
    if not path.is_file():
        raise ValueError(f'Required file missing: {path}')
    return path


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON key {key!r} in {path}')
            result[key] = value
        return result
    return json.loads(require_file(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def resolve_path(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Paths must be nonempty strings')
    return (base / value).resolve()


def sdk_tools(sdk_dir, version, evaluator=None):
    sdk = sdk_dir.resolve()
    if (sdk / version / 'CMakeLists.txt').is_file():
        sdk /= version
    cmake = require_file(sdk / 'CMakeLists.txt').read_text()
    declared = re.search(r'project\s*\([^)]*\bVERSION\s+([\d.]+)', cmake, re.I)
    if not declared or declared[1] != version:
        raise ValueError(f'SDK source version does not match {version}: {sdk}')
    summary = require_file(sdk / 'cli/summarize_asr_package_eval.py')
    if evaluator:
        binary = require_file(evaluator.resolve())
    else:
        # Supported historical build names; never silently select an ASan or
        # Android build, or guess when multiple host builds exist.
        names = ('build', 'build_' + version.replace('.', '_'),
                 'build_' + version.replace('.', ''))
        candidates = [sdk / name / 'asr_package_eval' for name in names
                      if (sdk / name / 'asr_package_eval').is_file()]
        if len(candidates) != 1:
            raise ValueError(f'Expected one built host evaluator under {sdk}; found '
                             f'{len(candidates)}. Build this SDK first or supply --evaluator.')
        binary = candidates[0]
    if not os.access(binary, os.X_OK):
        raise ValueError(f'Evaluator is not executable: {binary}')
    return sdk, binary, summary


def prepare(args):
    config_path = args.test_set.resolve()
    config = read_json(config_path)
    if not isinstance(config, dict) or set(config) - FIELDS:
        raise ValueError(f'Test-set JSON must contain only: {", ".join(sorted(FIELDS))}')
    base = config_path.parent
    sdk, evaluator, summarizer = sdk_tools(args.sdk_dir, args.version, args.evaluator)
    mode = args.mode or config.get('mode', 'lm')
    if mode not in ('lm', 'greedy', 'both'):
        raise ValueError('mode must be lm, greedy, or both')
    modes = ['greedy', 'lm'] if mode == 'both' else [mode]
    metric = config.get('metric', 'wer')
    if metric not in ('wer', 'cer', 'auto'):
        raise ValueError('metric must be wer, cer, or auto')
    threads = config.get('threads', 1)
    if type(threads) is not int or threads < 1:
        raise ValueError('threads must be a positive integer')
    for key in ('itn', 'debug'):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f'{key} must be a JSON boolean')
    itn = args.itn == 'on' if args.itn is not None else config.get('itn', False)
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    metadata = require_file(resolve_path(base, config['metadata']))
    wav_parent = resolve_path(base, config.get('wav_parent', str(metadata.parent)))
    if not wav_parent.is_dir():
        raise ValueError(f'Audio root missing: {wav_parent}')
    fmt = 'compact' if args.version == '0.0.16' else 'text'
    package = resolve_path(base, config['package'])
    if 'lm' in modes and fmt == 'compact':
        require_file(package / 'lexicon.bin')
    manifest = read_json(package / 'sdk_model.json')
    if not isinstance(manifest, dict):
        raise ValueError('Package manifest must be a JSON object')
    inputs = [config_path, metadata, package, sdk]
    required = ['model_path', 'tokens']
    contacts = None
    if 'lm' in modes:
        expected_type = ('flashlight_compact_lexicon_kenlm' if fmt == 'compact'
                         else 'flashlight_lexicon_kenlm')
        if manifest.get('decoder_type') != expected_type:
            raise ValueError(f'SDK {args.version} LM mode requires {expected_type}; '
                             'supply a matching package (no implicit conversion).')
        required += ['words', 'lexicon', 'lm_search']
        if fmt == 'compact' and manifest.get('lexicon_format') != 'compact_trie_v1':
            raise ValueError('SDK 0.0.16 requires lexicon_format=compact_trie_v1')
        if config.get('contacts') is not None:
            contacts = require_file(resolve_path(base, config['contacts']))
            if not contacts.read_text(encoding='utf-8').strip():
                raise ValueError(f'Contact list is empty: {contacts}')
            inputs.append(contacts)
            if fmt == 'compact':
                required += ['sentencepiece_model']
    if itn:
        required += ['itn_tagger', 'itn_verbalizer']
        if manifest.get('itn_language') != 'en':
            raise ValueError('Enabled ITN requires a configured English ITN package')
    for key in required:
        if key not in manifest:
            raise ValueError(f'Package manifest is missing {key}')
        inputs.append(require_file(resolve_path(package, manifest[key])))
    if 'lm' in modes:
        lms = read_json(resolve_path(package, manifest['lm_search']))
        if not isinstance(lms, dict) or not lms:
            raise ValueError('lm_search must contain at least one LM')
        for filename in lms:
            inputs.append(require_file(resolve_path(package, filename)))
    ref_rules = None
    if args.ref_sub is not None or config.get('ref_rules') is not None:
        ref_rules = require_file(args.ref_sub.resolve() if args.ref_sub is not None
                                 else resolve_path(base, config['ref_rules']))
        inputs.append(ref_rules)
        # Use this SDK's exact rule parser before creating/archiving outputs.
        import runpy
        runpy.run_path(str(summarizer))['load_regex_rules'](ref_rules)
    records = []
    with metadata.open(encoding='utf-8') as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('text'), str):
                raise ValueError(f'{metadata}:{line_no}: text must be a string')
            duration = row.get('duration')
            if type(duration) not in (float, int) or not math.isfinite(duration) or duration <= 0:
                raise ValueError(f'{metadata}:{line_no}: duration must be positive and finite')
            audio = require_file(resolve_path(wav_parent, row.get('audio_filepath')))
            inputs.append(audio)
            records.append(row)
    if not records:
        raise ValueError('Dataset is empty')
    output = (args.output_dir.resolve() if args.output_dir
              else resolve_path(base, config['output_dir']))
    version_dir = output / ('sdk_' + args.version)
    if version_dir.is_symlink():
        raise ValueError(f'Result directory must not be a symlink: {version_dir}')
    for mode in modes:
        work = version_dir / ('work_' + mode)
        if work.is_symlink():
            raise ValueError(f'Result directory must not be a symlink: {work}')
        for path in inputs:
            if path == work or work in path.parents:
                raise ValueError(f'Input is inside a result directory that would be archived: {path}')
    return dict(version=args.version, evaluator=str(evaluator), summarizer=str(summarizer),
                test_set=str(config_path), package=str(package), metadata=str(metadata),
                wav_parent=str(wav_parent), contacts=str(contacts) if contacts else None,
                ref_rules=str(ref_rules) if ref_rules else None, modes=modes,
                metric=metric, threads=threads, itn=itn,
                debug=config.get('debug', False), limit=args.limit,
                rows=min(len(records), args.limit or len(records)),
                output_dir=str(output), version_dir=str(version_dir))


def commands(plan, mode, work):
    evaluator = [plan['evaluator'], '--model_dir', plan['package'],
                 '--metadata', plan['metadata'], '--wav_parent', plan['wav_parent'],
                 '--output_json', str(work / f'output_{mode}.jsonl'),
                 '--decode_mode', mode, '--num_threads', str(plan['threads']),
                 '--enable_itn', str(plan['itn']).lower(),
                 '--debug', str(plan['debug'] and mode == 'lm').lower()]
    if plan['limit']:
        evaluator += ['--limit', str(plan['limit'])]
    if mode == 'lm':
        if plan['contacts']:
            evaluator += ['--contact_names', plan['contacts'], '--skip_unencodable_contacts', 'false']
        if plan['debug']:
            evaluator += ['--debug_log', str(work / 'debug_lm.txt')]
    scorer = [sys.executable, plan['summarizer'], '--input_json', str(work / f'output_{mode}.jsonl'),
              '--summary', str(work / f'summary_{mode}.txt'), '--detail_json',
              str(work / f'detail_{mode}.jsonl'), '--metric', plan['metric']]
    if plan['ref_rules']:
        scorer += ['--ref_regex_rules', plan['ref_rules']]
    return evaluator, scorer


def execute(command, log_path):
    with log_path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace') as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    if code:
        raise RuntimeError(f'Command exited with status {code}; see {log_path}')


@contextmanager
def output_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / '.pipeline.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f'Output is locked: {lock}. Check for an active runner; '
                         'remove the lock only if that process has stopped.') from None
    try:
        (lock / 'owner.json').write_text(json.dumps({'pid': os.getpid()}))
        yield
    finally:
        (lock / 'owner.json').unlink(missing_ok=True)
        lock.rmdir()


def run(plan):
    directory = Path(plan['version_dir'])
    with output_lock(directory):
        for mode in plan['modes']:
            work = directory / ('work_' + mode)
            if work.is_symlink():
                raise ValueError(f'Refusing symlink output: {work}')
            if work.exists():
                backup_root = Path(plan['output_dir']) / 'bak'
                if backup_root.is_symlink():
                    raise ValueError(f'Refusing symlink backup directory: {backup_root}')
                backup_root.mkdir(parents=True, exist_ok=True)
                prefix = f"sdk_{plan['version']}_{datetime.now():%Y%m%d_%H%M%S}_"
                backup = Path(tempfile.mkdtemp(prefix=prefix, dir=backup_root)) / work.name
                work.rename(backup)
                print(f'Previous results preserved: {backup}')
            work.mkdir()
            evaluator, scorer = commands(plan, mode, work)
            record = dict(plan, mode=mode, commands=[evaluator, scorer], status='running')
            record_path = work / 'run_info.json'
            record_path.write_text(json.dumps(record, indent=2) + '\n')
            try:
                execute(evaluator, work / f'eval_{mode}.log')
                execute(scorer, work / f'score_{mode}.log')
                rows = [json.loads(line) for line in (work / f'output_{mode}.jsonl').read_text().splitlines()
                        if line.strip()]
                if len(rows) != plan['rows'] or any(not isinstance(row, dict) or row.get('error') for row in rows):
                    raise RuntimeError(f'Incomplete decoding: expected {plan["rows"]} rows, got {len(rows)}')
                summary = (work / f'summary_{mode}.txt').read_text()
                fields = dict(line.split(': ', 1) for line in summary.splitlines() if ': ' in line)
                if fields.get('failed_count') != '0' or fields.get('sentence_count') != str(plan['rows']):
                    raise RuntimeError('Scorer reports failed or missing utterances')
                record['status'] = 'complete'
                print(f'Completed: {work / ("summary_" + mode + ".txt")}')
            except BaseException as error:
                record.update(status='failed', error=str(error))
                raise
            finally:
                record_path.write_text(json.dumps(record, indent=2) + '\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk-dir', type=Path, required=True, help='SDK collection root or version source directory')
    p.add_argument('--version', '--sdk-version', choices=VERSIONS, required=True)
    p.add_argument('--test-set', type=Path, required=True, help='Dataset settings JSON (paths relative to this file)')
    p.add_argument('--check', action='store_true', help='Validate inputs and show commands without writing results')
    p.add_argument('--limit', type=int, help='Decode only the first N records (smoke test)')
    p.add_argument('--mode', choices=('lm', 'greedy', 'both'), help='Override dataset mode (default lm)')
    p.add_argument('--itn', choices=('on', 'off'), help='Override ITN setting (default off)')
    p.add_argument('--ref-sub', '--ref-sub-file', '--ref-rules', dest='ref_sub', type=Path,
                   help='Override reference substitution/regex rules file; relative to current directory')
    p.add_argument('--output-dir', type=Path, help='Override dataset output directory')
    p.add_argument('--evaluator', type=Path, help='Explicit host binary for a custom/ambiguous build directory')
    return p


def main():
    try:
        args = parser().parse_args()
        plan = prepare(args)
        if args.check:
            print(json.dumps(plan, indent=2))
            for mode in plan['modes']:
                print(json.dumps(commands(plan, mode, Path(plan['version_dir']) / ('work_' + mode))))
        else:
            run(plan)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Interrupted; partial results retained.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())

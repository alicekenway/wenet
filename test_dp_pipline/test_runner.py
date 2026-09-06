import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import run as pipeline

EVALUATOR = '''import json, sys
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
if '--contact_names' in args:
    assert args['--skip_unencodable_contacts'] == 'false'
    assert '--contacts_tsv' not in args
rows = [json.loads(x) for x in Path(args['--metadata']).read_text().splitlines() if x.strip()]
rows = rows[:int(args.get('--limit', len(rows)))]
for row in rows:
    row.update(hyp=row['text'], error='')
Path(args['--output_json']).write_text(''.join(json.dumps(x)+'\\n' for x in rows))
print('fixture evaluator completed')
'''
SCORER = '''import json, sys
from pathlib import Path
def load_regex_rules(path):
    return []
if __name__ == '__main__':
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    rows = Path(args['--input_json']).read_text().splitlines()
    Path(args['--summary']).write_text(f'sentence_count: {len(rows)}\\nfailed_count: 0\\nwer: 0\\n')
    Path(args['--detail_json']).write_text('\\n'.join(rows)+'\\n')
'''


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dp pipeline tests ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sdks = self.root / 'sdk'
        for version in pipeline.VERSIONS:
            sdk = self.sdks / version
            (sdk / 'build').mkdir(parents=True)
            (sdk / 'cli').mkdir()
            (sdk / 'CMakeLists.txt').write_text(f'project(asr_sdk VERSION {version} LANGUAGES CXX)')
            (sdk / 'cli/summarize_asr_package_eval.py').write_text(SCORER)
            executable = sdk / 'build/asr_package_eval'
            executable.write_text(f'#!{sys.executable}\n' + EVALUATOR)
            executable.chmod(0o755)
        for fmt in ('text', 'compact'):
            package = self.root / fmt
            package.mkdir()
            manifest = dict(model_path='model.onnx', tokens='tokens.txt', words='words.txt',
                            lexicon='lexicon.bin' if fmt == 'compact' else 'lexicon.txt',
                            lm_search='lm_search.json', decoder_type='flashlight_' +
                            ('compact_' if fmt == 'compact' else '') + 'lexicon_kenlm')
            if fmt == 'compact':
                manifest.update(lexicon_format='compact_trie_v1', sentencepiece_model='sentencepiece.model')
            for key in ('model_path', 'tokens', 'words', 'lexicon', 'sentencepiece_model'):
                if key in manifest:
                    (package / manifest[key]).touch()
            (package / 'main.bin').touch()
            (package / 'lm_search.json').write_text(json.dumps({'main.bin': {}}))
            (package / 'sdk_model.json').write_text(json.dumps(manifest))
        (self.root / 'sample.wav').touch()
        rows = [dict(audio_filepath='sample.wav', text=f'hello {i}', duration=1.0) for i in range(2)]
        (self.root / 'manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        (self.root / 'names.txt').write_text('Zara Davis\n')
        self.config = dict(metadata='manifest.jsonl', package='compact',
                           contacts='names.txt', output_dir='results')
        self.test_set = self.root / 'test_set.json'
        self.save()

    def save(self):
        self.test_set.write_text(json.dumps(self.config))

    def args(self, version='0.0.16', *extra):
        return pipeline.parser().parse_args(['--sdk-dir', str(self.sdks), '--version', version,
                                             '--test-set', str(self.test_set), *extra])

    def quiet_run(self, plan):
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.run(plan)

    def test_all_version_contracts_and_actual_subprocess_workflow(self):
        for version in pipeline.VERSIONS:
            with self.subTest(version=version):
                expected = 'compact' if version == '0.0.16' else 'text'
                self.config['package'] = expected
                self.save()
                plan = pipeline.prepare(self.args(version))
                self.assertEqual(plan['package'], str(self.root / expected))
                self.quiet_run(plan)
                work = Path(plan['version_dir']) / 'work_lm'
                info = json.loads((work / 'run_info.json').read_text())
                self.assertEqual(info['status'], 'complete')
                self.assertIn('--contact_names', info['commands'][0])
                self.assertNotIn('--contacts_tsv', info['commands'][0])
                self.assertEqual(len((work / 'output_lm.jsonl').read_text().splitlines()), 2)

    def test_check_preparation_does_not_write_results(self):
        plan = pipeline.prepare(self.args('0.0.16', '--check'))
        self.assertEqual(plan['rows'], 2)
        self.assertFalse((self.root / 'results').exists())

    def test_itn_off_overrides_dataset_on(self):
        self.config['itn'] = True
        self.save()
        plan = pipeline.prepare(self.args('0.0.16', '--itn', 'off'))
        self.assertFalse(plan['itn'])
        command = pipeline.commands(plan, 'lm', self.root)[0]
        self.assertEqual(command[command.index('--enable_itn') + 1], 'false')

    def test_itn_on_requires_fsts_and_is_forwarded(self):
        with self.assertRaisesRegex(ValueError, 'English ITN'):
            pipeline.prepare(self.args('0.0.16', '--itn', 'on'))
        package = self.root / 'compact'
        manifest = json.loads((package / 'sdk_model.json').read_text())
        manifest.update(itn_language='en', itn_tagger='tagger.fst', itn_verbalizer='verbalizer.fst')
        (package / 'sdk_model.json').write_text(json.dumps(manifest))
        (package / 'tagger.fst').touch()
        (package / 'verbalizer.fst').touch()
        plan = pipeline.prepare(self.args('0.0.16', '--itn', 'on'))
        self.assertTrue(plan['itn'])
        command = pipeline.commands(plan, 'lm', self.root)[0]
        self.assertEqual(command[command.index('--enable_itn') + 1], 'true')

    def test_reference_substitution_cli_overrides_config(self):
        self.config['ref_rules'] = 'missing-old-rules.tsv'
        self.save()
        rules = self.root / 'new rules.tsv'
        rules.write_text('hello\thi\n')
        plan = pipeline.prepare(self.args('0.0.16', '--ref-sub', str(rules)))
        self.assertEqual(plan['ref_rules'], str(rules))
        scorer = pipeline.commands(plan, 'lm', self.root)[1]
        self.assertEqual(scorer[scorer.index('--ref_regex_rules') + 1], str(rules))
        with self.assertRaisesRegex(ValueError, 'Required file missing'):
            pipeline.prepare(self.args('0.0.16', '--ref-sub', str(self.root / 'absent.tsv')))

    def test_backup_and_limit(self):
        plan = pipeline.prepare(self.args('0.0.16', '--limit', '1'))
        self.quiet_run(plan)
        work = Path(plan['version_dir']) / 'work_lm'
        (work / 'preserve.txt').write_text('old result')
        self.quiet_run(plan)
        backups = list((self.root / 'results/bak').glob('*/work_lm/preserve.txt'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), 'old result')
        self.assertEqual(len((work / 'output_lm.jsonl').read_text().splitlines()), 1)

    def test_modes_and_contacts_disabled(self):
        self.config['contacts'] = None
        self.config['package'] = 'text'
        self.save()
        plan = pipeline.prepare(self.args('0.0.14', '--mode', 'both'))
        self.quiet_run(plan)
        for mode in ('lm', 'greedy'):
            self.assertNotIn('--contact_names', pipeline.commands(plan, mode, self.root)[0])
            self.assertTrue((Path(plan['version_dir']) / f'work_{mode}/summary_{mode}.txt').is_file())

    def test_legacy_build_name_and_version_directory(self):
        self.config['package'] = 'text'
        self.save()
        sdk = self.sdks / '0.0.14'
        (sdk / 'build').rename(sdk / 'build_0_0_14')
        args = self.args('0.0.14')
        args.sdk_dir = sdk
        self.assertIn('build_0_0_14', pipeline.prepare(args)['evaluator'])

    def test_missing_or_ambiguous_binary(self):
        sdk = self.sdks / '0.0.16'
        (sdk / 'build_0_0_16').mkdir()
        second = sdk / 'build_0_0_16/asr_package_eval'
        second.write_text('')
        with self.assertRaisesRegex(ValueError, 'found 2'):
            pipeline.prepare(self.args())
        second.unlink()
        (sdk / 'build/asr_package_eval').unlink()
        with self.assertRaisesRegex(ValueError, 'Build this SDK first'):
            pipeline.prepare(self.args())

    def test_wrong_source_version(self):
        args = self.args()
        args.sdk_dir = self.sdks / '0.0.13'
        with self.assertRaisesRegex(ValueError, 'version does not match'):
            pipeline.prepare(args)

    def test_wrong_package_format(self):
        self.config['package'] = 'text'
        (self.root / 'text/lexicon.bin').touch()
        self.save()
        with self.assertRaisesRegex(ValueError, 'matching package'):
            pipeline.prepare(self.args())

    def test_bin_required_only_for_sdk16_lm(self):
        (self.root / 'compact/lexicon.bin').unlink()
        with self.assertRaisesRegex(ValueError, 'lexicon.bin'):
            pipeline.prepare(self.args())
        self.assertEqual(pipeline.prepare(self.args('0.0.16', '--mode', 'greedy'))['rows'], 2)
        self.config['package'] = 'text'
        self.save()
        self.assertFalse((self.root / 'text/lexicon.bin').exists())
        for version in ('0.0.13', '0.0.14', '0.0.15'):
            self.assertEqual(pipeline.prepare(self.args(version))['rows'], 2)

    def test_missing_tokenizer_only_blocks_automatic_lm(self):
        (self.root / 'compact/sentencepiece.model').unlink()
        with self.assertRaisesRegex(ValueError, 'sentencepiece.model'):
            pipeline.prepare(self.args())
        self.assertEqual(pipeline.prepare(self.args('0.0.16', '--mode', 'greedy'))['rows'], 2)
        self.config['contacts'] = None
        self.save()
        self.assertEqual(pipeline.prepare(self.args())['rows'], 2)

    def test_missing_audio_keeps_existing_results(self):
        work = self.root / 'results/sdk_0.0.16/work_lm'
        work.mkdir(parents=True)
        (work / 'old.txt').write_text('keep')
        (self.root / 'sample.wav').unlink()
        with self.assertRaisesRegex(ValueError, 'sample.wav'):
            pipeline.prepare(self.args())
        self.assertEqual((work / 'old.txt').read_text(), 'keep')

    def test_bad_config_and_duration(self):
        self.config['threadz'] = 2
        self.save()
        with self.assertRaises(ValueError):
            pipeline.prepare(self.args())
        del self.config['threadz']
        self.save()
        (self.root / 'manifest.jsonl').write_text(json.dumps(dict(
            audio_filepath='sample.wav', text='hello', duration=-1))+'\n')
        with self.assertRaisesRegex(ValueError, 'duration'):
            pipeline.prepare(self.args())

    def test_duplicate_json_keys(self):
        self.test_set.write_text('{"metadata":"a", "metadata":"b"}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            pipeline.prepare(self.args())

    def test_inputs_in_output_rejected(self):
        work = self.root / 'results/sdk_0.0.16/work_lm'
        work.mkdir(parents=True)
        (self.root / 'manifest.jsonl').rename(work / 'manifest.jsonl')
        self.config.update(metadata=str(work / 'manifest.jsonl'), wav_parent=str(self.root))
        self.save()
        with self.assertRaisesRegex(ValueError, 'Input is inside'):
            pipeline.prepare(self.args())

    def test_symlink_version_output_rejected(self):
        (self.root / 'results').mkdir()
        (self.root / 'outside').mkdir()
        (self.root / 'results/sdk_0.0.16').symlink_to(self.root / 'outside', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'must not be a symlink'):
            pipeline.prepare(self.args())

    def test_concurrent_output_lock(self):
        plan = pipeline.prepare(self.args())
        directory = Path(plan['version_dir'])
        with pipeline.output_lock(directory):
            with self.assertRaisesRegex(ValueError, 'locked'):
                pipeline.run(plan)
        self.assertFalse((directory / '.pipeline.lock').exists())

    def test_subprocess_failure_preserves_status_and_unlocks(self):
        plan = pipeline.prepare(self.args())
        Path(plan['evaluator']).write_text(f'#!{sys.executable}\nimport sys; sys.exit(3)\n')
        with self.assertRaisesRegex(RuntimeError, 'status 3'):
            self.quiet_run(plan)
        directory = Path(plan['version_dir'])
        info = json.loads((directory / 'work_lm/run_info.json').read_text())
        self.assertEqual(info['status'], 'failed')
        self.assertFalse((directory / '.pipeline.lock').exists())


if __name__ == '__main__':
    unittest.main()

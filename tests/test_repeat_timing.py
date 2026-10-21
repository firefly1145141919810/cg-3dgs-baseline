"""CPU-only timing protocol regression tests."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import repeat_timing as timing
from run import write_json


class RepeatTimingTests(unittest.TestCase):
    def test_mock_run_resume_and_log_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [{'scene': s, 'budget': 50, 'views': 1,
                        'models': {m: {'count': 4, 'config': 'fixture.json', 'ply': 'fixture.ply',
                                       'config_sha256': 'digest', 'ply_sha256': 'digest'}
                                   for m in timing.METHODS}} for s in ('train', 'truck')]
            def execute(cmd, out, stage):
                self.assertTrue(out.is_dir())
                target = Path(cmd[cmd.index('--result') + 1])
                write_json(target, {'samples_ms': [2.]*3, 'mean_ms': 2., 'fps_from_mean_ms': 500.,
                                    'gaussian_count': 4, 'test_views': 1, 'repeats': 3, 'warmup': 20,
                                    'gpu': 'fixture', 'image_sizes': [[4, 3]], 'peak_torch_allocated_bytes': 100})
            with patch.object(timing, 'ROOT', root), \
                 patch.object(timing, 'sha256', return_value='digest'), \
                 patch.object(timing, 'collect_sources', return_value=sources), \
                 patch('doctor.diagnose', return_value={'errors': []}), \
                 patch.object(sys, 'argv', ['repeat_timing.py', '--name', 'fixture']), \
                 patch.object(timing, 'execute', side_effect=execute) as runner:
                timing.main()
                self.assertEqual(runner.call_count, 18)
                timing.main()
                self.assertEqual(runner.call_count, 18)

    def test_sources_include_each_budget_and_historical_names(self):
        def info(old):
            names = ['identity100_recovery', 'opacity50_recovery', 'contribution50_recovery'] if old else timing.METHODS
            return {'formal': True, 'steps': 3000, 'source_iteration': 30000,
                    'test_views': 38, 'variants': {v: {'count': 4} for v in names}}
        with patch.object(timing.phase4, 'verify', return_value=info(True)), \
             patch.object(timing.phase5, 'verify', return_value=info(False)), \
             patch.object(timing.phase4, 'check_finished'), \
             patch.object(timing.phase5, 'check_finished'), \
             patch.object(timing, 'sha256', return_value='digest'), \
             patch.object(timing, 'read_json', return_value={
                 'complete': True, 'manifest_sha256': 'digest', 'artifact_sha256': {}}):
            sources = timing.collect_sources()
        self.assertEqual(len(sources), 6)
        self.assertEqual({(s['scene'], s['budget']) for s in sources},
                         {(s, b) for s in ('train', 'truck') for b in (70, 50, 30)})
        fifty = next(s for s in sources if s['budget'] == 50)
        self.assertIn('opacity50_recovery', fifty['models']['opacity_recovery']['config'])

    def test_balanced_order(self):
        orders = [timing.order(r) for r in range(3)]
        for pos in range(3):
            self.assertEqual({o[pos] for o in orders}, set(timing.METHODS))
        self.assertEqual(timing.order(0), timing.order(3))

    def test_sample_validation(self):
        b = dict(samples_ms=[2., 2., 2.], gaussian_count=4, test_views=1,
                 warmup=20, repeats=3, mean_ms=2., fps_from_mean_ms=500.)
        timing.check_sample(b, 4, 1)
        for bad in (dict(b, mean_ms=3), dict(b, samples_ms=[float('nan')]*3),
                    dict(b, repeats=2), dict(b, gaussian_count=5), dict(b, fps_from_mean_ms=499)):
            with self.assertRaises(ValueError): timing.check_sample(bad, 4, 1)

    def test_paired_summary_and_missing_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            sources = []
            for scene in ('train', 'truck'):
                sources.append({'scene': scene, 'budget': 50, 'views': 1,
                                'models': {m: {'count': 4} for m in timing.METHODS}})
                for r in range(3):
                    for i, m in enumerate(timing.METHODS):
                        ms = (r + 1) * (4 if i == 0 else 2)
                        write_json(dest / f'{scene}-50-r{r+1}-{m}.json', {
                            'samples_ms': [ms]*3, 'mean_ms': ms, 'fps_from_mean_ms': 1000/ms,
                            'gaussian_count': 4, 'test_views': 1, 'repeats': 3, 'warmup': 20,
                            'gpu': 'fixture', 'image_sizes': [[4, 3]], 'peak_torch_allocated_bytes': 100})
            manifest = {'sources': sources, 'rounds': 3}
            write_json(dest / 'protocol.json', manifest)
            result = timing.summarize(manifest, dest)
            self.assertEqual(len(result['sample_sha256']), 18)
            self.assertEqual(result['rows'][1]['paired_speedup_across_rounds']['median'], 2)
            self.assertEqual(result['rows'][1]['mean_ms_across_rounds'], {'median': 4, 'min': 2, 'max': 6})
            (dest / 'train-50-r1-opacity_recovery.json').unlink()
            with self.assertRaises(FileNotFoundError): timing.summarize(manifest, dest)

    def test_worker_keeps_original_measurement_algorithm(self):
        folder = Path(timing.__file__).parent
        old = ast.parse((folder / 'benchmark.py').read_text(encoding='utf-8'))
        new = ast.parse((folder / 'benchmark_repeat_worker.py').read_text(encoding='utf-8'))
        # Compare the whole CUDA measurement block, not just function names.
        blocks = lambda tree: [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.With)]
        self.assertEqual(blocks(old), blocks(new))
        source = (folder / 'benchmark_repeat_worker.py').read_text(encoding='utf-8')
        self.assertIn('write_json(args.result,result)', source)
        self.assertNotIn("write_json(out/'benchmark.json'", source)


if __name__ == '__main__':
    unittest.main()

"""CPU-only orchestration tests; subprocess training is never launched."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import truck_suite as suite


class TruckSuiteTests(unittest.TestCase):
    def test_frozen_commands_and_separate_paths(self):
        c, paths = suite.paths('configs/truck.json')
        self.assertEqual(c['scene'], 'truck')
        self.assertEqual(c['iterations'], 30000)
        for path in paths.values():
            self.assertIn('truck', path.name)
            self.assertNotIn('train_', path.name)
        cmds = {s: suite.commands('configs/truck.json', s) for s in suite.STAGES}
        self.assertEqual(len(cmds['baseline']), 4)
        self.assertIn('--baseline-config', cmds['opacity'][0])
        self.assertEqual(cmds['contribution'][0][-2:], ['--views', '0'])
        self.assertEqual(cmds['recovery'][0][-4:], ['--steps', '3000', '--seed', '0'])
        self.assertEqual(cmds['recovery'][0][cmds['recovery'][0].index('--parent')+1],
                         str(paths['contribution']))
        self.assertIn('--resume', suite.commands('configs/truck.json', 'baseline', 'local.pth')[0])

    def test_reject_protocol_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = json.loads((suite.ROOT/'configs/truck.json').read_text())
            cfg['iterations'] = 33000
            cfg['eval_iterations'] = [33000]
            f = Path(temp)/'config.json'; f.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                suite.paths(str(f))

    def test_verified_stage_is_not_rerun(self):
        with patch.object(suite, 'ready', return_value=True), patch.object(suite, 'run_command') as run:
            suite.run_stage('configs/truck.json', 'recovery')
            run.assert_not_called()

    def test_partial_recovery_reuses_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = {s: Path(temp)/s for s in suite.STAGES}
            paths['recovery'].mkdir()
            (paths['recovery']/'experiment.json').write_text('{}')
            with patch.object(suite, 'paths', return_value=({}, paths)), \
                 patch.object(suite, 'ready', side_effect=[False, True, True]), \
                 patch.object(suite, 'commands', return_value=[['prepare'], ['train'], ['evaluate']]), \
                 patch.object(suite, 'run_command') as run:
                suite.run_stage('fixture', 'recovery')
                self.assertEqual([c.args[0] for c in run.call_args_list], [['train'], ['evaluate']])

    def test_partial_baseline_requires_explicit_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = {s: Path(temp)/s for s in suite.STAGES}
            paths['baseline'].mkdir()
            (paths['baseline']/'interrupted.log').write_text('fixture')
            with patch.object(suite, 'paths', return_value=({}, paths)), \
                 patch.object(suite, 'ready', return_value=False), \
                 patch.object(suite, 'commands', return_value=[['train'], ['render'], ['metrics'], ['benchmark']]), \
                 patch.object(suite, 'run_command') as run:
                with self.assertRaisesRegex(ValueError, 'explicitly pass --checkpoint'):
                    suite.run_stage('fixture', 'baseline')
                run.assert_not_called()

    def test_incomplete_upstream_does_not_invalidate_descendants(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = {s: Path(temp)/s for s in suite.STAGES}
            paths['contribution'].mkdir()
            (paths['contribution']/'experiment.json').write_text('{}')
            with patch.object(suite, 'paths', return_value=({}, paths)), \
                 patch.object(suite, 'ready', return_value=False), \
                 patch.object(suite, 'run_command') as run:
                with self.assertRaisesRegex(ValueError, 'derived experiments'):
                    suite.run_stage('fixture', 'opacity')
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()

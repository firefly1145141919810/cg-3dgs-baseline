"""CPU contract tests: no actual torch/CUDA training."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import recovery_phase4 as p2
from recovery_train import camera_sequence, optimization, check_plan
from dataset import sha256
from run import write_json


class RecoveryPhase4Tests(unittest.TestCase):
    def test_fixed_schedule_and_budget(self):
        seq = camera_sequence(7, 20, 0)
        self.assertEqual(seq, camera_sequence(7, 20, 0))
        self.assertEqual(sorted(seq[:7]), list(range(7)))
        self.assertEqual(seq[:12], camera_sequence(7, 12, 0))
        self.assertNotEqual(seq, camera_sequence(7, 20, 1))
        for steps in (0, -1, 3001, True):
            with self.assertRaises(ValueError): check_plan(steps, 0)
        with self.assertRaises(ValueError): check_plan(20, -1)
        opt = optimization(3000)
        self.assertEqual(opt.position_lr_init, opt.position_lr_final)
        self.assertEqual(opt.lambda_dssim, .2)

    def test_no_topology_mutation_or_test_camera_use_in_training(self):
        source = Path(p2.__file__).with_name('recovery_train.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & {'densify_and_prune', 'reset_opacity', 'oneupSHdegree',
                                 'prune_points', 'add_densification_stats', 'getTestCameras'})
        self.assertIn('step', calls)
        self.assertIn('save_ply', calls)
        # Regression for load_ply leaving spatial_lr_scale=0.
        self.assertIn('gaussians.spatial_lr_scale = float(scene.cameras_extent)', source)
        self.assertLess(source.index('gaussians.optimizer.step()'), source.index('gaussians.save_ply'))

    def test_mocked_six_arm_lifecycle_and_tamper_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); parent = root / 'phase1'; parent.mkdir()
            baseline = root / 'baseline'; baseline.mkdir()
            c = dict(scene='fixture', source=str(root / 'data'), output=str(baseline), images='images',
                     resolution=-1, iterations=30000, eval_iterations=[30000],
                     checkpoint_iterations=[30000], data_device='cuda', white_background=False, port=6009)
            metric = dict(PSNR=20., SSIM=.8, LPIPS=.2)
            pi = {'baseline_config': c, 'test_views': 1, 'baseline_metrics': metric,
                  'train_views': 2, 'variants': {'identity100': {'kept_count': 8}, 'opacity50': {'kept_count': 4}, 'contribution50': {'kept_count': 4}}}
            write_json(parent / 'experiment.json', pi)
            write_json(parent / 'evaluation.json', {'complete': True})
            write_json(parent / 'dataset.json', {'train': ['a.png', 'b.png']})
            for v in ('identity100', 'opacity50', 'contribution50'):
                ply = p2.ply_path(parent / v, 30000); ply.parent.mkdir(parents=True)
                ply.write_bytes(b'synthetic-copy-only-' + v.encode())
                images = parent / v / 'test/ours_30000/renders'; images.mkdir(parents=True)
                Image.new('RGB', (4, 3), (23, 45, 67)).save(images / '00000.png')
            gt = baseline / 'test/ours_30000/gt'; gt.mkdir(parents=True)
            shutil.copy2(parent / 'identity100/test/ours_30000/renders/00000.png', gt / '00000.png')
            experiment = root / 'recovery'
            with patch.object(p2, 'verify_parent', return_value=pi), \
                 patch.object(p2, 'code_hashes', return_value={'test': 'hash'}):
                p2.prepare(parent, experiment, 2, 0)
                info = p2.verify(experiment)
                self.assertFalse(info['formal'])
                with self.assertRaises(FileExistsError): p2.prepare(parent, experiment, 2, 0)
                with self.assertRaises(FileNotFoundError): p2.evaluate(experiment)
                for v in p2.TRAIN_VARIANTS:
                    config = p2.model_config(info, experiment, v)
                    self.assertEqual(config['iterations'], 30002)
                    final = p2.ply_path(experiment / v, config['iterations'])
                    final.parent.mkdir(parents=True)
                    final.write_bytes(b'mock-updated-model')
                    write_json(experiment / v / 'training.json', {
                        'complete': True, 'steps': 2, 'count': info['variants'][v]['count'],
                        'source_ply_sha256': info['variants'][v]['source_sha256'],
                        'final_ply_sha256': sha256(final),
                        'manifest_sha256': sha256(experiment / 'experiment.json'),
                        'camera_sequence_sha256': 'same', 'trace': [{'step': 1, 'loss': .2}, {'step': 2, 'loss': .1}]})
                    write_json(experiment / v / 'training_environment.json', {})
                def fake_execute(cmd, model, stage):
                    config = p2.model_config(info, experiment, model.name)
                    key = f'ours_{config["iterations"]}'
                    if stage == 'render':
                        for kind in ('gt', 'renders'):
                            folder = model / 'test' / key / kind; folder.mkdir(parents=True)
                            shutil.copy2(gt / '00000.png', folder / '00000.png')
                    elif stage == 'metrics':
                        write_json(model / 'results.json', {key: metric})
                        write_json(model / 'per_view.json', {key: {k: {'00000.png': x} for k, x in metric.items()}})
                    elif stage == 'benchmark':
                        write_json(model / 'benchmark.json', {
                            'gaussian_count': info['variants'][model.name]['count'], 'test_views': 1,
                            'repeats': 3, 'samples_ms': [2., 2., 2.], 'mean_ms': 2.,
                            'fps_from_mean_ms': 500., 'peak_torch_allocated_bytes': 1024})
                with patch('doctor.diagnose', return_value={'errors': []}), \
                     patch.object(p2, 'execute', side_effect=fake_execute):
                    p2.evaluate(experiment)
                with patch.object(p2, 'ROOT', root):
                    p2.export(experiment, 'mock-result')
                    result = root / 'results/mock-result'
                    self.assertEqual(len(list(result.glob('*/renders/00000.png'))), 6)
                    self.assertFalse(list(result.rglob('*.ply')))
                    self.assertFalse(list(result.rglob('*.pth')))
                    with self.assertRaises(FileExistsError): p2.export(experiment, 'mock-result')
                    (experiment / 'opacity50_recovery/training.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError, 'artifact changed'):
                        p2.export(experiment, 'tampered')
                (experiment / 'opacity50/cfg_args').write_text('Namespace()')
                with self.assertRaisesRegex(ValueError, 'Render configuration'):
                    p2.verify(experiment)


    def test_training_algorithm_matches_validated_phase2(self):
        folder = Path(p2.__file__).parent
        old = ast.parse((folder / 'recovery_train.py').read_text(encoding='utf-8'))
        new = ast.parse((folder / 'recovery_train_phase4.py').read_text(encoding='utf-8'))
        for node in ast.walk(new):
            if isinstance(node, ast.ImportFrom) and node.module == 'recovery_phase4':
                node.module = 'recovery_phase2'
        old_functions = [n for n in old.body if isinstance(n, ast.FunctionDef)]
        new_functions = [n for n in new.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual([ast.dump(n) for n in old_functions], [ast.dump(n) for n in new_functions])

    def test_parent_requires_formal_complete_contribution_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            write_json(parent / 'evaluation.json', {'complete': False})
            with patch.object(p2.phase3, 'prepared', return_value=({'formal': False}, {}, {})):
                with self.assertRaisesRegex(ValueError, 'full-train-view'):
                    p2.verify_parent(parent)
            with patch.object(p2.phase3, 'prepared', return_value=({'formal': True}, {}, {})):
                with self.assertRaisesRegex(ValueError, 'completed evaluation'):
                    p2.verify_parent(parent)


if __name__ == '__main__':
    unittest.main()

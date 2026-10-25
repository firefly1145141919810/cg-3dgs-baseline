"""CPU contract tests: no actual torch/CUDA training."""
import ast
import numpy as np
from plyfile import PlyData, PlyElement
from test_compression import vertices
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import budget_phase5 as p2
from recovery_train import camera_sequence, optimization, check_plan
from dataset import sha256
from run import write_json


class BudgetPhase5Tests(unittest.TestCase):
    def test_predeclared_budgets_and_nested_selection(self):
        for invalid in (0, 1, True, .4, float('nan')):
            with self.assertRaises(ValueError):
                p2.check_keep(invalid)
        scores = np.array([1., 1., 5., 8., 3., 2., 9., 0.])
        subsets = []
        for keep, count in ((.3, 2), (.5, 4), (.7, 5)):
            p2.check_keep(keep)
            selected = p2.select_indices(scores, keep)
            self.assertEqual(len(selected), count)
            self.assertTrue(np.all(np.diff(selected) > 0))
            subsets.append(set(selected))
        self.assertTrue(subsets[0] < subsets[1] < subsets[2])

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
                  'train_views': 2, 'variants': {'identity100': {'kept_count': 8}, 'opacity': {'kept_count': 4}, 'contribution': {'kept_count': 4}}}
            write_json(parent / 'experiment.json', pi)
            write_json(parent / 'evaluation.json', {'complete': True})
            write_json(parent / 'dataset.json', {'train': ['a.png', 'b.png']})
            for v in ('identity100', 'opacity', 'contribution'):
                ply = p2.ply_path(parent / v, 30000); ply.parent.mkdir(parents=True)
                PlyData([PlyElement.describe(vertices(), 'vertex')], text=False, byte_order='<').write(ply)
                images = parent / v / 'test/ours_30000/renders'; images.mkdir(parents=True)
                Image.new('RGB', (4, 3), (23, 45, 67)).save(images / '00000.png')
            gt = baseline / 'test/ours_30000/gt'; gt.mkdir(parents=True)
            shutil.copy2(parent / 'identity100/test/ours_30000/renders/00000.png', gt / '00000.png')
            np.save(parent / 'scores.npy', np.arange(8, dtype=float))
            experiment = root / 'recovery'
            with patch.object(p2, 'verify_parent', return_value=pi), \
                 patch.object(p2, 'code_hashes', return_value={'test': 'hash'}), \
                 patch.object(p2.phase3, 'prepared', return_value=({}, pi, {})), \
                 patch.object(p2.phase3, 'scoring', return_value=np.arange(8, dtype=float)):
                p2.prepare(parent, experiment, 2, 0, .3)
                info = p2.verify(experiment)
                self.assertFalse(info['formal'])
                self.assertEqual(info['variants']['opacity']['count'], 2)
                np.testing.assert_array_equal(np.load(experiment / 'opacity/kept_indices.npy'), [3, 6])
                np.testing.assert_array_equal(np.load(experiment / 'contribution/kept_indices.npy'), [6, 7])
                for v, indices in (('opacity', [3, 6]), ('contribution', [6, 7])):
                    saved = p2.read_vertices(p2.ply_path(experiment / v, 30000)).copy()
                    for field in saved.dtype.names:
                        np.testing.assert_array_equal(saved[field], vertices()[field][indices])
                for v in ('opacity', 'contribution', 'identity100'):
                    self.assertEqual(sha256(p2.ply_path(experiment / v, 30000)),
                                     sha256(p2.ply_path(experiment / (v + '_recovery'), 30000)))
                with self.assertRaises(FileExistsError): p2.prepare(parent, experiment, 2, 0, .3)
                with self.assertRaises(FileNotFoundError): p2.evaluate(experiment)
                for v in p2.TRAIN_VARIANTS:
                    config = p2.model_config(info, experiment, v)
                    self.assertEqual(config['iterations'], 30002)
                    final = p2.ply_path(experiment / v, config['iterations'])
                    final.parent.mkdir(parents=True)
                    final.write_bytes(('ply\nformat binary_little_endian 1.0\nelement vertex ' + str(info['variants'][v]['count']) + '\nend_header\n').encode())
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
                            'config_resolution': -1, 'image_sizes': [[4, 3]], 'repeats': 3, 'samples_ms': [2., 2., 2.], 'mean_ms': 2.,
                            'fps_from_mean_ms': 500., 'peak_torch_allocated_bytes': 1024})
                with patch('doctor.diagnose', return_value={'errors': []}), \
                     patch.object(p2, 'execute', side_effect=fake_execute):
                    p2.evaluate(experiment)
                with patch.object(p2, 'ROOT', root):
                    saved_evaluation = json.loads((experiment / 'evaluation.json').read_text())
                    for field in ('mean_ms', 'fps_from_mean_ms', 'peak_torch_allocated_bytes',
                                  'gaussian_count', 'ply_bytes', 'recovery_steps', 'delta_PSNR'):
                        changed = json.loads(json.dumps(saved_evaluation))
                        changed['rows'][0][field] += 999
                        write_json(experiment / 'evaluation.json', changed)
                        with self.assertRaises(ValueError): p2.export(experiment, 'bad-summary')
                        self.assertFalse((root / 'results/bad-summary').exists())
                    write_json(experiment / 'evaluation.json', saved_evaluation)
                    with patch.object(p2.shutil, 'copy2', side_effect=OSError('copy interrupted')):
                        with self.assertRaises(OSError): p2.export(experiment, 'mock-result')
                    self.assertFalse((root / 'results/mock-result').exists())
                    p2.export(experiment, 'mock-result')
                    result = root / 'results/mock-result'
                    self.assertEqual(len(list(result.glob('*/renders/00000.png'))), 6)
                    self.assertFalse(list(result.rglob('*.ply')))
                    self.assertFalse(list(result.rglob('*.pth')))
                    with self.assertRaises(FileExistsError): p2.export(experiment, 'mock-result')
                    (experiment / 'opacity_recovery/training.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError, 'artifact changed'):
                        p2.export(experiment, 'tampered')
                index_file = experiment / 'opacity/kept_indices.npy'
                original_indices = index_file.read_bytes()
                np.save(index_file, [0, 1])
                with self.assertRaisesRegex(ValueError, 'Selected indices changed'):
                    p2.verify(experiment)
                index_file.write_bytes(original_indices)
                (experiment / 'opacity/cfg_args').write_text('Namespace()')
                with self.assertRaisesRegex(ValueError, 'Render configuration'):
                    p2.verify(experiment)


    def test_training_algorithm_matches_validated_phase2(self):
        folder = Path(p2.__file__).parent
        old = ast.parse((folder / 'recovery_train.py').read_text(encoding='utf-8'))
        new = ast.parse((folder / 'recovery_train_phase5.py').read_text(encoding='utf-8'))
        for node in ast.walk(new):
            if isinstance(node, ast.ImportFrom) and node.module == 'budget_phase5':
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

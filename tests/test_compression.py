"""Synthetic CPU tests only; never load torch or launch the renderer."""
import json
import argparse
import importlib.util
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prune_ply import read_vertices, select_indices, write_pruned
from compression_phase1 import compare_images, checked_metrics, evaluate, prepare, export, verify_inputs, render_config
from run import UPSTREAM, command


def vertices(n=8):
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz'] +
             [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)] +
             ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    v = np.zeros(n, dtype=[(k, 'f4') for k in names])
    for i, k in enumerate(names): v[k] = np.arange(n) + i / 64
    v['opacity'] = np.array([2, -3, 2, 7, 1, 0, 5, -1])[:n]
    return v


class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.ply'
        PlyData([PlyElement.describe(vertices(), 'vertex')]).write(self.source)

    def test_upstream_render_argument_merge_preserves_false_background(self):
        # Exercise the real pinned upstream parser (stdlib only), not a mock of
        # rendering. This catches missing False values in its sentinel merge.
        spec = importlib.util.spec_from_file_location(
            'upstream_arguments_test', UPSTREAM / 'arguments' / '__init__.py')
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        for white in (False, True):
            with self.subTest(white_background=white):
                c = {'source': str(self.root / "scene with 'quote'"),
                     'output': str(self.root), 'images': 'images',
                     'resolution': -1, 'white_background': white,
                     'data_device': 'cuda', 'iterations': 30000}
                (self.root / 'cfg_args').write_text(repr(render_config(c)), encoding='utf-8')
                parser = argparse.ArgumentParser()
                model = upstream.ModelParams(parser, sentinel=True)
                upstream.PipelineParams(parser)
                parser.add_argument('--iteration', type=int, default=-1)
                parser.add_argument('--skip_train', action='store_true')
                with patch.object(sys, 'argv', command('render', c)[1:]):
                    merged = upstream.get_combined_args(parser)
                actual = model.extract(merged)
                self.assertEqual(vars(actual), vars(render_config(c)))
                self.assertIs(actual.white_background, white)

    def test_identity_preserves_every_field_and_order(self):
        indices, report = write_pruned(self.source, self.root / 'identity.ply', 1)
        self.assertEqual(indices.tolist(), list(range(8)))
        self.assertTrue(report['all_selected_fields_bitwise_equal'])
        self.assertEqual(read_vertices(self.source).tobytes(), read_vertices(self.root / 'identity.ply').tobytes())

    def test_top_half_ties_and_original_order(self):
        indices, _ = write_pruned(self.source, self.root / 'half.ply', .5)
        self.assertEqual(indices.tolist(), [0, 2, 3, 6])
        self.assertEqual(select_indices(np.array([100, 101, 102, 103]), .5).tolist(), [2, 3])
        self.assertEqual(select_indices(np.ones(5), .5).tolist(), [0, 1])
        self.assertEqual(select_indices(np.ones(3), .1).tolist(), [0])

    def test_invalid_input_and_refuse_overwrite(self):
        for fraction in (0, -1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): select_indices(np.zeros(2), fraction)
        with self.assertRaises(ValueError): select_indices(np.array([np.nan]), .5)
        with self.assertRaises(FileExistsError): write_pruned(self.source, self.source, 1)
        generic = self.root / 'generic.ply'
        PlyData([PlyElement.describe(np.zeros(2, dtype=[('x', 'f4')]), 'vertex')]).write(generic)
        with self.assertRaises(ValueError): read_vertices(generic)
        v = vertices()
        v['f_rest_44'][0] = np.inf
        PlyData([PlyElement.describe(v, 'vertex')]).write(generic)
        with self.assertRaises(ValueError): read_vertices(generic)

    def test_pixel_check_detects_change_and_missing_view(self):
        a, b = self.root / 'a', self.root / 'b'
        a.mkdir(); b.mkdir()
        arr = np.zeros((3, 4, 3), dtype=np.uint8)
        Image.fromarray(arr).save(a / '00000.png')
        Image.fromarray(arr).save(b / '00000.png')
        self.assertTrue(compare_images(a, b, 1)['passed'])
        arr[0, 0, 1] = 1
        Image.fromarray(arr).save(b / '00000.png')
        result = compare_images(a, b, 1)
        self.assertFalse(result['passed'])
        self.assertEqual(result['changed_channels'], 1)
        with self.assertRaises(ValueError): compare_images(a, b, 2)

    def test_metrics_validation(self):
        mean = {'PSNR': 20., 'SSIM': .8, 'LPIPS': .2}
        per = {k: {'00000.png': v} for k, v in mean.items()}
        (self.root / 'results.json').write_text(json.dumps({'ours_30000': mean}))
        (self.root / 'per_view.json').write_text(json.dumps({'ours_30000': per}))
        self.assertEqual(checked_metrics(self.root, 30000, 1), mean)
        mean['PSNR'] = 30.
        (self.root / 'results.json').write_text(json.dumps({'ours_30000': mean}))
        with self.assertRaises(ValueError): checked_metrics(self.root, 30000, 1)

    def test_failed_identity_blocks_half_evaluation(self):
        # Mock external CUDA calls, not the gate under test.
        info = {'baseline_config': {'output': str(self.root / 'parent'), 'iterations': 30000},
                'test_views': 1}
        with patch('compression_phase1.verify_inputs', return_value=info), \
             patch('doctor.diagnose', return_value={'errors': []}), \
             patch('compression_phase1.execute') as execute_mock, \
             patch('compression_phase1.command', return_value=['render']), \
             patch('compression_phase1.compare_images', side_effect=[{'passed': True}, {'passed': False}]):
            with self.assertRaisesRegex(ValueError, 'Opacity50 evaluation blocked'):
                evaluate(self.root)
            self.assertEqual(execute_mock.call_count, 1)
            self.assertFalse(json.loads((self.root / 'evaluation.json').read_text())['complete'])

    def test_cpu_mocked_end_to_end_and_export_tamper_guard(self):
        # Real PLY/images/metadata; only cloud CUDA subprocesses and data discovery are mocked.
        parent = self.root / 'baseline'
        source_dir = self.root / 'dataset'
        source_dir.mkdir()
        pc = parent / 'point_cloud/iteration_30000/point_cloud.ply'
        pc.parent.mkdir(parents=True)
        shutil.copy2(self.source, pc)
        cfg = {'scene': 'fixture', 'source': str(source_dir), 'output': str(parent),
               'images': 'images', 'resolution': -1, 'iterations': 30000,
               'eval_iterations': [30000], 'checkpoint_iterations': [30000],
               'data_device': 'cuda', 'white_background': False, 'port': 6009}
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(cfg))
        upstream = {'commit': 'fixture', 'submodules': []}
        data = {'train': ['train.jpg'], 'test': ['test.jpg'], 'camera_hashes': {},
                'image_sha256': {}}
        (parent / 'manifest.json').write_text(json.dumps(
            {'config': cfg, 'upstream': upstream, 'wrapper_commit': 'fixture'}))
        (parent / 'dataset.json').write_text(json.dumps(data))
        metric = {'PSNR': 20., 'SSIM': .8, 'LPIPS': .2}
        (parent / 'results.json').write_text(json.dumps({'ours_30000': metric}))
        (parent / 'per_view.json').write_text(json.dumps(
            {'ours_30000': {k: {'00000.png': v} for k, v in metric.items()}}))
        for kind in ('renders', 'gt'):
            folder = parent / 'test/ours_30000' / kind
            folder.mkdir(parents=True)
            Image.new('RGB', (4, 3), color=(23, 45, 67)).save(folder / '00000.png')
        experiment = self.root / 'experiment'

        def fake_execute(cmd, model, stage):
            if stage == 'render':
                shutil.copytree(parent / 'test', model / 'test', dirs_exist_ok=True)
            elif stage == 'metrics':
                for filename in ('results.json', 'per_view.json'):
                    shutil.copy2(parent / filename, model / filename)
            elif stage == 'benchmark':
                count = 8 if model.name == 'identity100' else 4
                (model / 'benchmark.json').write_text(json.dumps(
                    {'config_resolution': -1, 'image_sizes': [[4, 3]], 'gaussian_count': count, 'test_views': 1, 'repeats': 3,
                     'samples_ms': [2., 2., 2.], 'mean_ms': 2., 'fps_from_mean_ms': 500.,
                     'peak_torch_allocated_bytes': 1024}))
            else:
                self.fail('Unexpected subprocess stage: ' + stage)

        with patch('compression_phase1.verify_upstream', return_value=upstream), \
             patch('compression_phase1.validate', return_value=data), \
             patch('compression_phase1.code_hashes', return_value={'fixture': 'stable'}), \
             patch('doctor.diagnose', return_value={'errors': [], 'gpu': 'mock-only'}), \
             patch('compression_phase1.execute', side_effect=fake_execute):
            prepare(str(config_path), experiment)
            self.assertEqual(verify_inputs(experiment)['variants']['opacity50']['kept_count'], 4)
            with self.assertRaises(FileExistsError): prepare(str(config_path), experiment)
            evaluate(experiment)
            self.assertTrue(json.loads((experiment / 'evaluation.json').read_text())['complete'])
            with patch('compression_phase1.ROOT', self.root):
                saved = (experiment / 'evaluation.json').read_text()
                for field in ('mean_ms', 'fps_from_mean_ms', 'peak_torch_allocated_bytes', 'gaussian_count', 'ply_bytes'):
                    bad = json.loads(saved); bad['rows'][0][field] += 999
                    (experiment / 'evaluation.json').write_text(json.dumps(bad))
                    with self.assertRaises(ValueError): export(experiment, 'bad-summary')
                (experiment / 'evaluation.json').write_text(saved)
                with patch('compression_phase1.shutil.copy2', side_effect=OSError('copy interrupted')):
                    with self.assertRaises(OSError): export(experiment, 'fixture-result')
                self.assertFalse((self.root / 'results/fixture-result').exists())
                export(experiment, 'fixture-result')
                output = self.root / 'results/fixture-result'
                self.assertTrue((output / 'gt/00000.png').exists())
                self.assertTrue((output / 'opacity50/renders/00000.png').exists())
                self.assertFalse(list(output.rglob('*.ply')))
                with self.assertRaises(FileExistsError): export(experiment, 'fixture-result')
                edited = experiment / 'opacity50/test/ours_30000/renders/00000.png'
                Image.new('RGB', (4, 3), color=(0, 0, 0)).save(edited)
                with self.assertRaisesRegex(ValueError, 'changed after evaluation'):
                    export(experiment, 'tampered')


if __name__ == '__main__': unittest.main()

"""CPU checks; CUDA color-gradient identity is gated again on the cloud."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import contribution_phase3 as p3
from run import write_json
from dataset import sha256
from test_compression import vertices


class ContributionTests(unittest.TestCase):
    def test_weight_gradient_formula(self):
        # Three depth-ordered Gaussians at two pixels; fixed geometry/opacity.
        alpha = np.array([[.2, .6, .3], [.7, .1, .4]])
        trans = np.cumprod(np.concatenate([np.ones((2, 1)), 1-alpha[:, :-1]], axis=1), axis=1)
        weights = alpha * trans
        colors = np.full((3, 3), .5)
        render = lambda c: weights @ c
        base = render(colors).mean()
        score = weights.mean(axis=0)
        self.assertAlmostEqual(score.sum(), base * 2)
        changed = colors.copy(); changed[1] += .01
        self.assertAlmostEqual((render(changed).mean()-base)/.01, score[1])
        # A relatively opaque Gaussian hidden behind another can matter less.
        self.assertLess(.8 * (1 - .99), .1)

    def test_inputs_and_train_only_calls(self):
        self.assertEqual(p3.selected_views(263, 0), list(range(263)))
        self.assertEqual(p3.selected_views(263, 2), [0, 1])
        with self.assertRaises(ValueError): p3.selected_views(263, -1)
        for score in ([0., 0.], [1., -1.], [1., float('nan')], [1.]):
            with self.assertRaises(ValueError): p3.checked_scores(score, 2)
        tree = ast.parse(Path(p3.__file__).read_text(encoding='utf-8'))
        score_func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score')
        attrs = {n.attr for n in ast.walk(score_func) if isinstance(n, ast.Attribute)}
        self.assertIn('getTrainCameras', attrs)
        self.assertFalse(attrs & {'getTestCameras', 'original_image', 'optimizer', 'step'})

    def test_selection_and_mocked_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); parent = root/'parent'; parent.mkdir()
            baseline = root/'baseline'; baseline.mkdir()
            c = dict(scene='test', source=str(root/'data'), output=str(baseline), images='images',
                     iterations=30000, eval_iterations=[30000], checkpoint_iterations=[30000],
                     resolution=-1, data_device='cuda', white_background=False, port=6009)
            metric = {'PSNR':20., 'SSIM':.8, 'LPIPS':.2}
            previous = {'baseline_config':c, 'test_views':1, 'baseline_metrics':metric,
                        'upstream':{'commit':'fixture'}, 'variants':{}}
            for v, n in [('identity100',8),('opacity50',4)]:
                path=p3.ply_path(parent/v,30000);path.parent.mkdir(parents=True)
                PlyData([PlyElement.describe(vertices()[:n], 'vertex')]).write(path)
                previous['variants'][v]={'kept_count':n,'ply_sha256':sha256(path)}
                folder=parent/v/'test/ours_30000/renders';folder.mkdir(parents=True)
                Image.new('RGB',(4,3),(32,45,67)).save(folder/'00000.png')
            np.save(parent/'opacity50/kept_indices.npy',np.array([0,1,2,3]))
            gt=baseline/'test/ours_30000/gt';gt.mkdir(parents=True)
            shutil.copy2(parent/'identity100/test/ours_30000/renders/00000.png',gt/'00000.png')
            experiment=root/'experiment';experiment.mkdir()
            info={'schema':3,'parent':str(parent),'train_indices':[0], 'formal':False,'keep_fraction':.5}
            write_json(experiment/'experiment.json',info)
            scores=np.arange(8,dtype=float);np.save(experiment/'scores.npy',scores)
            write_json(experiment/'scoring.json',{'complete':True,
                'manifest_sha256':sha256(experiment/'experiment.json'),'scores_sha256':sha256(experiment/'scores.npy'),
                'source_ply_sha256':previous['variants']['identity100']['ply_sha256'],'views':[{'train_index':0}]})
            with patch.object(p3,'context',return_value=(info,previous)):
                p3.prepare(experiment)
                self.assertEqual(np.load(experiment/'contribution50/kept_indices.npy').tolist(),[4,5,6,7])
                selected=p3.read_vertices(p3.ply_path(experiment/'contribution50',30000)).copy()
                self.assertEqual(selected.tobytes(),vertices()[4:].tobytes())
                with self.assertRaises(FileExistsError):p3.prepare(experiment)
                def fake_execute(cmd,model,stage):
                    if stage=='render':
                        for kind in ('gt','renders'):
                            target=model/'test/ours_30000'/kind;target.mkdir(parents=True)
                            shutil.copy2(gt/'00000.png',target/'00000.png')
                    elif stage=='metrics':
                        write_json(model/'results.json',{'ours_30000':metric})
                        write_json(model/'per_view.json',{'ours_30000':{k:{'00000.png':v} for k,v in metric.items()}})
                    elif stage=='benchmark':
                        write_json(model/'benchmark.json',{'gaussian_count':8 if model.name=='identity100' else 4,
                            'test_views':1,'repeats':3,'samples_ms':[2.,2.,2.],'mean_ms':2.,
                            'fps_from_mean_ms':500.,'peak_torch_allocated_bytes':1024})
                with patch('doctor.diagnose',return_value={'errors':[]}),patch.object(p3,'execute',side_effect=fake_execute):
                    p3.evaluate(experiment)
                with patch.object(p3,'ROOT',root):
                    p3.export(experiment,'mock')
                    self.assertEqual(len(list((root/'results/mock').glob('*/renders/00000.png'))),3)
                    self.assertFalse(list((root/'results/mock').rglob('*.ply')))
                    with self.assertRaises(FileExistsError):p3.export(experiment,'mock')
                    (experiment/'contribution50/benchmark.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError,'artifacts changed'):p3.export(experiment,'tampered')
                np.save(experiment/'scores.npy',np.ones(8))
                with self.assertRaisesRegex(ValueError,'scoring result'):p3.prepared(experiment)


if __name__=='__main__':
    unittest.main()

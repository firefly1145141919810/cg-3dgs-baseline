"""Regression cases for export consistency and interrupted artifact publication."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import artifact_io
import archive_run
import build_report_assets as report
import export_results
from download_data import safe_extract
from result_validation import checked_benchmark, config_digest
from run import write_json
from dataset import sha256


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        out = self.root / 'model'; out.mkdir()
        c = dict(scene='fixture', output=str(out), source='fixture', iterations=30000, resolution=-1)
        write_json(out / 'manifest.json', {'config': c, 'seed': 0, 'upstream': {'commit': 'fixture'}, 'wrapper_commit': 'fixture'})
        write_json(out / 'dataset.json', {'train': ['train'], 'test': ['a', 'b'], 'camera_hashes': {}})
        write_json(out / 'environment.json', {'gpu': 'fixture'})
        metrics = dict(PSNR=20., SSIM=.8, LPIPS=.2)
        write_json(out / 'results.json', {'ours_30000': metrics})
        write_json(out / 'per_view.json', {'ours_30000': {k: {'00000.png': v, '00001.png': v} for k,v in metrics.items()}})
        ply = out / 'point_cloud/iteration_30000/point_cloud.ply'; ply.parent.mkdir(parents=True)
        ply.write_bytes(b'ply\nformat ascii 1.0\nelement vertex 1\nend_header\n')
        for kind in ('renders','gt'):
            folder = out / 'test/ours_30000' / kind; folder.mkdir(parents=True)
            for i in range(2): Image.new('RGB', (4,3)).save(folder / f'{i:05d}.png')
        b = dict(gaussian_count=1, test_views=2, repeats=3, config_resolution=-1,
                 samples_ms=[2.]*6, mean_ms=2., median_ms=2., fps_from_mean_ms=500.,
                 image_sizes=[[4,3]], peak_torch_allocated_bytes=100,
                 source={'ply_sha256':sha256(ply), 'config_sha256':config_digest(c)})
        write_json(out / 'benchmark.json', b)
        return c, out, ply, b

    def export(self, c, name='fixture'):
        with patch.object(export_results, 'ROOT', self.root), patch.object(export_results, 'load_config', return_value=c), \
             patch.object(sys, 'argv', ['export_results.py','--name',name]):
            export_results.main()

    def test_baseline_rejects_wrong_mean_missing_image_and_size(self):
        c,out,_,_ = self.fixture()
        result = out / 'results.json'; original = result.read_bytes()
        write_json(result, {'ours_30000': dict(PSNR=30., SSIM=.8, LPIPS=.2)})
        with self.assertRaises(ValueError): self.export(c)
        self.assertFalse((self.root / 'results/fixture').exists())
        result.write_bytes(original)
        image = out / 'test/ours_30000/gt/00001.png'; original_image = image.read_bytes(); image.unlink()
        with self.assertRaises(ValueError): self.export(c)
        Image.new('RGB', (5,3)).save(image)
        with self.assertRaises(ValueError): self.export(c)
        image.write_bytes(original_image)
        self.export(c)
        self.assertTrue((self.root / 'results/fixture/summary.json').is_file())

    def test_baseline_benchmark_mismatches(self):
        c,out,ply,b = self.fixture()
        for field,value in [('gaussian_count',999),('test_views',1),('config_resolution',2),
                            ('image_sizes',[[5,3]]),('samples_ms',[2.]),('mean_ms',3.),
                            ('fps_from_mean_ms',999.),('median_ms',9.)]:
            with self.subTest(field=field):
                write_json(out / 'benchmark.json', dict(b, **{field:value}))
                with self.assertRaises(ValueError): self.export(c)
                self.assertFalse((self.root / 'results/fixture').exists())
        write_json(out / 'benchmark.json', b)
        ply.write_bytes(ply.read_bytes() + b'changed-but-same-count')
        with self.assertRaisesRegex(ValueError, 'source'): self.export(c)

    def test_legacy_benchmark_boundary(self):
        c,_,ply,b = self.fixture(); b.pop('source')
        self.assertIn('freshness unproven', checked_benchmark(b,1,2,-1,[[4,3]],ply,c))

    def test_atomic_json_failure_keeps_previous_and_allows_retry(self):
        path = self.root / 'record.json'; write_json(path, {'old':True})
        with patch.object(artifact_io.os, 'replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError): write_json(path, {'new':True})
        self.assertEqual(json.loads(path.read_text()), {'old':True})
        self.assertFalse(list(self.root.glob('*.partial')))
        write_json(path, {'new':True})
        self.assertEqual(json.loads(path.read_text()), {'new':True})

    def test_directory_failure_never_publishes_and_retry_succeeds(self):
        target = self.root / 'export'
        with self.assertRaises(OSError):
            with artifact_io.export_directory(target) as staging:
                write_json(staging / 'summary.json', {'complete':True})
                raise OSError('copy interrupted')
        self.assertFalse(target.exists())
        with artifact_io.export_directory(target) as staging:
            write_json(staging / 'summary.json', {'complete':True})
        with self.assertRaises(FileExistsError):
            with artifact_io.export_directory(target): pass

    def test_recover_truncated_json_preserves_evidence(self):
        path = self.root / 'sample.json'; path.write_text('{"incomplete":')
        self.assertIsNone(artifact_io.recover_json(path))
        self.assertFalse(path.exists())
        self.assertEqual(len(list(self.root.glob('*.corrupt'))),1)
        write_json(path, {'value':1})
        self.assertEqual(artifact_io.recover_json(path), {'value':1})

    def test_zip_resume_identical_but_never_overwrites_different(self):
        archive = self.root / 'data.zip'; dest = self.root / 'data'; dest.mkdir()
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('a.txt','same'); z.writestr('b.txt','second')
        (dest / 'a.txt').write_text('same')
        safe_extract(archive,dest)
        self.assertEqual((dest / 'b.txt').read_text(),'second')
        (dest / 'a.txt').write_text('diff')
        with self.assertRaises(ValueError): safe_extract(archive,dest)
        self.assertEqual((dest / 'a.txt').read_text(),'diff')

    def test_zip_interrupted_member_has_no_final_partial(self):
        archive = self.root / 'data.zip'; dest = self.root / 'data'
        with zipfile.ZipFile(archive,'w') as z: z.writestr('a.txt','content')
        with patch('download_data.shutil.copyfileobj', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError): safe_extract(archive,dest)
        self.assertFalse((dest / 'a.txt').exists())
        safe_extract(archive,dest)
        self.assertEqual((dest / 'a.txt').read_text(),'content')

    def test_archive_failure_retry_and_checksum(self):
        out=self.root/'model';out.mkdir();write_json(out/'manifest.json',{})
        destination=self.root/'archives'
        with patch.object(tarfile.TarFile,'add',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):archive_run.archive_run(out,destination)
        self.assertFalse((destination/'model.tar.gz').exists())
        target=archive_run.archive_run(out,destination)
        self.assertEqual(target.with_suffix('.gz.sha256').read_text().split()[0],sha256(target))
        with self.assertRaises(FileExistsError):archive_run.archive_run(out,destination)

    def test_report_requires_image_sets(self):
        with self.assertRaises(ValueError):report.expected_images(self.root,['00000.png'])
        Image.new('RGB',(4,3)).save(self.root/'00000.png')
        report.expected_images(self.root,['00000.png'])
        with self.assertRaises(ValueError):report.expected_images(self.root,[])

    def test_report_checks_survive_optimized_python(self):
        scripts=Path(report.__file__).parent
        program = "import sys;sys.path.insert(0,sys.argv[1]);import build_report_assets as b;b.require(False,'tampered evidence')"
        for flags in ([],['-O']):
            result=subprocess.run([sys.executable,*flags,'-c',program,str(scripts)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('tampered evidence',result.stderr)

    def test_audit_rejects_missing_images_without_touching_evidence(self):
        original = Path.glob
        def missing(path, pattern):
            if path == report.folder('train', 70) / 'gt':
                return iter(())
            return original(path, pattern)
        with patch.object(Path, 'glob', missing):
            with self.assertRaisesRegex(ValueError, 'image set'): report.audit()

    def test_optimized_audit_rejects_tampered_timing(self):
        program = '''import sys
sys.path.insert(0,sys.argv[1])
import build_report_assets as b
original=b.load
def tampered(path):
    value=original(path)
    if 'samples_ms' in value: value['mean_ms']=999999
    return value
b.load=tampered
b.audit()
'''
        for flags in ([], ['-O']):
            result = subprocess.run([sys.executable,*flags,'-c',program,str(Path(report.__file__).parent)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Evidence audit failed',result.stderr)


if __name__ == '__main__':unittest.main()

import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from publish_timing import digest, encode, public_path, publish
from check_public_repo import findings


class PublicationTests(unittest.TestCase):
    def test_public_scan(self):
        self.assertIn('local-only-file', findings(Path('job.slurm'), ''))
        self.assertIn('unredacted-host', findings(Path('x.json'), '{"measurement_host":"internal.example"}'))
        self.assertEqual(findings(Path('x.json'), '{"measurement_host":"host-1", "PSNR": 21.13611}'), [])
        self.assertIn('private-key', findings(Path('x.txt'), '-----BEGIN ' + 'OPENSSH PRIVATE KEY-----'))

    def test_paths(self):
        self.assertEqual(public_path('/cluster/user/project/outputs/x/model.ply'), 'outputs/x/model.ply')
        self.assertEqual(public_path(r'C:\work\project\outputs\x\config.json'), 'outputs/x/config.json')
        for value in ['/etc/passwd', '/project/outputs/../secret']:
            with self.assertRaises(ValueError): public_path(value)

    def test_derivative_preserves_values_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw'; raw.mkdir(); dest = Path(tmp) / 'public'
            protocol = {'sources': [{'models': {'full': {'config': '/private/project/outputs/x/config.json',
                                                       'ply': '/private/project/outputs/x/model.ply'}}}]}
            a = {'measurement_host': 'private-host', 'samples_ms': [1.2, 3.4], 'mean_ms': 2.3}
            files = {'protocol.json': encode(protocol), 'a.json': encode(a), 'b.json': encode(a)}
            files['summary.json'] = encode({'complete': True, 'rows': [{'fps': 42.0}],
                                           'manifest_sha256': digest(files['protocol.json']),
                                           'sample_sha256': {n: digest(files[n]) for n in ['a.json', 'b.json']}})
            for n, data in files.items(): (raw/n).write_bytes(data)
            record = publish(raw, dest)
            for n, data in files.items():
                self.assertEqual((raw/n).read_bytes(), data)
                self.assertEqual(record['files'][n]['original_sha256'], digest(data))
                self.assertEqual(record['files'][n]['public_sha256'], digest((dest/n).read_bytes()))
            sample = json.loads((dest/'a.json').read_bytes())
            self.assertEqual(sample['samples_ms'], a['samples_ms'])
            self.assertEqual(sample['measurement_host'], 'host-1')
            self.assertEqual(sample, json.loads((dest/'b.json').read_bytes()))
            self.assertNotIn('private', (dest/'protocol.json').read_text())
            with self.assertRaises(FileExistsError): publish(raw, dest)
            (raw/'a.json').write_bytes(encode({'changed': True}))
            with self.assertRaises(ValueError): publish(raw, Path(tmp)/'bad')
            self.assertFalse((Path(tmp)/'bad').exists())

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): publish(tmp, Path(tmp)/'child')


if __name__ == '__main__': unittest.main()

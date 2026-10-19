"""Cloud/CI CPU tests for wrapper logic; no torch, CUDA, images or training."""
import json
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset import validate, binary_names
from run import load_config, command
from download_data import safe_extract, copy_download

class ToolTests(unittest.TestCase):
    def test_download_progress_and_length(self):
        class Progress:
            def __init__(self): self.count = 0
            def update(self, size): self.count += size
        payload = b'x' * (64 * 1024 + 9)
        for total in (len(payload), None):
            progress = Progress()
            output = io.BytesIO()
            copy_download(io.BytesIO(payload), output, progress, total)
            self.assertEqual(output.getvalue(), payload)
            self.assertEqual(progress.count, len(payload))
        with self.assertRaisesRegex(OSError, 'Incomplete download'):
            copy_download(io.BytesIO(b'x'), io.BytesIO(), Progress(), 10)

    def test_config_and_eval_flags(self):
        c=load_config('configs/train.json')
        self.assertEqual(c['iterations'],30000)
        self.assertEqual(c['resolution'],-1)
        for stage in ['train','render']:
            cmd=command(stage,c)
            self.assertIn('--eval',cmd)
            self.assertIn(c['source'],cmd)
        self.assertIn('--skip_train',command('render',c))
        self.assertIn('--checkpoint_iterations',command('train',c))
        self.assertEqual(command('metrics',c)[-2:],['-m',c['output']])

    def test_text_dataset_split_and_missing_image(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); sparse=root/'sparse/0';sparse.mkdir(parents=True)
            images=root/'images';images.mkdir()
            (sparse/'cameras.txt').write_text('1 PINHOLE 800 600 500 500 400 300\n')
            lines=[]
            for i in range(17):
                name=f'{i:03}.png';(images/name).write_bytes(b'fixture-not-an-actual-image')
                lines.append(f'{i+1} 1 0 0 0 0 0 0 1 {name}\n\n')
            (sparse/'images.txt').write_text(''.join(lines))
            (sparse/'points3D.txt').write_text('1 0 0 0 255 0 0 0\n')
            result=validate(root)
            self.assertEqual(result['test'],['000.png','008.png','016.png'])
            self.assertEqual(len(result['train']),14)
            (images/'005.png').unlink()
            with self.assertRaisesRegex(ValueError,'Missing registered'):validate(root)

    def test_binary_image_reader(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'images.bin'
            p.write_bytes(struct.pack('<Q',1)+struct.pack('<idddddddi',1,1,0,0,0,0,0,0,7)+b'a.png\0'+struct.pack('<Q',0))
            self.assertEqual(binary_names(p),[('a.png',7)])

    def test_zip_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);z=root/'bad.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('../escape.txt','bad')
            with self.assertRaises(ValueError):safe_extract(z,root/'data')
            self.assertFalse((root/'escape.txt').exists())

    def test_invalid_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text('{}')
            with self.assertRaises(ValueError):load_config(str(p))

if __name__=='__main__':unittest.main()

"""Mock HTTP downloads: no network, GPU or training dependencies."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from download_data import URL, download_archive


class Progress:
    def __init__(self, **kwargs): self.initial = kwargs['initial']
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, size): pass


class Response(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data)
        self.status, self.headers = status, headers


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive = Path(self.temp.name) / 'data.zip'
        self.part = self.archive.with_suffix('.zip.part')
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z: z.writestr('images/sample.txt', 'test-data' * 100)
        self.payload = buffer.getvalue()

    def seed(self, metadata=True):
        self.part.write_bytes(self.payload[:50])
        if metadata:
            self.part.with_suffix('.part.json').write_text(json.dumps(
                {'url': URL, 'validator': '"v1"', 'total': len(self.payload)}))

    def test_new_download(self):
        def opener(req, **kwargs):
            self.assertIsNone(req.get_header('Range'))
            return Response(self.payload, 200, {'Content-Length': str(len(self.payload)), 'ETag': '"v1"'})
        download_archive(self.archive, Progress, opener)
        self.assertEqual(self.archive.read_bytes(), self.payload)

    def test_resume_and_legacy_partial(self):
        for metadata in (True, False):
            with self.subTest(metadata=metadata):
                self.seed(metadata)
                def opener(req, **kwargs):
                    self.assertEqual(req.get_header('Range'), 'bytes=50-')
                    self.assertEqual(req.get_header('If-range'), '"v1"' if metadata else None)
                    return Response(self.payload[50:], 206, {
                        'Content-Range': f'bytes 50-{len(self.payload)-1}/{len(self.payload)}', 'ETag': '"v1"'})
                download_archive(self.archive, Progress, opener)
                self.assertEqual(self.archive.read_bytes(), self.payload)
                self.archive.unlink()
                self.part.with_suffix('.part.json').unlink()

    def test_range_ignored_restarts_not_appends(self):
        self.seed()
        download_archive(self.archive, Progress, lambda *a, **k: Response(
            self.payload, 200, {'Content-Length': str(len(self.payload)), 'ETag': '"v2"'}))
        self.assertEqual(self.archive.read_bytes(), self.payload)

    def test_bad_range_or_changed_validator_preserves_partial(self):
        for start, etag in [(0, '"v1"'), (50, '"v2"')]:
            self.seed()
            with self.assertRaises(OSError):
                download_archive(self.archive, Progress, lambda *a, **k: Response(
                    self.payload[50:], 206, {'Content-Range': f'bytes {start}-{len(self.payload)-1}/{len(self.payload)}', 'ETag': etag}))
            self.assertEqual(self.part.read_bytes(), self.payload[:50])
            self.assertFalse(self.archive.exists())

    def test_truncated_transfer_kept_then_resumed(self):
        with self.assertRaises(OSError):
            download_archive(self.archive, Progress, lambda *a, **k: Response(
                self.payload[:50], 200, {'Content-Length': str(len(self.payload)), 'ETag': '"v1"'}))
        self.assertEqual(self.part.stat().st_size, 50)
        download_archive(self.archive, Progress, lambda *a, **k: Response(
            self.payload[50:], 206, {'Content-Range': f'bytes 50-{len(self.payload)-1}/{len(self.payload)}', 'ETag': '"v1"'}))
        self.assertEqual(self.archive.read_bytes(), self.payload)

    def test_complete_partial_416(self):
        self.part.write_bytes(self.payload)
        def opener(*args, **kwargs):
            raise urllib.error.HTTPError(URL, 416, 'range', {'Content-Range': f'bytes */{len(self.payload)}'}, None)
        download_archive(self.archive, Progress, opener)
        self.assertEqual(self.archive.read_bytes(), self.payload)

    def test_invalid_zip_not_published(self):
        with self.assertRaises(zipfile.BadZipFile):
            download_archive(self.archive, Progress, lambda *a, **k: Response(b'not-a-zip', 200, {}))
        self.assertFalse(self.archive.exists())
        self.assertTrue(self.part.exists())


if __name__ == '__main__': unittest.main()

"""Crash-safe local publication. Staging paths are never successful artifacts."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid


@contextmanager
def staged_file(target):
    target = Path(target)
    fd, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.partial', dir=target.parent)
    os.close(fd)
    path = Path(name)
    try:
        yield path
        with path.open('r+b') as stream:
            os.fsync(stream.fileno())
        os.replace(path, target)
    finally:
        path.unlink(missing_ok=True)


def atomic_json(path, data):
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with staged_file(path) as temporary:
        temporary.write_text(payload, encoding='utf-8')


@contextmanager
def export_directory(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError('Export exists; use a new label: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + target.name + '.', suffix='.partial', dir=target.parent))
    try:
        yield staging
        if target.exists():
            raise FileExistsError('Export appeared during publication: ' + str(target))
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def recover_json(path):
    """Quarantine only syntactically truncated JSON; never silently accept bad values."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('JSON artifact must be an object: ' + str(path))
        return value
    except (json.JSONDecodeError, UnicodeDecodeError):
        quarantine = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.corrupt')
        path.rename(quarantine)
        print('Quarantined incomplete JSON:', quarantine)
        return None

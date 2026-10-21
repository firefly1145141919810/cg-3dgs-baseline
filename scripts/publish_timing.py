"""Create a redacted timing copy with a separate public hash chain.

Never edit or overwrite the raw measurement directory. No GPU execution.
The original protocol/sample hashes remain in public_redaction.json.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def public_path(value):
    value = value.replace('\\', '/')
    # Retain only the repository-local model path, never a user/cluster prefix.
    match = re.search(r'(?:^|/)outputs/[^\r\n]+$', value)
    if not match or '..' in value[match.start():].split('/'):
        raise ValueError('Expected model path within outputs/')
    return value[match.start():].lstrip('/')


def publish(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Raw and public directories must be disjoint')
    if destination.exists():
        raise FileExistsError('Use a new public directory')
    original = {}
    def read(name):
        data = (source / name).read_bytes()
        original[name] = digest(data)
        return json.loads(data)
    protocol, summary = read('protocol.json'), read('summary.json')
    if not summary.get('complete') or summary['manifest_sha256'] != original['protocol.json']:
        raise ValueError('Incomplete or mismatched source protocol')
    if (source / 'public_redaction.json').exists():
        raise ValueError('Input must be raw evidence, not an already redacted copy')
    public = copy.deepcopy(protocol)
    for item in public['sources']:
        for model in item['models'].values():
            for key in ('config', 'ply'):
                model[key] = public_path(model[key])
    output = {'protocol.json': encode(public)}
    host_ids = {}
    for name, expected in summary['sample_sha256'].items():
        if Path(name).name != name or not re.fullmatch(r'[A-Za-z0-9_-]+\.json', name):
            raise ValueError('Unsafe sample name')
        sample = read(name)
        if original[name] != expected:
            raise ValueError('Raw sample checksum mismatch: ' + name)
        host = sample['measurement_host']
        if not isinstance(host, str) or not host:
            raise ValueError('Missing measurement host')
        host_ids.setdefault(host, 'host-' + str(len(host_ids) + 1))
        sample['measurement_host'] = host_ids[host]
        output[name] = encode(sample)
    public_summary = copy.deepcopy(summary)
    public_summary['manifest_sha256'] = digest(output['protocol.json'])
    public_summary['sample_sha256'] = {name: digest(output[name]) for name in summary['sample_sha256']}
    output['summary.json'] = encode(public_summary)
    record = {'schema': 1, 'purpose': 'Public derivative, not byte-identical raw evidence',
              'changes': ['Model paths made repository-relative', 'Hostnames replaced by within-export aliases',
                          'Summary file checksums recomputed for public bytes'],
              'metrics_changed': False,
              'files': {name: {'original_sha256': original[name], 'public_sha256': digest(data)}
                        for name, data in output.items()}}
    # Validate everything before creating the destination.
    destination.mkdir(parents=True)
    for name, data in output.items():
        (destination / name).write_bytes(data)
    (destination / 'public_redaction.json').write_bytes(encode(record))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    record = publish(args.source, args.destination)
    print(f'Published {len(record["files"])} redacted files; raw input unchanged.')

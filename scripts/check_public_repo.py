"""Heuristic public-tree review (not a history scanner or a secret guarantee).

Checks tracked and unignored new files. Reports paths/categories, never values.
"""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'private-key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'github-token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    'credential-url': re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
    'personal-absolute-path': re.compile(r'(?:[A-Za-z]:[/\\]Users[/\\][^\s"\']+|/(?:gpfs\w*|home)/[A-Za-z0-9_][^\s"\']*)'),
}


def findings(path, text):
    hits = [name for name, pattern in PATTERNS.items() if pattern.search(text)]
    if path.suffix.lower() in ('.slurm', '.pem', '.key'):
        hits.append('local-only-file')
    if path.suffix.lower() == '.json':
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == 'measurement_host' and not re.fullmatch(r'host-\d+', str(item)):
                        hits.append('unredacted-host')
                    walk(item)
            elif isinstance(value, list):
                for item in value: walk(item)
        try: walk(json.loads(text))
        except json.JSONDecodeError: hits.append('invalid-json')
    return sorted(set(hits))


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
    bad, count = [], 0
    for name in sorted(set(names.decode('utf-8').split('\0')) - {''}):
        path = ROOT / name
        if not path.is_file(): continue  # e.g. submodule gitlink
        if path.suffix.lower() not in ('.py', '.md', '.json', '.tex', '.sh', '.slurm', '.txt', '.yml', '.yaml', '.pem', '.key'):
            continue  # binaries require separate PDF/image review
        text = path.read_text(encoding='utf-8-sig')
        hits = findings(path, text); count += 1
        if hits: bad.append((name, hits))
    for name, hits in bad: print(name + ': ' + ', '.join(hits))
    print(f'Checked {count} text files; {len(bad)} flagged. Git history/binaries not scanned.')
    return int(bool(bad))


if __name__ == '__main__': raise SystemExit(main())

"""Parse project code/config only. Never import or execute the training system."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
count=0
for folder in ['scripts','tests']:
    for path in (ROOT/folder).glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path));count+=1
for path in [ROOT/'baseline.lock.json',*(ROOT/'configs').glob('*.json')]:
    json.loads(path.read_text(encoding='utf-8'))
if json.loads((ROOT/'baseline.lock.json').read_text())['commit'] != '2eee0e26d2d5fd00ec462df47752223952f6bf4e':
    raise ValueError('Unexpected upstream commit')
print(f'PASS: parsed {count} Python files and all baseline JSON. GPU execution NOT tested.')

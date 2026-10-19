"""Fetch the author-hosted paper with verified byte ranges; no GPU dependencies."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

URL='https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf'
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if not 1<=a.workers<=16: raise ValueError('Use 1 to 16 workers')
    destination=ROOT/'references/3dgs_original.pdf'
    cache=ROOT/'tmp/paper_parts';cache.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(URL,method='HEAD'),timeout=40) as r:
        size=int(r.headers['Content-Length'])
    chunk=512*1024
    ranges=[(i,min(i+chunk-1,size-1)) for i in range(0,size,chunk)]
    def fetch(pair):
        start,end=pair;path=cache/f'{start}-{end}.part'
        if path.exists() and path.stat().st_size==end-start+1: return path
        req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req,timeout=120) as r:
            if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                raise ValueError('Server did not honor the requested byte range')
            data=r.read()
        if len(data)!=end-start+1: raise ValueError('Incomplete download')
        path.write_bytes(data)
        print(f'Fetched {end+1}/{size}',flush=True)
        return path
    with ThreadPoolExecutor(max_workers=a.workers) as pool: parts=list(pool.map(fetch,ranges))
    temporary=destination.with_suffix('.pdf.part')
    with temporary.open('wb') as out:
        for part in parts: out.write(part.read_bytes())
    from pypdf import PdfReader
    reader=PdfReader(temporary)
    if not reader.pages or 'Gaussian' not in reader.pages[0].extract_text(): raise ValueError('Unexpected PDF')
    temporary.replace(destination)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    (ROOT/'references/paper_source.json').write_text(json.dumps({'url':URL,'sha256':digest,
        'bytes':size,'pages':len(reader.pages),'checksum_type':'locally computed'},indent=2))
    print('Validated PDF:',len(reader.pages),'pages',digest)

if __name__=='__main__':main()

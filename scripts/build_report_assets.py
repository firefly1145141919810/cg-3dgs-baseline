"""CPU-only audit and reproducible report figures/tables from frozen exported JSON/PNG.

Optional reporting dependencies: numpy, Pillow, matplotlib. Never imports torch.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'report'
TIMING = ROOT / 'results/timing-repeat-job4380448'
METHODS = ('identity100_recovery', 'opacity_recovery', 'contribution_recovery')
READS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    READS[path.relative_to(ROOT).as_posix()] = sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def folder(scene, budget):
    name = f'{scene}-scoring-recovery3k-first' if budget == 50 else f'{scene}-budget{budget}-recovery3k-job4371574'
    return ROOT / 'results' / name


def variant(method, budget):
    if budget == 50:
        return method.replace('opacity', 'opacity50').replace('contribution', 'contribution50')
    return method


def audit():
    t = load(TIMING / 'summary.json'); p = load(TIMING / 'protocol.json')
    redaction = TIMING / 'public_redaction.json'
    if redaction.exists():
        disclosure = load(redaction)
        assert disclosure['metrics_changed'] is False
        assert set(disclosure['files']) == {'protocol.json', 'summary.json', *t['sample_sha256']}
        for name, hashes in disclosure['files'].items():
            assert Path(name).name == name
            assert sha(TIMING / name) == hashes['public_sha256']
    assert t['complete'] and p['rounds'] == 3
    assert t['manifest_sha256'] == sha(TIMING / 'protocol.json')
    for name, h in p['script_sha256'].items():
        assert sha(ROOT / 'scripts' / name) == h
    assert p['order'] == [list(METHODS[i:] + METHODS[:i]) for i in range(3)]
    hosts, jobs = set(), set()
    timings = {(r['scene'], r['budget'], r['method']): r for r in t['rows']}
    assert len(timings) == 18 and len(t['sample_sha256']) == 54
    raw = {}
    for name, h in t['sample_sha256'].items():
        assert sha(TIMING / name) == h
        b = load(TIMING / name); raw[name] = b
        assert b['warmup'] == 20 and b['repeats'] == 3
        assert len(b['samples_ms']) == b['test_views'] * 3
        assert all(math.isfinite(x) and x > 0 for x in b['samples_ms'])
        assert math.isclose(statistics.mean(b['samples_ms']), b['mean_ms'], rel_tol=1e-8)
        assert math.isclose(1000 / b['mean_ms'], b['fps_from_mean_ms'], rel_tol=1e-8)
        hosts.add(b['measurement_host']); jobs.add(b['slurm_job_id'])
    quality, counts = {}, {'images': 0, 'pixel_psnr': 0, 'training_reports': 0}
    for source in p['sources']:
        scene, budget = source['scene'], source['budget']
        d = folder(scene, budget); e = load(d / 'summary.json'); protocol = load(d / 'protocol.json')
        assert sha(d / 'summary.json') == source['evaluation_sha256']
        assert e['complete'] and protocol['formal'] and protocol['steps'] == 3000
        quality[scene, budget] = {r['variant']: r for r in e['rows']}
        assert len(quality[scene, budget]) == 6
        seq = set()
        for r in e['rows']:
            v = r['variant']; model = d / v
            per = next(iter(load(model / 'metrics_per_view.json').values()))
            assert sha(model / 'metrics_per_view.json') == e['artifact_sha256'][f'{v}/per_view.json']
            for metric in ('PSNR', 'SSIM', 'LPIPS'):
                values = list(per[metric].values())
                assert len(values) == protocol['test_views'] and all(math.isfinite(x) for x in values)
                assert abs(statistics.mean(values) - r[metric]) < 1e-5
            recovered = v.endswith('_recovery'); iteration = 33000 if recovered else 30000
            for image in (model / 'renders').glob('*.png'):
                assert sha(image) == e['artifact_sha256'][f'{v}/test/ours_{iteration}/renders/{image.name}']
                READS[image.relative_to(ROOT).as_posix()] = sha(image)
                gt = d / 'gt' / image.name
                a = np.asarray(Image.open(image), dtype=np.float64) / 255
                b = np.asarray(Image.open(gt), dtype=np.float64) / 255
                assert abs(-10 * np.log10(np.mean((a-b)**2)) - per['PSNR'][image.name]) < 1e-4
                counts['images'] += 1; counts['pixel_psnr'] += 1
            if recovered:
                tr = load(model / 'training.json'); seq.add(tr['camera_sequence_sha256'])
                assert sha(model / 'training.json') == e['artifact_sha256'][f'{v}/training.json']
                assert tr['complete'] and tr['steps'] == 3000 and len(tr['trace']) == 3000
                assert tr['count'] == r['gaussian_count'] and tr['manifest_sha256'] == e['manifest_sha256']
                assert all(x['step'] == i+1 and math.isfinite(x['loss']) for i, x in enumerate(tr['trace']))
                counts['training_reports'] += 1
        assert len(seq) == 1
        for gt in (d / 'gt').glob('*.png'):
            assert sha(gt) == e['artifact_sha256'][f'identity100/test/ours_30000/gt/{gt.name}']
            READS[gt.relative_to(ROOT).as_posix()] = sha(gt); counts['images'] += 1
        for method in METHODS:
            v = variant(method, budget); tr = load(d / v / 'training.json')
            assert tr['final_ply_sha256'] == source['models'][method]['ply_sha256']
            assert tr['count'] == source['models'][method]['count']
            values = [raw[f'{scene}-{budget}-r{i}-{method}.json'] for i in (1, 2, 3)]
            for b in values:
                assert b['gaussian_count'] == tr['count'] and b['test_views'] == protocol['test_views']
                assert b['gpu'] == p['environment']['gpu']
            means = [b['mean_ms'] for b in values]
            ratios = [raw[f'{scene}-{budget}-r{i}-identity100_recovery.json']['mean_ms'] / means[i-1] for i in (1,2,3)]
            row = timings[scene, budget, method]
            for key, values in [('mean_ms_across_rounds', means), ('fps_across_rounds', [1000/x for x in means]),
                                ('paired_speedup_across_rounds', ratios),
                                ('peak_allocated_bytes', [b['peak_torch_allocated_bytes'] for b in values])]:
                expected = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
                assert row[key] == expected
    spreads = [(r['mean_ms_across_rounds']['max'] - r['mean_ms_across_rounds']['min']) /
               r['mean_ms_across_rounds']['median'] for r in t['rows']]
    assert len(hosts) == len(jobs) == 1 and max(spreads) < .1
    return quality, timings, {'passed': True, **counts, 'timing_samples': 54,
                             'hosts': sorted(hosts), 'jobs': sorted(jobs), 'max_relative_timing_range': max(spreads)}


def qrow(quality, scene, budget, method):
    return quality[scene, budget][variant(method, budget)]


def savefig(fig, name):
    for extension in ('pdf', 'png'):
        fig.savefig(REPORT / 'figures' / f'{name}.{extension}', dpi=180, bbox_inches='tight')


def figures(quality, timings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2, 'pdf.fonttype': 42})
    xs = [30, 50, 70]
    styles = [('opacity', 'Opacity, no recovery', '#db8135', '--'),
              ('contribution', 'Contribution, no recovery', '#288b9b', '--'),
              ('opacity_recovery', 'Opacity + 3k', '#ba5a11', '-'),
              ('contribution_recovery', 'Contribution + 3k', '#00768a', '-')]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.3), constrained_layout=True)
    for i, scene in enumerate(('train', 'truck')):
        for j, metric in enumerate(('PSNR', 'SSIM', 'LPIPS')):
            ax = axes[i, j]
            for method, label, color, ls in styles:
                ax.plot(xs, [qrow(quality,scene,b,method)[metric] for b in xs], 'o', ls=ls, color=color, label=label)
            ax.plot(xs, [qrow(quality,scene,b,'identity100_recovery')[metric] for b in xs], ':', color='#26324a', label='Full + 3k')
            ax.axhline(qrow(quality,scene,50,'identity100')[metric], color='#777777', ls='-.', label='Full 30k')
            ax.set(title=f'{scene}: {metric} ({"lower" if metric=="LPIPS" else "higher"} is better)',
                   xlabel='Retained Gaussians (%)', xticks=xs)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
    savefig(fig, 'quality_curves'); plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(10,6.3),constrained_layout=True)
    colors = ['#26324a','#ba5a11','#00768a']; labels=['Full + 3k','Opacity + 3k','Contribution + 3k']
    for i,scene in enumerate(('train','truck')):
        for j,(key,ylab,scale) in enumerate([('mean_ms_across_rounds','Mean ms / frame',1),
                                            ('paired_speedup_across_rounds','Paired speedup',1),
                                            ('peak_allocated_bytes','Peak allocated MiB',1/2**20)]):
            ax=axes[i,j]
            for method,color,label in zip(METHODS,colors,labels):
                vals=[timings[scene,b,method][key] for b in xs]
                med=np.array([v['median'] for v in vals])*scale
                lo=np.array([v['min'] for v in vals])*scale; hi=np.array([v['max'] for v in vals])*scale
                ax.errorbar(xs,med,yerr=[med-lo,hi-med],marker='o',capsize=4,color=color,label=label)
            ax.set(title=scene,xlabel='Retention budget (%)',ylabel=ylab,xticks=xs)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False)
    savefig(fig,'performance_curves');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,scene in zip(axes,('train','truck')):
        for method,color,label in zip(METHODS,colors,labels):
            xx=[timings[scene,b,method]['fps_across_rounds']['median'] for b in xs]
            yy=[qrow(quality,scene,b,method)['PSNR'] for b in xs]
            ax.plot(xx,yy,'o-',color=color,label=label)
            if method != METHODS[0]:
                for x,y,b in zip(xx,yy,xs): ax.annotate(f'{b}%',(x,y),xytext=(4,4),textcoords='offset points',fontsize=9)
        ax.set(title=scene,xlabel='FPS (median over three processes)',ylabel='PSNR (dB)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False)
    savefig(fig,'quality_speed');plt.close(fig)
    # Fixed diagnostic views and normalized crops; no denoising/color correction.
    views={'train': [('00014',(.05,.35,.60,.90)),('00033',(.18,.30,.72,.90))],
           'truck': [('00014',(.02,.35,.63,.90)),('00031',(.30,.27,.94,.85))]}
    for scene, records in views.items():
        for view,box in records:
            fig, axes=plt.subplots(3,4,figsize=(12,6.3),layout='constrained')
            for row,budget in enumerate((50,50,30)):
                d=folder(scene,budget)
                paths=[d/'gt'/f'{view}.png']+[d/variant(m,budget)/'renders'/f'{view}.png' for m in METHODS]
                for col,path in enumerate(paths):
                    a=np.asarray(Image.open(path)); h,w=a.shape[:2]
                    x0,y0,x1,y1=[int(v*n) for v,n in zip(box,(w,h,w,h))]
                    ax=axes[row,col]; ax.grid(False); ax.set_xticks([]);ax.set_yticks([])
                    ax.imshow(a if row==0 else a[y0:y1,x0:x1], interpolation='nearest')
                    if row==0:
                        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#f9c74f',linewidth=1))
                        ax.set_title(['Ground truth','Full + 3k','Opacity + 3k','Contribution + 3k'][col],fontsize=15)
                    if col==0: ax.set_ylabel(['50% full','50% crop','30% crop'][row],fontsize=14)
            fig.suptitle(f'{scene} / test view {view} - matched full control in each budget',fontsize=16)
            savefig(fig,f'visual_{scene}_{view}');plt.close(fig)


def table(path, headings, rows, align):
    lines=[r'\begin{center}',r'\small',r'\begin{tabular}{'+align+'}',r'\toprule',
           ' & '.join(headings)+r' \\',r'\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{center}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def tables(quality,timings):
    out=REPORT/'generated'
    for scene in ('train','truck'):
        rows=[]
        for b in (70,50,30):
            for method,label in [('identity100','完整'),('opacity','不透明度'),('contribution','贡献度'),
                                 ('identity100_recovery','完整+恢复'),('opacity_recovery','不透明度+恢复'),
                                 ('contribution_recovery','贡献度+恢复')]:
                r=qrow(quality,scene,b,method)
                rows.append([f'{b}\\%',label,f'{r["PSNR"]:.6f}',f'{r["SSIM"]:.6f}',f'{r["LPIPS"]:.6f}',
                             f'{r["gaussian_count"]:,}',f'{r["ply_bytes"]/2**20:.2f}'])
        table(out/f'quality_{scene}.tex',['预算','方法','PSNR','SSIM','LPIPS','高斯数','PLY MiB'],rows,'llrrrrr')
        rows=[]
        for b in (70,50,30):
            for method,label in zip(METHODS,['完整','不透明度','贡献度']):
                r=timings[scene,b,method]; t=r['mean_ms_across_rounds'];s=r['paired_speedup_across_rounds']
                rows.append([f'{b}\\%',label,f'{t["median"]:.4f}',f'{t["min"]:.4f}--{t["max"]:.4f}',
                             f'{r["fps_across_rounds"]["median"]:.2f}',f'{s["median"]:.3f}',
                             f'{r["peak_allocated_bytes"]["median"]/2**20:.2f}'])
        table(out/f'timing_{scene}.tex',['预算','方法','中位 ms','范围 ms','FPS','加速比','显存 MiB'],rows,'llrrrrr')
    rows=[]
    for scene in ('train','truck'):
        for b in (70,50,30):
            c=qrow(quality,scene,b,'contribution_recovery');i=qrow(quality,scene,b,'identity100_recovery')
            t=timings[scene,b,'contribution_recovery']
            rows.append([scene,f'{b}\\%',f'{c["PSNR"]-i["PSNR"]:+.6f}',f'{c["LPIPS"]-i["LPIPS"]:+.6f}',
                         f'{t["paired_speedup_across_rounds"]["median"]:.3f}',f'{c["ply_bytes"]/i["ply_bytes"]:.3f}'])
    table(out/'key_results.tex',['场景','保留率',r'$\Delta$PSNR',r'$\Delta$LPIPS','加速比','大小比例'],rows,'llrrrr')
    rows=[]
    for scene in ('train','truck'):
        scorepath=ROOT/'results'/('train-contribution50-first' if scene=='train' else 'truck-scoring-recovery3k-first')/'scoring.json'
        score=load(scorepath)
        assert score['complete'] and len(score['views']) == (263 if scene=='train' else 219)
        for b in (70,50,30):
            secs=[load(folder(scene,b)/variant(m,b)/'training.json')['training_seconds_excluding_checkpoint_io'] for m in METHODS]
            rows.append([scene,f'{b}\\%',f'{score["seconds_excluding_checkpoint_io"]:.3f}']+[f'{s:.3f}' for s in secs])
    table(out/'overhead.tex',['场景','预算','共享评分 s','完整恢复 s','不透明度恢复 s','贡献度恢复 s'],rows,'llrrrr')


def main():
    for sub in ('figures','generated'):(REPORT/sub).mkdir(parents=True,exist_ok=True)
    quality,timings,checks=audit()
    figures(quality,timings);tables(quality,timings)
    checks['source_sha256']=READS
    checks['builder_sha256']=sha(Path(__file__))
    checks['note']='Checks use exported evidence; full cloud PLY not locally available. No GPU execution.'
    (REPORT/'generated/audit.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k!='source_sha256'},indent=2,ensure_ascii=False))


if __name__=='__main__':main()

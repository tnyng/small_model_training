"""Fixed-checkpoint teacher subgroup/bootstrap analysis, then source-split dev calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load(path):
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    assert len(rows) == len({r['uid'] for r in rows}), f"Duplicate UIDs: {path}"
    assert all(r.get('source_text_sha256') and r.get('cot_id') for r in rows)
    for r in rows:
        for head in ('G5', 'F'):
            assert np.isfinite(r['scores'][head]) and 0 <= r['scores'][head] <= 1
        if r['masks']['G']:
            assert r['labels']['G_score'] in range(1, 6)
        if r['masks']['F']:
            assert r['labels']['F'] in (0, 1)
    return rows


def align(base, large):
    lookup = {r['uid']: r for r in large}
    assert set(lookup) == {r['uid'] for r in base}
    large = [lookup[r['uid']] for r in base]
    for a, b in zip(base, large):
        for key in ('labels', 'masks', 'source_text_sha256', 'cot_id', 'sid', 'generator', 'task'):
            assert a[key] == b[key], (a['uid'], key)
    return large


class Ranking:
    """Sort once; preserve full score ties in weighted AP and threshold curves."""
    def __init__(self, y, p):
        self.y = np.asarray(y, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.order = np.argsort(-self.p, kind='stable')
        self.ends = np.r_[np.flatnonzero(np.diff(self.p[self.order]) != 0), len(y)-1]

    def ap(self, weights=None):
        w = np.ones(len(self.y)) if weights is None else np.asarray(weights)
        sw = w[self.order]
        tp = np.cumsum(sw * self.y[self.order])[self.ends]
        total = np.cumsum(sw)[self.ends]
        if tp[-1] <= 0:
            return None
        precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
        return float(np.sum(np.diff(np.r_[0., tp]) * precision) / tp[-1])

    def top_precision(self, k):
        k = min(k, len(self.y))
        boundary = self.p[self.order[k-1]]
        above, tied = self.p > boundary, self.p == boundary
        return float((self.y[above].sum() + (k-above.sum()) * self.y[tied].mean()) / k)

    def curve(self):
        tp = np.cumsum(self.y[self.order])[self.ends]
        selected = self.ends + 1
        return [dict(threshold=float(self.p[self.order[e]]), selected=int(n),
                     positives=int(hit), negatives=int(n-hit), precision=float(hit/n),
                     recall=float(hit/self.y.sum()) if self.y.sum() else None,
                     coverage=float(n/len(self.y)))
                for e, n, hit in zip(self.ends, selected, tp)]


def arrays(rows, head):
    mask = np.array([bool(r['masks']['G' if head == 'G5' else 'F']) for r in rows])
    subset = [r for r, keep in zip(rows, mask) if keep]
    y = np.array([int(r['labels']['G_score'] == 5) if head == 'G5' else r['labels']['F'] for r in subset])
    p = np.array([r['scores'][head] for r in subset])
    return subset, y, p, mask


def interval(values):
    valid = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    return {'low': float(np.quantile(valid, .025)), 'high': float(np.quantile(valid, .975)),
            'valid_resamples': len(valid)} if len(valid) else None


def summary(rows, y, p):
    rank = Ranking(y, p)
    _, cot_indices, cot_counts = np.unique([r['cot_id'] for r in rows], return_inverse=True, return_counts=True)
    return dict(n=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
                sources=len({r['source_text_sha256'] for r in rows}), cots=len(cot_counts),
                ap=rank.ap(), equal_cot_weight_ap=rank.ap(1/cot_counts[cot_indices]),
                precision_at_20=rank.top_precision(20), precision_at_100=rank.top_precision(100))


def paired_analysis(base, large, out, n_boot, seed):
    source_names, all_source_indices = np.unique([r['source_text_sha256'] for r in base], return_inverse=True)
    # The same source resample is used for both models and both heads.
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(source_names), np.ones(len(source_names))/len(source_names), size=n_boot)
    report = {'sentence_rows': len(base), 'sources': len(source_names), 'seed': seed,
              'bootstrap_replicates': n_boot, 'heads': {}, 'subgroups': []}
    curves = {}
    for head in ('G5', 'F'):
        subset, y, bp, mask = arrays(base, head)
        _, ly, lp, lm = arrays(large, head)
        assert np.array_equal(y, ly) and np.array_equal(mask, lm)
        br, lr = Ranking(y, bp), Ranking(y, lp)
        assert np.isclose(br.ap(), average_precision_score(y, bp))
        assert np.isclose(lr.ap(), average_precision_score(y, lp))
        bootstrap_base, bootstrap_large, deltas = [], [], []
        for sampled in counts:
            weights = sampled[all_source_indices[mask]]
            a, b = br.ap(weights), lr.ap(weights)
            bootstrap_base.append(a); bootstrap_large.append(b)
            deltas.append(b-a if a is not None and b is not None else None)
        bs, ls = summary(subset, y, bp), summary(subset, y, lp)
        report['heads'][head] = {'base': bs, 'large': ls, 'ap_delta': ls['ap']-bs['ap'],
            'ap_relative_change': ls['ap']/bs['ap']-1,
            'base_ap_ci': interval(bootstrap_base), 'large_ap_ci': interval(bootstrap_large),
            'paired_ap_delta_ci': interval(deltas),
            'bootstrap_fraction_delta_positive': float(np.mean([d > 0 for d in deltas if d is not None]))}
        for name, rank in [('base', br), ('large', lr)]:
            curves[head, name] = rank.curve()
            write_csv(out/f'test_{head}_{name}_precision_coverage.csv', curves[head, name])
        for field in ('task', 'generator'):
            vals = np.array([r[field] for r in subset])
            subgroup_rows = []
            for value in sorted(set(vals)):
                take = vals == value
                a, b = Ranking(y[take], bp[take]).ap(), Ranking(y[take], lp[take]).ap()
                row = dict(head=head, grouping=field, group=value, n=int(take.sum()),
                           positives=int(y[take].sum()), source_groups=len({r['source_text_sha256'] for r,t in zip(subset,take) if t}),
                           prevalence=float(y[take].mean()), base_ap=a, large_ap=b,
                           ap_delta=b-a if a is not None else None,
                           sparse_positive_warning=bool(y[take].sum() < 5))
                report['subgroups'].append(row); subgroup_rows.append(row)
            valid = [r for r in subgroup_rows if r['base_ap'] is not None]
            report['heads'][head][f'macro_{field}'] = {
                'groups': len(valid), 'excluded_zero_positive_groups': len(subgroup_rows)-len(valid),
                'base_ap': float(np.mean([r['base_ap'] for r in valid])),
                'large_ap': float(np.mean([r['large_ap'] for r in valid])),
                'large_better_groups': sum(r['ap_delta'] > 0 for r in valid)}
        print(f"[B1] {head}: AP delta={ls['ap']-bs['ap']:.6f}, CI={interval(deltas)}", flush=True)
    report['scope'] = 'One training seed; source sampling uncertainty only. Teacher test has previously been inspected. F negatives are unflagged, not verified nonfallacies.'
    write_csv(out/'subgroups.csv', report['subgroups'])
    write_json(out/'paired_analysis.json', report)
    return report, curves


def dev_partition(source):
    h = hashlib.sha256(('prm-cal-v1:' + source).encode()).hexdigest()
    return 'calibration' if int(h[:8], 16)/16**8 < .6 else 'threshold'


def fit_platt(y, p):
    q = np.clip(p, 1e-7, 1-1e-7)
    z = np.log(q/(1-q))
    def loss_grad(parameters):
        slope = np.exp(parameters[0]); linear = slope*z + parameters[1]
        error = expit(linear)-y
        return (float(np.mean(np.logaddexp(0, linear)-y*linear)),
                np.array([np.mean(error*slope*z), np.mean(error)]))
    result = minimize(loss_grad, [0., 0.], jac=True, method='L-BFGS-B',
                      bounds=[(-8, 8), (-40, 40)], options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-9})
    assert result.success, result.message
    slope, bias = float(np.exp(result.x[0])), float(result.x[1])
    return slope, bias, {'converged': bool(result.success), 'iterations': int(result.nit), 'fit_nll': float(result.fun)}


def quality(y, p):
    bins = []
    ece = 0.
    for left, right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = (p >= left) & (p < right if right < 1 else p <= right)
        if m.any():
            expected, actual = float(p[m].mean()), float(y[m].mean())
            ece += float(m.mean())*abs(expected-actual)
            bins.append(dict(left=float(left), right=float(right), n=int(m.sum()),
                             mean_probability=expected, positive_rate=actual))
    return {'brier': float(brier_score_loss(y,p)), 'nll': float(log_loss(y,np.clip(p,1e-7,1-1e-7), labels=[0,1])),
            'ece_10_equal_width': ece, 'mean_probability': float(p.mean()), 'bins': bins}


def calibration_analysis(rows, out, n_boot, seed):
    partition_manifest = [{'source_text_sha256': s, 'partition': dev_partition(s)}
                          for s in sorted({r['source_text_sha256'] for r in rows})]
    write_csv(out/'dev_source_partitions.csv', partition_manifest)
    report = {'source_groups': {part: sum(r['partition']==part for r in partition_manifest)
                               for part in ('calibration','threshold')}, 'heads': {}}
    curves = {}
    for head in ('G5','F'):
        subset,y,p,_ = arrays(rows,head)
        fit_mask = np.array([dev_partition(r['source_text_sha256'])=='calibration' for r in subset])
        assert 0 < y[fit_mask].sum() < fit_mask.sum() and 0 < y[~fit_mask].sum() < (~fit_mask).sum()
        slope,bias,fit = fit_platt(y[fit_mask],p[fit_mask])
        yt,raw = y[~fit_mask],p[~fit_mask]
        q = np.clip(raw,1e-7,1-1e-7)
        calibrated = expit(slope*np.log(q/(1-q))+bias)
        rank = Ranking(yt,raw)
        # Confirm the mapping has not collapsed ranking through numerical saturation.
        assert np.isclose(rank.ap(),Ranking(yt,calibrated).ap(),atol=1e-7)
        threshold_rows=[r for r,m in zip(subset,~fit_mask) if m]
        sources,inverse=np.unique([r['source_text_sha256'] for r in threshold_rows],return_inverse=True)
        rng=np.random.default_rng(seed)
        boot=rng.multinomial(len(sources),np.ones(len(sources))/len(sources),size=n_boot)
        curve=rank.curve()
        for point in curve:
            val=np.clip(point['threshold'],1e-7,1-1e-7)
            point['calibrated_threshold']=float(expit(slope*np.log(val/(1-val))+bias))
        gates={}
        for target in (.9,.95,.99):
            candidates=[v for v in curve if v['selected']>=20 and v['precision']>=target]
            selected=dict(candidates[-1]) if candidates else None
            if selected is not None:
                take=raw>=selected['threshold']
                n=np.bincount(inverse,weights=take,minlength=len(sources))
                tp=np.bincount(inverse,weights=take*yt,minlength=len(sources))
                bn,btp=boot@n,boot@tp
                selected.update(source_groups=int((n>0).sum()), teacher_negative_per_1000=selected['negatives']*1000/len(yt),
                                selected_threshold_precision_ci=interval(btp[bn>0]/bn[bn>0]),
                                bootstrap_zero_selection_count=int((bn==0).sum()))
            gates[str(target)]=selected
        baseline=np.full(len(yt),float(y[fit_mask].mean()))
        report['heads'][head]={'fit':dict(slope=slope,bias=bias,n=int(fit_mask.sum()),positives=int(y[fit_mask].sum()),**fit),
            'threshold_partition':dict(n=len(yt),positives=int(yt.sum()),prevalence=float(yt.mean()),ap=rank.ap(),
                raw=quality(yt,raw),calibrated=quality(yt,calibrated),prevalence_only=quality(yt,baseline)),
            'candidate_gates':gates,
            'raw_at_0_5':{'selected':int((raw>=.5).sum()),'positives':int(yt[raw>=.5].sum())},
            'calibrated_at_0_5':{'selected':int((calibrated>=.5).sum()),'positives':int(yt[calibrated>=.5].sum())}}
        curves[head]=curve
        write_csv(out/f'dev_{head}_precision_coverage.csv',curve)
        print(f"[B2] {head}: slope={slope:.4f}, bias={bias:.4f}, gates={gates}",flush=True)
    report['scope']='Teacher-dev only. Full dev was already used for checkpoint selection. Gate selection uses the threshold partition, so precision and its fixed-threshold bootstrap interval are optimistic. G calibration targets G_score=5 only. Targets and min20 are exploratory, not deployment requirements.'
    write_json(out/'calibration_dev.json',report)
    return report,curves


def plots(test_curves, dev_curves, cal, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,head,limit in zip(axes,('G5','F'),(.3,.03)):
        for name,color in [('base','#777777'),('large','#0072B2')]:
            curve=test_curves[head,name]
            ax.plot([c['coverage']*100 for c in curve],[c['precision'] for c in curve],label=name,color=color)
        ax.set(xlim=(0,limit*100),ylim=(0,1.02),xlabel='Selected sentences (%)',ylabel='Teacher-positive precision',title=f'{head}: teacher test')
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'test_precision_coverage.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,head,limit in zip(axes,('G5','F'),(.3,.03)):
        curve=dev_curves[head]
        ax.plot([c['coverage']*100 for c in curve],[c['precision'] for c in curve],color='#0072B2')
        for target in (.9,.95,.99):ax.axhline(target,color='gray',alpha=.3,linestyle='--')
        ax.set(xlim=(0,limit*100),ylim=(0,1.02),xlabel='Selected sentences (%)',ylabel='Teacher-positive precision',title=f'{head}: dev threshold partition')
        ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'dev_precision_coverage.png',dpi=180);plt.close(fig)


def checks():
    # Tied scores must contribute as a group, not depend on row order.
    y=np.array([1,0,1,0]);p=np.array([.8,.8,.4,.1]);w=np.array([2,2,1,0])
    rank=Ranking(y,p)
    assert np.isclose(rank.ap(w),average_precision_score(y,p,sample_weight=w))
    assert rank.top_precision(1)==.5
    for perm in ([1,0,2,3],[3,2,1,0]):
        assert np.isclose(rank.ap(),Ranking(y[perm],p[perm]).ap())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-test',type=Path,required=True)
    parser.add_argument('--large-test',type=Path,required=True)
    parser.add_argument('--large-dev',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20261019)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    checks()
    base,large,dev=load(args.base_test),load(args.large_test),load(args.large_dev)
    large=align(base,large)
    assert not ({r['source_text_sha256'] for r in base}&{r['source_text_sha256'] for r in dev})
    provenance={'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.base_test,args.large_test,args.large_dev)},
                'bootstrap':args.bootstrap,'seed':args.seed,'checkpoints':{'base':'epoch3','large':'epoch2'}}
    write_json(args.out/'provenance.json',provenance)
    paired,tc=paired_analysis(base,large,args.out,args.bootstrap,args.seed)
    calibrated,dc=calibration_analysis(dev,args.out,args.bootstrap,args.seed)
    plots(tc,dc,calibrated,args.out)
    print('[done] '+str(args.out),flush=True)


if __name__=='__main__':
    main()

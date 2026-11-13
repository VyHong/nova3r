"""Compare all shared successful samples and expose per-scene regressions."""
import argparse
import datetime
import json
from pathlib import Path
import time

METRICS = {
    'visible_cd': False, 'visible_apd_0.1': True, 'visible_apd_0.05': True,
    'occluded_cd': False, 'occluded_apd_0.1': True, 'occluded_apd_0.05': True,
    'complete_cd': False, 'f_score_0.1_full': True, 'f_score_0.05_full': True,
}


def records(root):
    rows = {}
    for path in sorted(root.rglob('metrics.json'), key=lambda p: p.stat().st_mtime):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # An active worker may still be writing this file.
        rows[record['sample']] = record
    return rows


def report(reference, candidate, output, expected, reference_label='512', candidate_label='1024'):
    a, b = records(reference), records(candidate)
    common = sorted(k for k in a.keys() & b.keys() if a[k].get('status') == b[k].get('status') == 'ok')
    result = dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  expected=expected, reference_successful=sum(r.get('status') == 'ok' for r in a.values()),
                  candidate_successful=sum(r.get('status') == 'ok' for r in b.values()),
                  candidate_failed=sum(r.get('status') == 'error' for r in b.values()),
                  matched=len(common), complete=len(b) == expected, metrics={}, per_sample=[])
    for key, higher in METRICS.items():
        if not common: continue
        delta = [b[k][key] - a[k][key] for k in common]
        gains = [v if higher else -v for v in delta]
        result['metrics'][key] = dict(reference=sum(a[k][key] for k in common)/len(common),
                                     candidate=sum(b[k][key] for k in common)/len(common),
                                     delta=sum(delta)/len(delta), higher_is_better=higher,
                                     improved=sum(v > 1e-9 for v in gains),
                                     regressed=sum(v < -1e-9 for v in gains),
                                     tied=sum(abs(v) <= 1e-9 for v in gains))
    for sample in common:
        result['per_sample'].append(dict(sample=sample, delta={k:b[sample][k]-a[sample][k] for k in METRICS}))
    result['all_aggregate_metrics_at_least_reference'] = bool(common) and all(
        (m['delta'] if m['higher_is_better'] else -m['delta']) >= -1e-9 for m in result['metrics'].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Paired evaluation comparison', '',
             f"Matched successful samples: {len(common)}/{expected}. Candidate errors: {result['candidate_failed']}.",
             'All numbers use the same matched sample IDs. CD is cm; APD/F-score are percent. Partial results do not establish dataset parity.', '',
             f'| Metric | {reference_label} | {candidate_label} | Delta | Improved / regressed / tied scenes |',
             '|---|---:|---:|---:|---:|']
    for key, m in result['metrics'].items():
        lines.append(f"| {key} | {100*m['reference']:.4f} | {100*m['candidate']:.4f} | {100*m['delta']:+.4f} | {m['improved']} / {m['regressed']} / {m['tied']} |")
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected', type=int, default=100)
    p.add_argument('--watch-pid', type=int)
    p.add_argument('--reference-label', default='512')
    p.add_argument('--candidate-label', default='1024')
    args = p.parse_args()
    while True:
        result = report(args.reference, args.candidate, args.output, args.expected,
                        args.reference_label, args.candidate_label)
        if args.watch_pid is None or result['complete']: break
        try:
            state = (Path('/proc') / str(args.watch_pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == 'Z': break
        time.sleep(30)
    print(json.dumps({k:v for k,v in result.items() if k != 'per_sample'}, indent=2))


if __name__ == '__main__': main()

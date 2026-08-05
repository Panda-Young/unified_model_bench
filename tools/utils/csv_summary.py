import csv

rows = list(csv.DictReader(open('summary.csv', encoding='utf-8-sig', errors='replace')))
dml = [r for r in rows if 'DML' in r['backend_name']]
print('total DML rows:', len(dml))
hdr = f"{'time':8} {'arch':8} {'rep':6} {'avg_ms':>8} {'maxdiff':>10} {'avgdiff':>10} {'accel':>7}"
print(hdr)
for r in dml[-14:]:
    print(f"{r['time']:8} {r['arch']:8} {r['repeat_runs']:6} {r['avg_run_ms']:>8} "
          f"{r['max_output_diff']:>10} {r['avg_output_diff']:>10} {r['acceleration_vs_cpu']:>7}")

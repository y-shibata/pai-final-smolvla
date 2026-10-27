from pathlib import Path
import csv, json, hashlib, collections
from datetime import datetime, timezone
p=Path.home()/'projects/pai-final-smolvla/p4a'
cats=['未把持','持上げ後の脱落','移動・設置未完了','判別不能','技術的不成立']
rows=list(csv.DictReader((p/'failure_labels.csv').open()))
targets=list(csv.DictReader((p/'failure_targets.csv').open()))
key=lambda r:(r['run_id'],r['condition'],r['initial_state_index'])
assert len(rows)==106 and len({key(r) for r in rows})==106
assert {key(r) for r in rows}=={key(r) for r in targets}
primary=[r for r in rows if r['scope']=='primary']
reference=[r for r in rows if r['scope']=='reference_Z20']
assert len(primary)==97 and len(reference)==9
assert all(r['category'] in cats for r in rows)
rule=hashlib.sha256((p/'failure_rule.md').read_bytes()).hexdigest()
assert rule=='9ec05dcee88ae87385c5b165cc1e6a06693e771c3508d1c1a1cdf7f3d628333a'
assert all(r['rule_sha256']==rule for r in rows)
cnt=collections.Counter((r['run_id'][2:],r['condition'],r['category']) for r in primary)
totals=collections.Counter(r['category'] for r in primary)
assert totals['判別不能']<=29 and totals['技術的不成立']==0
out=[]
for condition in ['clean','dim','bright']:
 for model in ['N20','N20B']:
  total=sum(cnt[(model,condition,c)] for c in cats)
  for category in cats:
   out.append(dict(condition=condition,model=model,category=category,count=cnt[(model,condition,category)],failure_total=total,evaluation_denominator=60))
with (p/'figures/data/fig3_failure_counts.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
summary={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'primary_count':97,'reference_count':9,'primary_category_totals':{c:totals[c] for c in cats},'unknown_rate_percent':100*totals['判別不能']/97,'condition_model_counts':out,'reference_category_totals':{c:sum(r['category']==c for r in reference) for c in cats},'unknown_trials':[dict(run_id=r['run_id'],condition=r['condition'],initial_state_index=int(r['initial_state_index']),reason=r['reason']) for r in primary if r['category']=='判別不能'],'rule_sha256':rule,'failure_labels_sha256':hashlib.sha256((p/'failure_labels.csv').read_bytes()).hexdigest(),'review_method':'one file at a time; two-camera overviews plus continuous event/approach frames; frozen rule unchanged','inter_rater_reliability':'not measured; single reviewer','stop_triggered':False}
(p/'failure_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'primary_totals':summary['primary_category_totals'],'condition_rows':out,'reference':summary['reference_category_totals']},ensure_ascii=False))
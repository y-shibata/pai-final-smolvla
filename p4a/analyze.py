"""P4a: fixed-plan, CPU-only descriptive analysis of immutable P3 rows."""
from pathlib import Path
import csv, json, hashlib, datetime
import numpy as np
ROOT=Path.home()/'projects/pai-final-smolvla'
OUT=ROOT/'p4a'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save_csv(name,rows):
    p=OUT/name
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
assert sha(OUT/'analysis_plan.json')=='b5da0abccba672b3a3a2d2f772064431837c94182c902456dad9c16274c53599'
assert sha(OUT/'failure_rule.md')=='9ec05dcee88ae87385c5b165cc1e6a06693e771c3508d1c1a1cdf7f3d628333a'
plan=json.loads((OUT/'analysis_plan.json').read_text())
assert sha(ROOT/'p3/results_all.csv')==plan['source_sha256']
assert sha(ROOT/'p3/freeze.json')==plan['freeze_sha256']
freeze=json.loads((ROOT/'p3/freeze.json').read_text())
rows=list(csv.DictReader((ROOT/'p3/results_all.csv').open()))
assert len(rows)==960
models={r['model_id']:r for r in freeze['models']}
conds=['clean','dim','bright']; reps=['R1','R2','R3']; ids=list(range(30,50))
lookup={}
# Validate all correspondence and immutable model/initial-state fingerprints.
for r in rows:
    key=(r['run_id'],r['condition'],int(r['initial_state_index']))
    assert key not in lookup and key[2] in ids
    assert r['status']=='COMPLETE' and not r['technical_error'] and r['attempt']=='1'
    assert r['success'] in ['True','False']
    assert float(r['brightness_factor'])==dict(clean=1,dim=.65,bright=1.35)[key[1]]
    assert r['weights_sha256']==models[key[0]]['weights_sha256']
    assert r['processor_sha256']==models[key[0]]['processor_sha256']
    assert r['wave']==('A' if key[2]<40 else 'B')
    lookup[key]=r
assert len(lookup)==16*3*20
for i in ids:
    rr=[r for r in rows if int(r['initial_state_index'])==i]
    assert len(rr)==48
    for field in ['raw_initial_state_sha256','post_settle_state_sha256','raw_agentview_image_sha256','raw_wrist_image_sha256']:
        assert len({r[field] for r in rr})==1,(i,field)
def success(m,c,i):return int(lookup[m,c,i]['success']=='True')
cells=[]; drops=[]
for m in models:
    cc={c:sum(success(m,c,i) for i in ids) for c in conds}
    for c in conds:
        cells.append(dict(run_id=m,replicate=models[m]['replicate'],training_condition=models[m]['training_condition'],condition=c,successes=cc[c],denominator=20,rate_percent=cc[c]*5))
        if c!='clean':
            drops.append(dict(run_id=m,condition=c,clean_successes=cc['clean'],shifted_successes=cc[c],denominator=20,clean_minus_shifted_pp=(cc['clean']-cc[c])*5))
save_csv('figures/data/fig1_success_rates.csv',cells)
save_csv('clean_to_brightness_drop.csv',drops)
# A single draw matrix carries the same selected initial indices across all groups/conditions.
rng=np.random.Generator(np.random.PCG64(plan['interval']['seed']))
draws=rng.integers(0,20,size=(plan['interval']['resamples'],20))
drawhash=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
A=np.array([[[success(r+'N20',c,i) for i in ids] for c in conds] for r in reps],dtype=np.int8)
B=np.array([[[success(r+'N20B',c,i) for i in ids] for c in conds] for r in reps],dtype=np.int8)
primary=[]; secondary_clean=[]
for ri,rname in [(0,'R1'),(1,'R2'),(2,'R3'),(None,'mean_R1_R3')]:
    ris=[ri] if ri is not None else [0,1,2]
    for cname,cis in [('dim',[1]),('bright',[2]),('equal_dim_bright',[1,2]),('clean',[0])]:
        a=A[np.ix_(ris,cis,range(20))]
        b=B[np.ix_(ris,cis,range(20))]
        # First average within each cluster; bootstrap its 20 index-level contrasts.
        delta=(b-a).mean(axis=(0,1)).astype(np.float64)*100
        boot=delta[draws].mean(axis=1)
        lo,hi=np.quantile(boot,plan['interval']['quantiles'],method='linear')
        rec=dict(group=rname,condition=cname,N20_successes=int(a.sum()),N20B_successes=int(b.sum()),denominator=int(a.size),
                 N20_rate_percent=float(a.mean()*100),N20B_rate_percent=float(b.mean()*100),
                 difference_pp=float(delta.mean()),ci95_low_pp=float(lo),ci95_high_pp=float(hi),
                 B_only=int(((b==1)&(a==0)).sum()),N20_only=int(((b==0)&(a==1)).sum()),
                 both_success=int(((b==1)&(a==1)).sum()),both_failure=int(((b==0)&(a==0)).sum()),
                 index_clusters=20)
        assert rec['B_only']-rec['N20_only']==rec['N20B_successes']-rec['N20_successes']
        (secondary_clean if cname=='clean' else primary).append(rec)
save_csv('primary_comparison.csv',primary)
save_csv('secondary_clean_paired.csv',secondary_clean)
save_csv('figures/data/fig2_primary_paired.csv',primary)
demo=[]
for n in [10,20,46]:
    vals=[sum(success(r+'N'+str(n),'clean',i) for i in ids) for r in reps]
    for r,v in zip(reps,vals):demo.append(dict(demonstrations=n,group=r,successes=v,denominator=20,rate_percent=v*5,min_percent=v*5,max_percent=v*5))
    demo.append(dict(demonstrations=n,group='mean_R1_R3',successes=sum(vals),denominator=60,rate_percent=sum(vals)/60*100,min_percent=min(vals)*5,max_percent=max(vals)*5))
for r in reps:
    demo.append(dict(demonstrations=0,group=r,successes=sum(success(r+'Z20','clean',i) for i in ids),denominator=20,rate_percent=0,min_percent=0,max_percent=0))
demo.append(dict(demonstrations=0,group='mean_R1_R3',successes=0,denominator=60,rate_percent=0,min_percent=0,max_percent=0))
save_csv('figures/data/fig4_demonstrations_clean.csv',demo)
# Candidate selection follows the unchanged minimum-index rule, regardless of outcome.
candidates=[]
for c in ['dim','bright','clean']:
    eligible1=[i for i in ids if not success('R1N20',c,i) and success('R1N20B',c,i)]
    eligible2=[i for i in ids if not success('R1N20B',c,i)]
    for kind,eligible in [('N20失敗・N20B成功',eligible1),('N20B失敗',eligible2)]:
        i=min(eligible) if eligible else min(ids)
        a=lookup['R1N20',c,i];b=lookup['R1N20B',c,i]
        candidates.append(dict(condition=c,selection=kind,eligible_count=len(eligible),fallback=not bool(eligible),initial_state_index=i,N20_success=a['success'],N20B_success=b['success'],
          N20_video_path=a['video_path'],N20_wrist_video_path=a['wrist_video_path'],N20B_video_path=b['video_path'],N20B_wrist_video_path=b['wrist_video_path'],
          N20_control_steps=int(a['control_steps']),N20B_control_steps=int(b['control_steps']),
          N20_sim_seconds=int(a['control_steps'])/20,N20B_sim_seconds=int(b['control_steps'])/20,
          N20_2x_seconds=int(a['control_steps'])/40,N20B_2x_seconds=int(b['control_steps'])/40,
          paired_sim_seconds=max(int(a['control_steps']),int(b['control_steps']))/20,
          paired_2x_seconds=max(int(a['control_steps']),int(b['control_steps']))/40))
save_csv('video_candidates.csv',candidates)
first=candidates[0];r=lookup['R1N20',first['condition'],first['initial_state_index']]
meta=json.loads(Path(r['frame_metadata_path']).expanduser().read_text())
# Each raw frame is captured before its numbered control action, including frame zero.
assert len(meta)==int(r['control_steps'])
for f,x in enumerate(meta):
    assert x['frame']==f and x['control_step']==f and x['saved_fps']==80
    assert abs(x['simulation_time_seconds']-f/20)<1e-9
timebase=[]
for f in sorted(set([0,1,20,40,80,160,len(meta)-1])):
    if f<len(meta):
        x=meta[f]
        timebase.append(dict(run_id=r['run_id'],condition=r['condition'],initial_state_index=r['initial_state_index'],
          frame=f,control_step=x['control_step'],simulation_seconds=x['simulation_time_seconds'],
          mujoco_seconds=x['mujoco_time_seconds'],raw80_file_seconds=f/80,edit_2x_seconds=f/40,
          note='保存フレーム（制御直前）'))
timebase.append(dict(run_id=r['run_id'],condition=r['condition'],initial_state_index=r['initial_state_index'],
 frame='',control_step=int(r['control_steps']),simulation_seconds=int(r['control_steps'])/20,mujoco_seconds='',
 raw80_file_seconds=int(r['control_steps'])/80,edit_2x_seconds=int(r['control_steps'])/40,
 note='区間終端。対応する保存フレームではない'))
save_csv('video_timebase.csv',timebase)
targets=[]
for scope in ['primary','reference_Z20']:
    for m in ([r+n for r in reps for n in ['N20','N20B']] if scope=='primary' else [r+'Z20' for r in reps]):
        for c in (conds if scope=='primary' else ['clean']):
            for i in (ids if scope=='primary' else [30,31,32]):
                r=lookup[m,c,i]
                if r['success']=='False':
                    targets.append(dict(scope=scope,run_id=m,condition=c,initial_state_index=i,control_steps=int(r['control_steps']),video_path=r['video_path'],wrist_video_path=r['wrist_video_path'],frame_metadata_path=r['frame_metadata_path']))
assert sum(r['scope']=='primary' for r in targets)==97
assert sum(r['scope']=='reference_Z20' for r in targets)==9
save_csv('failure_targets.csv',targets)
z=[r for r in rows if 'Z20' in r['run_id']]
trained=[r for r in rows if r['run_id']!='OLD46' and 'Z20' not in r['run_id']]
summary=dict(computed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_row_count=len(rows),technical_errors=0,
 primary=primary,secondary_clean=secondary_clean,all_cells=cells,clean_drops=drops,demonstrations=demo,
 Z20=dict(successes=sum(r['success']=='True' for r in z),denominator=len(z)),
 trained_12_models=dict(successes=sum(r['success']=='True' for r in trained),denominator=len(trained),cell_min=min(r['successes'] for r in cells if r['run_id']!='OLD46' and 'Z20' not in r['run_id']),cell_max=max(r['successes'] for r in cells if r['run_id']!='OLD46' and 'Z20' not in r['run_id'])),
 bootstrap_draws_sha256=drawhash,numpy_version=np.__version__,analysis_plan_sha256=sha(OUT/'analysis_plan.json'),failure_rule_sha256=sha(OUT/'failure_rule.md'),video_candidates=candidates,timebase=timebase)
save_json('analysis_summary.json',summary)
print(json.dumps({k:summary[k] for k in ['computed_at_utc','primary','secondary_clean','Z20','trained_12_models','bootstrap_draws_sha256']},ensure_ascii=False,indent=2))

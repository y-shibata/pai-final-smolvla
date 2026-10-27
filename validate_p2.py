"""P2全runの設定・重み・ログと原本を照合する。GPUは使わない。"""
from pathlib import Path
import datetime,hashlib,json,math,statistics,subprocess
from p1_runtime import ROOT,SOURCE,BASE_SHA,sha,short
P2=ROOT/"p2"
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def validate():
 s=read(P2/"summary.json")
 assert s["status"]=="TRAINING_AND_DEVCHECK_COMPLETE"
 reps=["R1","R2","R3"] if s["additional_replicates_decision"]["eligible"] else ["R1"]
 assert len(s["runs"])==4*len(reps)
 assert {(x["replicate"],x["condition"]) for x in s["runs"]}=={(r,c) for r in reps for c in ["N20","N20B","N10","N46"]}
 complete=[]
 for rec in s["runs"]:
  folder=P2/rec["replicate"]/rec["condition"]
  cfg=read(folder/"run_config.json");steps=read(folder/"steps.json")
  assert len(steps)==30000 and [x["update"] for x in steps]==list(range(1,30001))
  assert all(math.isfinite(x["loss"]) and math.isfinite(x["lr"]) for x in steps)
  assert [json.loads(x) for x in (folder/"steps.jsonl").read_text().splitlines()]==steps
  assert cfg["base_file_sha256"]==BASE_SHA and cfg["updates_requested"]==30000
  # 組別のseedと拡張の有無は計画値と突き合わせる。成績から変更しない。
  rep_number=int(rec["replicate"][1:])
  assert cfg["replicate"]==rec["replicate"] and cfg["condition"]==rec["condition"]
  assert cfg["selection_seed"]==101*rep_number and cfg["training_seed"]==20261010+rep_number
  assert cfg["augmentation_seed"]==20262010+rep_number
  assert cfg["brightness_enabled"]==(rec["condition"]=="N20B")
  assert steps[-1]["lr"]==.0001
  assert cfg["batch_size"]==1 and cfg["num_workers"]==4 and cfg["lora_r"]==cfg["lora_alpha"]==16
  assert cfg["scheduler"]=={"warmup":1000,"decay_steps":30000,"peak_lr":.001,"decay_lr":.0001}
  assert cfg["optimizer"]=={"lr":.001,"betas":[.9,.95],"eps":1e-8,"weight_decay":1e-10,"grad_clip_norm":10}
  assert cfg["chunk_size"]==cfg["n_action_steps"]==50
  loader=read(folder/"loader_audit.json")
  assert loader["selected_outside_count"]==0
  assert sorted(loader["expected_episode_ids"])==loader["seen_episode_ids_one_epoch"]
  assert rec["training_elapsed_seconds"]<=5400
  preflight=read(folder/"preflight.json");assert preflight["status"]=="PASS" and preflight["compute_process_count"]==0
  for step in [0,3000,12000,30000]:
   cp=folder/f"checkpoint_{step:06d}";audit=read(cp/"checkpoint_audit.json")
   assert audit["step"]==step and audit["model_file_sha256"]==sha(cp/"model.safetensors")
   if step==0:assert audit["model_file_sha256"]==BASE_SHA
   assert audit["normalizer_tensor_sha256"]==rec["normalizer_tensor_sha256"]
   assert all(e==0 for e in audit["normalizer_max_abs_errors_vs_stats_cast"].values())
  reload=read(folder/"checkpoint_030000/standalone_reload.json")
  assert reload["same_weights"] and reload["same_normalizers"] and reload["status"]=="PASS"
  rows=read(folder/"devcheck/results.json");initial=read(folder/"devcheck/initial_audit.json")
  assert len(rows)==len(initial)==10
  assert {r["initial_state_index"] for r in rows}==set(range(10))
  assert {r["initial_state_index"] for r in initial}==set(range(10))
  assert all(r["condition"]=="clean" and r["brightness_factor"]==1.0 and r["repeat"]==0 for r in rows)
  assert all(x["equal"] for x in read(folder/"devcheck/paired_hash_checks.json"))
  for stage in ["training","standalone_reload","devcheck"]:
   assert read(folder/(stage+".gpu_after.json"))["compute_processes"]==""
  windows=[]
  for lo,hi in [(1,100),(1,1000),(2001,3000),(11001,12000),(29001,30000)]:
   part=steps[lo-1:hi]
   windows.append({"first_update":lo,"last_update":hi,"mean_loss":statistics.mean(x["loss"] for x in part),
                   "mean_update_seconds":statistics.mean(x["seconds"] for x in part)})
  complete.append({**rec,"loss_windows":windows,"final_lr":steps[-1]["lr"],
       "frames_per_epoch":loader["frames"],"approximate_epochs":30000/loader["frames"],
       "training_started_at_utc":cfg["started_at_utc"],
       "training_finished_at_utc":read(folder/"result.json")["finished_at_utc"],
       "peak_total_gpu_memory_MiB":read(folder/"training.timing.json")["peak_total_gpu_memory_MiB"],
       "peak_gpu_temperature_C":read(folder/"training.timing.json")["peak_gpu_temperature_C"]})
 for rep in reps:assert read(P2/rep/"N20_N20B_pair_check.json")["status"]=="PASS"
 # 元ファイルの内容はhashだけを照合し、ログ本文は成果物へコピーしない。
 originals=[]
 for row in read(ROOT/"dev/evidence/final_originals_check.json")["source_files"]:
  path=Path(row["path"]).expanduser();actual=sha(path)
  assert actual==row["sha256"],"Original asset changed: "+row["path"]
  originals.append({"path":row["path"],"sha256":actual,"unchanged":True})
 p0_checks=[]
 for line in (ROOT/"dev/evidence/P0_SHA256SUMS.before_p1").read_text().splitlines():
  expected,rel=line.split("  ",1);actual=sha(ROOT/rel)
  assert actual==expected,"P0 artifact changed: "+rel
  p0_checks.append({"path":rel,"sha256":actual,"unchanged":True})
 p1_checks=[]
 for line in (P2/"_evidence/SHA256SUMS.before_p2").read_text().splitlines():
  expected,rel=line.split("  ",1)
  if not rel.startswith("dev/"):continue
  actual=sha(ROOT/rel);assert actual==expected,"P1 dev artifact changed: "+rel
  p1_checks.append({"path":rel,"sha256":actual,"unchanged":True})
 head=subprocess.check_output(["git","--no-optional-locks","-C",str(SOURCE/"lerobot"),"rev-parse","HEAD"],text=True).strip()
 status=subprocess.check_output(["git","--no-optional-locks","-C",str(SOURCE/"lerobot"),"status","--porcelain"],text=True)
 gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
 assert head=="7427f31801f944f5ec9ce53cda02862ce7d1638b" and status=="" and gpu==""
 save(P2/"_evidence/final_originals_check.json",{"status":"PASS","source_files":originals,"P0_files":p0_checks,
        "P1_dev_files":p1_checks,"lerobot_head":head,"lerobot_status":status,"compute_processes":gpu})
 summary={**s,"status":"P2_COMPLETE","runs":complete,"validated_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "next_phase_criteria":{"all_authorized_runs_at_30000":True,"settings_weights_logs_saved":True,
     "standalone_reload_all_pass":True,"clean_devcheck_all_completed":True},
  "forbidden_initial_states_loaded":False,"allowed_initial_indices":list(range(10)),
  "P3_not_started":True,"P3_carryover":["Connect final checkpoints and states 30-49 only after P3 authorization",
   "Freeze evaluation code and model list before P3","P2 clean devcheck scores are reference only; do not select checkpoints or change conditions"],
  "P4_comparison_reminder":"Compare N20 vs N20B at identical placement and identical input condition",
  "original_protection":{"source_files":len(originals),"P0_files":len(p0_checks),"P1_dev_files":len(p1_checks),"status":"PASS"},
  "total_training_seconds":sum(x["training_elapsed_seconds"] for x in complete)}
 save(P2/"validated_summary.json",summary)
 return summary
if __name__=="__main__":
 s=validate();print(json.dumps({"status":s["status"],"runs":len(s["runs"]),"total_training_seconds":s["total_training_seconds"]}))

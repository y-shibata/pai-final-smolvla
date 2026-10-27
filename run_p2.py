"""P2を指定順に直列実行する。停止条件に当たれば次のrunを起動しない。"""
import argparse,datetime,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from p1_runtime import ROOT,SOURCE,BASE,BASE_SHA,sha,short
P2=ROOT/"p2"
PYTHON=SOURCE/".venv/bin/python"
HEAD="7427f31801f944f5ec9ce53cda02862ce7d1638b"
MODEL_REV="c83c3163b8ca9b7e67c509fffd9121e66cb96205"
DATA_REV="86958911c0f959db2bbbdb107eb3e17c5f9c798e"
PLAN_MD5="6ccbb83f0f1449bcce0f1fd94556b81b"
JST=datetime.timezone(datetime.timedelta(hours=9))
ORDER=["N20","N20B","N10","N46"]
ACTIVE=None
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,obj):
    """監視側が書込途中のJSONを読むことを避けるため、同じフォルダ内で置換する。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".writing")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())
def command(args):return subprocess.check_output(args,text=True,timeout=30).strip()
def gpu_compute():return command(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"])
def idle_gpu():
    active=gpu_compute()
    assert not active,"STOP: other GPU compute process exists: "+active
    return command(["nvidia-smi"])
def event(kind,**kwargs):
    record={"at_utc":now(),"event":kind,**kwargs}
    with (P2/"events.jsonl").open("a",buffering=1) as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
    print(json.dumps(record,ensure_ascii=False),flush=True)
def expected_p0():
    return {line.split("  ",1)[1]:line.split("  ",1)[0] for line in
            (ROOT/"dev/evidence/P0_SHA256SUMS.before_p1").read_text().splitlines()}
def preflight(rep,condition,folder):
    """実際にloaderへ渡すsnapshot・subset・統計をP0の固定値と照合する。"""
    p0=expected_p0();checks=[]
    for rel in [f"subsets/{rep}/{condition}/selection.json",f"subsets/{rep}/{condition}/stats.json",
                "subsets/manifest.json","evidence/source_inventory.json"]:
        actual=sha(ROOT/rel);expected=p0[rel]
        checks.append({"path":rel,"sha256":actual,"expected_sha256":expected,"equal":actual==expected})
    assert all(c["equal"] for c in checks),"STOP: P0 subset/manifest hash mismatch"
    revision=command(["git","--no-optional-locks","-C",str(SOURCE/"lerobot"),"rev-parse","HEAD"])
    status=command(["git","--no-optional-locks","-C",str(SOURCE/"lerobot"),"status","--porcelain"])
    assert revision==HEAD and status=="","STOP: LeRobot HEAD/status mismatch"
    assert BASE.name==MODEL_REV and sha(BASE/"model.safetensors")==BASE_SHA,"STOP: base mismatch"
    inventory=read(ROOT/"evidence/source_inventory.json")
    dataset=Path(inventory["dataset_snapshot"]).expanduser()
    assert dataset.name==DATA_REV and dataset.is_dir(),"STOP: data snapshot mismatch"
    selection=read(ROOT/f"subsets/{rep}/{condition}/selection.json")
    manifest=read(ROOT/"subsets/manifest.json")
    source_records={str(Path(x["path"]).expanduser()):x for x in manifest["source_shards"]}
    shards=[]
    for path_string in sorted({x["source_shard"] for x in selection["episodes"]}):
        path=Path(path_string).expanduser()
        assert path.is_relative_to(dataset),"STOP: selected shard outside fixed snapshot"
        actual=sha(path);expected=source_records[str(path)]["sha256"]
        assert actual==expected,"STOP: source shard hash mismatch"
        shards.append({"path":short(path),"sha256":actual,"matches_P0":True})
    frozen=read(P2/"_evidence/code_sha256.json")
    for rel,expected in frozen.items():
        assert sha(ROOT/rel)==expected,"STOP: execution code changed after freeze: "+rel
    free=shutil.disk_usage(ROOT).free
    assert free>=50*2**30,"STOP: free disk below 50 GiB"
    gpu=idle_gpu()
    record={"status":"PASS","at_utc":now(),"replicate":rep,"condition":condition,
       "plan_md5_verified_on_mac":PLAN_MD5,"lerobot_head":revision,"lerobot_status":status,
       "base_snapshot":short(BASE),"base_revision":MODEL_REV,"base_sha256":BASE_SHA,
       "dataset_snapshot":short(dataset),"dataset_revision":DATA_REV,"subset_checks":checks,
       "source_shards":shards,"free_bytes":free,"free_GiB":free/2**30,
       "compute_process_count":0,"gpu_before":gpu,"Xorg_display_excluded":True,
       "code_sha256":frozen,"HF_HUB_OFFLINE":"1"}
    save(folder/"preflight.json",record)
    event("PREFLIGHT_PASS",run=rep+condition,free_GiB=free/2**30)
    return record
def stop_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
def run_process(cmd,folder,name,limit,cpu=False):
    """子プロセス群を監視し、時間・容量の停止条件を必ず適用する。"""
    global ACTIVE
    idle_gpu()
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
      HF_HUB_DISABLE_IMPLICIT_TOKEN="1",HF_HUB_DISABLE_TELEMETRY="1",
      TMPDIR=str(ROOT/"tmp"))
    if cpu:env["CUDA_VISIBLE_DEVICES"]=""
    started=time.monotonic();started_at=now();peak=0;peak_temp=0;failure=None
    with (folder/(name+".log")).open("x") as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ACTIVE=proc
        save(folder/(name+".process.json"),{"pid":proc.pid,"started_at_utc":started_at,
             "command":[short(Path(x)) if x.startswith("/") else x for x in cmd],"limit_seconds":limit})
        event("PROCESS_STARTED",run=folder.parent.name+folder.name,phase=name,pid=proc.pid)
        try:
            while proc.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>limit:failure="RUN_EXCEEDED_90_MIN" if name=="training" else "PHASE_TIMEOUT";break
                if shutil.disk_usage(ROOT).free<50*2**30:failure="FREE_DISK_BELOW_50_GiB";break
                values=command(["nvidia-smi","--query-gpu=memory.used,temperature.gpu","--format=csv,noheader,nounits"]).splitlines()[0].split(",")
                peak=max(peak,int(values[0]));peak_temp=max(peak_temp,int(values[1]))
                time.sleep(2)
        except BaseException:
            stop_process(proc);raise
        finally:
            if failure:stop_process(proc)
            rc=proc.wait();ACTIVE=None
    record={"command":[short(Path(x)) if x.startswith("/") else x for x in cmd],
       "started_at_utc":started_at,"finished_at_utc":now(),"elapsed_seconds":time.monotonic()-started,
       "returncode":rc,"limit_seconds":limit,"cpu_only":cpu,"stop_reason":failure,
       "peak_total_gpu_memory_MiB":peak,"peak_gpu_temperature_C":peak_temp,
       "vram_definition":"nvidia-smi whole GPU including EGL/Xorg; own PyTorch peak is recorded by child"}
    save(folder/(name+".timing.json"),record)
    after=gpu_compute();save(folder/(name+".gpu_after.json"),{"at_utc":now(),"compute_processes":after})
    assert after=="","STOP: compute process remains after "+name
    assert failure is None,"STOP: "+str(failure)
    assert rc==0,"STOP: process failure in "+name+"; see "+short(folder/(name+".log"))
    event("PROCESS_COMPLETE",run=folder.parent.name+folder.name,phase=name,elapsed_seconds=record["elapsed_seconds"])
    return record
def verify_finished(folder,rep,condition):
    result=read(folder/"result.json");config=read(folder/"run_config.json")
    cp=folder/"checkpoint_030000";audit=read(cp/"checkpoint_audit.json");reload=read(cp/"standalone_reload.json")
    assert result["status"]=="PASS" and result["completed_updates"]==30000
    assert config["training_seed"]==20261010+int(rep[1:]) and config["augmentation_seed"]==20262010+int(rep[1:])
    assert config["base_file_sha256"]==BASE_SHA and config["updates_requested"]==30000
    assert config["brightness_enabled"]==(condition=="N20B")
    assert reload["status"]=="PASS" and reload["same_weights"] and reload["same_normalizers"]
    assert audit["model_file_sha256"]==sha(cp/"model.safetensors")
    assert audit["normalizer_tensor_sha256"]==reload["normalizer_tensor_sha256"]
    records=[read(folder/f"checkpoint_{step:06d}/checkpoint_audit.json") for step in [0,3000,12000,30000]]
    assert len({x["normalizer_tensor_sha256"] for x in records})==1
    assert records[0]["model_file_sha256"]==BASE_SHA
    assert all(x["model_file_sha256"]!=BASE_SHA for x in records[1:])
    rows=read(folder/"devcheck/results.json")
    assert len(rows)==10 and {x["initial_state_index"] for x in rows}==set(range(10))
    assert all(x["condition"]=="clean" and x["weights_sha256"]==audit["model_file_sha256"] for x in rows)
    checks=read(folder/"devcheck/paired_hash_checks.json")
    assert checks and all(x["equal"] for x in checks)
    return {"run_id":rep+condition,"replicate":rep,"condition":condition,"status":"COMPLETE",
       "selection_seed":config["selection_seed"],"training_seed":config["training_seed"],
       "augmentation_seed":config["augmentation_seed"],"base_sha256":BASE_SHA,
       "selection_sha256":config["selection_sha256"],"stats_sha256":config["stats_sha256"],
       "model_safetensors_sha256":audit["model_file_sha256"],
       "normalizer_tensor_sha256":audit["normalizer_tensor_sha256"],
       "checkpoint_records":records,"training_elapsed_seconds":read(folder/"training.timing.json")["elapsed_seconds"],
       "training_inner_elapsed_seconds":result["elapsed_total_seconds"],
       "max_cuda_allocated_bytes":result["max_cuda_allocated_bytes"],
       "final_loss":result["final_loss"],"loss_first_100_mean":result["loss_first_100_mean"],
       "loss_last_100_mean":result["loss_last_100_mean"],"standalone_reload":"PASS",
       "reload_verified_at_utc":reload["verified_at_utc"],
       "devcheck_successes":sum(x["success"] for x in rows),"devcheck_trials":10,
       "devcheck_scope":"DEVELOPMENT_ONLY_0_9_CLEAN_REFERENCE_NOT_FOR_SELECTION",
       "directory":short(folder),"completed_at_utc":now()}
def compare_pair(rep):
    a=read(P2/rep/"N20/run_summary.json");b=read(P2/rep/"N20B/run_summary.json")
    keys=["selection_sha256","stats_sha256","base_sha256","training_seed","normalizer_tensor_sha256"]
    checks={k:a[k]==b[k] for k in keys}
    for cond,record in [("N20",a),("N20B",b)]:
        record["initial_adapter_sha256"]=sha(P2/rep/cond/"checkpoint_000000/adapter/adapter_model.safetensors")
    checks["initial_adapter_sha256"]=a["initial_adapter_sha256"]==b["initial_adapter_sha256"]
    checks["final_weights_differ"]=a["model_safetensors_sha256"]!=b["model_safetensors_sha256"]
    save(P2/rep/"N20_N20B_pair_check.json",{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
          "initial_adapter_sha256":a["initial_adapter_sha256"]})
    assert all(checks.values()),"STOP: N20/N20B paired input mismatch"
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--resume",action="store_true");a=parser.parse_args()
    summary_path=P2/"summary.json"
    if summary_path.exists():
        assert a.resume,"P2 summary exists; inspect before resume"
        summary=read(summary_path)
    else:summary={"status":"RUNNING","started_at_utc":now(),"runs":[],"additional_replicates_decision":None,"aborted_runs":[]}
    summary["status"]="RUNNING";save(summary_path,summary)
    active_folder=None
    try:
        for rep in ["R1","R2","R3"]:
            if rep!="R1" and not summary["additional_replicates_decision"]["eligible"]:break
            for condition in ORDER:
                folder=P2/rep/condition
                if any(x["run_id"]==rep+condition for x in summary["runs"]):
                    assert a.resume and read(folder/"run_summary.json")["status"]=="COMPLETE";continue
                assert not folder.exists(),"Existing incomplete run needs explicit inspection/archive"
                folder.mkdir(parents=True);active_folder=folder
                preflight(rep,condition,folder)
                save(P2/"active.json",{"status":"RUNNING","run_id":rep+condition,"phase":"training","at_utc":now()})
                run_process([str(PYTHON),str(ROOT/"train_experiment.py"),"--scope","p2","--condition",condition,
                            "--replicate",rep,"--updates","30000"],folder,"training",5400)
                save(P2/"active.json",{"status":"RUNNING","run_id":rep+condition,"phase":"reload","at_utc":now()})
                run_process([str(PYTHON),str(ROOT/"train_experiment.py"),"--reload",
                            str(folder/"checkpoint_030000"),"--device","cpu"],folder,"standalone_reload",600,True)
                save(P2/"active.json",{"status":"RUNNING","run_id":rep+condition,"phase":"devcheck","at_utc":now()})
                run_process([str(PYTHON),str(ROOT/"eval_paired.py"),"--model","P2","--condition","clean",
                            "--replicate",rep,"--training-condition",condition],folder,"devcheck",600)
                record=verify_finished(folder,rep,condition);save(folder/"run_summary.json",record)
                summary["runs"].append(record);save(summary_path,summary)
                active_folder=None
                event("RUN_COMPLETE",run=rep+condition,devcheck_successes_reference=record["devcheck_successes"],
                      training_seconds=record["training_elapsed_seconds"])
                if condition=="N20B":compare_pair(rep)
            if rep=="R1":
                completed=max(datetime.datetime.fromisoformat(x["reload_verified_at_utc"]) for x in summary["runs"] if x["replicate"]=="R1")
                deadline=datetime.datetime(2026,10,18,0,0,tzinfo=JST)
                decision={"R1_completed_at_JST":completed.astimezone(JST).isoformat(),
                          "deadline_exclusive_JST":deadline.isoformat(),"eligible":completed<deadline,
                          "rule":"R1 all four at 30000 saved and independently reloaded before end of 2026-10-17 JST",
                          "continue_replicates":["R2","R3"] if completed<deadline else [],
                          "based_on_performance":False}
                summary["additional_replicates_decision"]=decision
                save(P2/"R2_R3_decision.json",decision);save(summary_path,summary)
                event("ADDITIONAL_REPLICATES_DECISION",**decision)
        summary["status"]="TRAINING_AND_DEVCHECK_COMPLETE";summary["finished_at_utc"]=now()
        save(summary_path,summary);save(P2/"active.json",{"status":summary["status"],"at_utc":now()})
        event("P2_COMPUTE_COMPLETE",runs=len(summary["runs"]))
    except BaseException as exc:
        if ACTIVE is not None:stop_process(ACTIVE)
        error={"at_utc":now(),"reason":str(exc),"traceback":traceback.format_exc().replace(str(Path.home()),"~")}
        if active_folder is not None and active_folder.exists():
            target=P2/"_aborted"/(active_folder.parent.name+active_folder.name+"_"+datetime.datetime.now(JST).strftime("%Y%m%dT%H%M%S"))
            target.parent.mkdir(exist_ok=True)
            save(active_folder/"abort_record.json",error)
            active_folder.rename(target);error["preserved_at"]=short(target)
            summary["aborted_runs"].append(error)
        summary["status"]="STOPPED";summary["stop"]=error;save(summary_path,summary)
        save(P2/"active.json",{"status":"STOPPED",**error});event("STOPPED",**error)
        raise
if __name__=="__main__":main()

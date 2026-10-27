"""P3の凍結順序を直列実行し、技術エラー・資源・GPU残存を監査する。"""
from pathlib import Path
import csv,datetime,json,os,signal,subprocess,time,traceback,shutil
from p1_runtime import ROOT,SOURCE,sha,save,short
P3=ROOT/"p3"
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def gpu_processes():
    return subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader"],text=True).strip()
def assert_frozen(f):
    for p,d in f["code_sha256"].items():assert sha(ROOT/p)==d,"Frozen code changed: "+p
def csv_save(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def summarize(rows):
    cells=[]
    for spec in FREEZE["execution_order"]:
        part=[r for r in rows if r["run_id"]==spec["model_id"] and r["condition"]==spec["condition"] and r["wave"]==spec["wave"]]
        if not part:continue
        cells.append({"model_id":spec["model_id"],"condition":spec["condition"],"wave":spec["wave"],
          "planned":10,"recorded":len(part),"successes":sum(r["success"] is True for r in part),
          "completed":sum(r["status"]=="COMPLETE" for r in part),
          "unestablished":sum(r["status"]!="COMPLETE" for r in part)})
    csv_save(P3/"results_all.csv",rows);csv_save(P3/"cell_counts.csv",cells)
def run_wave(spec,attempt,summary):
    assert_frozen(FREEZE)
    assert not gpu_processes(),"Compute process remained before wave"
    assert shutil.disk_usage(ROOT).free>=50*1024**3,"Free space below 50 GiB"
    wave=f'{spec["model_id"]}_{spec["condition"]}_{spec["wave"]}'
    log=P3/"logs"/f"{wave}_attempt{attempt}.log"
    cmd=[str(SOURCE/".venv/bin/python"),str(ROOT/"eval_p3.py"),"--model",spec["model_id"],
         "--condition",spec["condition"],"--wave",spec["wave"],"--attempt",str(attempt)]
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",
       TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1",TMPDIR=str(ROOT/"tmp"))
    start=time.monotonic();started_at=now();peak=0;stop_reason=None
    with log.open("w") as out:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        summary["active"]={"wave":wave,"attempt":attempt,"pid":proc.pid,"started_at_utc":started_at}
        save(P3/"summary.json",summary)
        while proc.poll() is None:
            used=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
            peak=max(peak,int(used.strip().splitlines()[0]))
            if shutil.disk_usage(ROOT).free<50*1024**3:stop_reason="Free space below 50 GiB"
            if time.monotonic()-GLOBAL_START>10800:stop_reason="P3 exceeded three hours"
            if time.monotonic()-start>300:stop_reason="Wave exceeded 300 seconds; projected total exceeds three hours"
            if stop_reason:
                os.killpg(proc.pid,signal.SIGTERM);time.sleep(2)
                if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL)
                break
            time.sleep(1)
        rc=proc.wait()
    # 自分が開始したwaveの子プロセスだけを対象とする。外部プロセスを終了しない。
    try:
        os.killpg(proc.pid,0)
    except ProcessLookupError:pass
    else:
        os.killpg(proc.pid,signal.SIGTERM);time.sleep(1)
    remaining=gpu_processes()
    record={"wave":wave,"attempt":attempt,"started_at_utc":started_at,"finished_at_utc":now(),
       "elapsed_seconds":time.monotonic()-start,"returncode":rc,"peak_total_gpu_memory_MiB":peak,
       "compute_processes_after":remaining,"stop_reason":stop_reason,"log":short(log)}
    save(P3/"timings"/f"{wave}_attempt{attempt}.json",record)
    summary["wave_attempts"].append(record);summary["active"]=None
    save(P3/"summary.json",summary)
    assert not remaining,"Compute processes remain after wave"
    if stop_reason:raise RuntimeError(stop_reason)
    text=log.read_text()
    if "CUDA out of memory" in text or "torch.OutOfMemoryError" in text:
        raise RuntimeError("GPU OOM: "+short(log))
    out=P3/"eval"/wave/f"attempt_{attempt}"
    # プロセス自体が中断した場合は、結果を推測せず10件とも技術的不成立にする。
    if rc!=0 or not (out/"results.json").exists():
        return [{"run_id":spec["model_id"],"condition":spec["condition"],"condition_id":spec["condition"],
          "replicate":next(m["replicate"] for m in FREEZE["models"] if m["model_id"]==spec["model_id"]),
          "wave":spec["wave"],"attempt":attempt,"initial_state_index":i,
          "brightness_factor":FREEZE["conditions"]["brightness_factors"][spec["condition"]],
          "status":"TECHNICAL_ERROR","success":None,"technical_error":f"wave process returncode={rc}; {short(log)}"}
          for i in spec["indices"]]
    rows=read(out/"results.json");assert len(rows)==10
    return rows
def main():
    global FREEZE,GLOBAL_START
    GLOBAL_START=time.monotonic();FREEZE=read(P3/"freeze.json")
    assert not (P3/"summary.json").exists(),"Automatic resume/re-evaluation is forbidden"
    assert_frozen(FREEZE)
    assert not gpu_processes()
    (P3/"logs").mkdir();(P3/"timings").mkdir()
    summary={"status":"RUNNING","started_at_utc":now(),"freeze_sha256":sha(P3/"freeze.json"),
       "planned_trials":960,"completed_waves":0,"rows":0,"wave_attempts":[],"technical_errors":[],
       "retry_trial_count":0,"model_gpu_checks":[],"active":None}
    rows=[];problem_trials=set()
    save(P3/"summary.json",summary)
    try:
        for number,spec in enumerate(FREEZE["execution_order"]):
            wave_rows=run_wave(spec,1,summary)
            bad=[r for r in wave_rows if r["status"]!="COMPLETE"]
            for r in bad:
                problem_trials.add((r["run_id"],r["condition"],r["initial_state_index"]))
                summary["technical_errors"].append(r)
            cell_bad=sum(x[:2]==(spec["model_id"],spec["condition"]) for x in problem_trials)
            if cell_bad>=3 or len(problem_trials)>=20:
                rows+=wave_rows;summarize(rows)
                raise RuntimeError(f"Technical error stop threshold: cell={cell_bad}, total={len(problem_trials)}")
            if bad:
                # batch10のpolicy乱数列を保つためwaveを同条件で再構成する。
                # 再実行で採用するのは技術的不成立indexだけ。他9/8件は比較文脈であり上書きしない。
                retry=run_wave(spec,2,summary);retry_by_index={r["initial_state_index"]:r for r in retry}
                bad_ids={r["initial_state_index"] for r in bad}
                summary["retry_trial_count"]+=len(bad)
                for r in retry:
                    if r["initial_state_index"] in bad_ids and r["status"]!="COMPLETE":
                        summary["technical_errors"].append(r)
                wave_rows=[retry_by_index[r["initial_state_index"]] if r["initial_state_index"] in bad_ids else r for r in wave_rows]
            rows+=wave_rows;summarize(rows)
            summary.update(completed_waves=number+1,rows=len(rows),
              completed_trials=sum(r["status"]=="COMPLETE" for r in rows),
              unestablished_trials=sum(r["status"]!="COMPLETE" for r in rows),
              elapsed_seconds=time.monotonic()-GLOBAL_START)
            summary["projected_evaluation_seconds"]=summary["elapsed_seconds"]/(number+1)*96
            save(P3/"summary.json",summary)
            if summary["projected_evaluation_seconds"]>10800:
                raise RuntimeError("Projected evaluation time exceeds three hours")
            if spec["condition"]=="bright" and spec["wave"]=="B":
                after=gpu_processes();assert not after
                summary["model_gpu_checks"].append({"model_id":spec["model_id"],"at_utc":now(),"compute_processes":after})
                save(P3/"summary.json",summary)
            print(json.dumps({"completed_waves":number+1,"rows":len(rows),
                 "elapsed_seconds":summary["elapsed_seconds"],"technical_error_trials":len(problem_trials)}),flush=True)
        assert len(rows)==960
        summary.update(status="EVALUATION_COMPLETE",finished_at_utc=now(),elapsed_seconds=time.monotonic()-GLOBAL_START)
        assert_frozen(FREEZE)
    except BaseException:
        summary.update(status="STOPPED",stopped_at_utc=now(),error=traceback.format_exc(),
                       rows=len(rows),elapsed_seconds=time.monotonic()-GLOBAL_START)
        raise
    finally:save(P3/"summary.json",summary)
if __name__=="__main__":main()

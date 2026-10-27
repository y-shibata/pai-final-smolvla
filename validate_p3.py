"""P3の生データ・動画・入力記録・凍結資産を検査する。評価や統計比較は行わない。"""
from pathlib import Path
import collections,concurrent.futures,csv,datetime,hashlib,json,math,subprocess,time,shutil
import numpy as np
from PIL import Image
from p1_runtime import ROOT,SOURCE,sha,short,save
P3=ROOT/"p3"
def read(p):return json.loads(Path(p).read_text())
def path(s):return Path(s).expanduser()
def arr_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def probe(record):
    """両カメラのMP4を最後まで走査し、デコード可能なフレーム数も照合する。"""
    p,steps=record
    proc=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-count_frames",
        "-show_entries","stream=codec_name,width,height,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-of","json",str(p)],capture_output=True,text=True,check=True)
    assert not proc.stderr.strip(),(str(p),proc.stderr)
    v=json.loads(proc.stdout)["streams"][0]
    assert v["codec_name"]=="h264" and v["r_frame_rate"]=="80/1"
    assert int(v["nb_frames"])==int(v["nb_read_frames"])==steps,(str(p),v)
    assert (v["width"],v["height"])==(256,256)
    assert abs(float(v["duration"])-steps/80)<1e-5
    return {"path":short(p),"frames":steps,**v}
def main():
    started=time.monotonic();f=read(P3/"freeze.json");s=read(P3/"summary.json")
    assert s["status"]=="EVALUATION_COMPLETE"
    assert sha(P3/"freeze.json")==s["freeze_sha256"]=="e2173452be653a4c17a377a91f9991afecf5126328400efe425fa7c817f5ff5f"
    assert s["completed_waves"]==96 and s["rows"]==960
    for name,d in f["code_sha256"].items():
        assert sha(ROOT/name)==d and sha(P3/"_evidence/frozen_code"/name)==d
    original_checks=[]
    # P2までの既存成果物は全面照合する。P3実行による上書きがないことの証拠。
    for line in (P3/"_evidence/SHA256SUMS.before_p3").read_text().splitlines():
        digest,rel=line.split("  ",1);actual=sha(ROOT/rel)
        assert actual==digest,rel
        original_checks.append({"path":rel,"sha256":actual,"unchanged":True})
    print(json.dumps({"stage":"prior_artifacts","verified":len(original_checks)}),flush=True)
    source_checks=[]
    for row in read(ROOT/"p2/_evidence/final_originals_check.json")["source_files"]:
        actual=sha(path(row["path"]));assert actual==row["sha256"],row["path"]
        source_checks.append({**row,"unchanged":True})
    for row in f["dataset_shards"]:assert sha(path(row["path"]))==row["sha256"]
    for m in f["models"]:
        for name,d in m["checkpoint_files"].items():assert sha(path(m["checkpoint"])/name)==d
    git=SOURCE/"lerobot/.git";head=(git/"HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref=head[5:];refpath=git/ref
        head=refpath.read_text().strip() if refpath.exists() else next(x.split()[0] for x in (git/"packed-refs").read_text().splitlines() if x.endswith(" "+ref))
    assert head==f["lerobot_head"]
    p0=read(ROOT/"evidence/initial_state_check.json")
    assert sha(path(p0["path"]))==f["pruned_init_sha256"]
    extract=read(P3/"_evidence/initial_state_extraction.json")
    assert f["frozen_at_utc"]<extract["started_at_utc"]<s["started_at_utc"]
    assert sha(P3/"init_states_30_49.npy")==extract["saved_array_sha256"]
    states=np.load(P3/"init_states_30_49.npy",allow_pickle=False);assert states.shape==(20,92)
    assert [arr_sha(x) for x in states]==p0["row_sha256"][30:50]
    models={m["model_id"]:m for m in f["models"]}
    rows=[];audits=[];clipping={};inner_timings=[]
    for spec in f["execution_order"]:
        wave=f'{spec["model_id"]}_{spec["condition"]}_{spec["wave"]}'
        first=P3/"eval"/wave/"attempt_1"
        part=read(first/"results.json")
        bad={r["initial_state_index"] for r in part if r["status"]!="COMPLETE"}
        if bad:
            retry={r["initial_state_index"]:r for r in read(P3/"eval"/wave/"attempt_2/results.json")}
            part=[retry[r["initial_state_index"]] if r["initial_state_index"] in bad else r for r in part]
        assert len(part)==10 and [r["initial_state_index"] for r in part]==spec["indices"]
        assert all(r["run_id"]==spec["model_id"] and r["condition"]==spec["condition"] and r["wave"]==spec["wave"] for r in part)
        rows.extend(part)
        for attempt_dir in sorted(first.parent.glob("attempt_*")):
            audit=read(attempt_dir/"initial_audit.json")
            audits.extend([{"model_id":spec["model_id"],"condition":spec["condition"],"wave":spec["wave"],
                            "attempt":int(attempt_dir.name[-1]),**x} for x in audit])
            if (attempt_dir/"timing.json").exists():inner_timings.append(read(attempt_dir/"timing.json"))
            if (attempt_dir/"clipping.json").exists():clipping[(wave,int(attempt_dir.name[-1]))]=read(attempt_dir/"clipping.json")
    # CSVを元のJSON行と全カラムで照合。真偽値はCSVでの文字表現へそろえる。
    with (P3/"results_all.csv").open(newline="") as inp:csv_rows=list(csv.DictReader(inp))
    assert len(csv_rows)==len(rows)==960
    for a,b in zip(rows,csv_rows,strict=True):
        assert {k:"" if a.get(k) is None else str(a[k]) for k in b}==b
    keys=[(r["run_id"],r["condition"],r["initial_state_index"]) for r in rows]
    assert len(set(keys))==960
    expected={(m,c,i) for m in models for c in ["clean","dim","bright"] for i in range(30,50)}
    assert set(keys)==expected
    references={};mismatches=[]
    for a in audits:
        i=a["initial_state_index"];assert 30<=i<=49 and a["state_seed"]==20263010+i
        assert a["raw_initial_state_sha256"]==p0["row_sha256"][i]
        ref=references.setdefault(i,a)
        for k in ["raw_initial_state_sha256","post_settle_state_sha256","raw_image_sha256"]:
            if a[k]!=ref[k]:mismatches.append({"index":i,"model":a["model_id"],"condition":a["condition"],
                   "attempt":a["attempt"],"key":k,"reference":ref[k],"observed":a[k]})
    audit_lookup={(a["model_id"],a["condition"],a["initial_state_index"],a["attempt"]):a for a in audits}
    videos=[];input_checks=[];frame_count=0;clean_inputs={}
    # 凍結順はcleanが先のため同じ配置の明暗入力も原入力と数値照合できる。
    for r in rows:
        m=models[r["run_id"]];i=r["initial_state_index"]
        assert r["weights_sha256"]==m["weights_sha256"] and r["processor_sha256"]==m["processor_sha256"]
        assert r["brightness_factor"]==f["conditions"]["brightness_factors"][r["condition"]]
        assert r["policy_seed"]==20264010 and r["state_seed"]==20263010+i
        a=audit_lookup[(r["run_id"],r["condition"],i,r["attempt"])]
        for k in ["raw_initial_state_sha256","post_settle_state_sha256"]:assert r[k]==a[k]
        assert r["raw_agentview_image_sha256"]==a["raw_image_sha256"]["image"]
        assert r["raw_wrist_image_sha256"]==a["raw_image_sha256"]["image2"]
        if r["status"]!="COMPLETE":
            assert r["success"] is None and r["technical_error"];continue
        assert isinstance(r["success"],bool) and 1<=r["control_steps"]<=280
        if not r["success"]:assert r["control_steps"]==280
        steps=r["control_steps"]
        videos.extend([(path(r["video_path"]),steps),(path(r["wrist_video_path"]),steps)])
        frame=read(path(r["frame_metadata_path"]));assert len(frame)==steps
        for j,item in enumerate(frame):
            assert item["frame"]==item["control_step"]==j and item["saved_fps"]==80
            assert abs(item["simulation_time_seconds"]-j/20)<1e-10
            assert abs(item["mujoco_time_seconds"]-a["post_settle_simulation_time"]-j/20)<1e-8
        frame_count+=len(frame)
        meta=read(path(r["model_input_metadata_path"]))
        assert meta["control_step"]==0 and meta["brightness_factor"]==r["brightness_factor"]
        with np.load(path(r["model_input_path"]),allow_pickle=False) as arrays:
            for cam,label in [("observation.images.image","agentview"),("observation.images.image2","wrist")]:
                val=arrays[cam];info=meta["inputs"][cam]
                assert list(val.shape)==info["shape"]==[3,256,256] and str(val.dtype)=="float32"
                assert np.isfinite(val).all() and val.min()>=0 and val.max()<=1
                assert arr_sha(val)==info["array_sha256"]
                with Image.open(path(info["preview_path"])) as preview:
                    actual=np.asarray(preview)
                expected_preview=np.round(np.clip(val.transpose(1,2,0),0,1)*255).astype(np.uint8)
                assert np.array_equal(actual,expected_preview)
                key=(r["run_id"],i,cam)
                if r["condition"]=="clean":clean_inputs[key]=val.copy()
                else:assert np.allclose(val,np.clip(clean_inputs[key]*np.float32(r["brightness_factor"]),0,1),atol=1e-7,rtol=0)
        wave=f'{r["run_id"]}_{r["condition"]}_{r["wave"]}';clips=clipping[(wave,r["attempt"])]
        assert len(clips)>=steps
        slot=i-(30 if r["wave"]=="A" else 40)
        for j,c in enumerate(clips[:steps]):
            assert c["control_step"]==j and c["k"]==[float(np.float32(r["brightness_factor"]))]*10
        for cam,label in [("observation.images.image","agentview"),("observation.images.image2","wrist")]:
            for metric in ["clipped_fraction","clipped_pixel_fraction"]:
                expected_mean=float(np.mean([c[metric][cam][slot] for c in clips[:steps]]))
                assert r[label+"_"+metric]==expected_mean
        input_checks.append({"run_id":r["run_id"],"condition":r["condition"],"index":i,
                             "exact_tensor_and_png_match":True,"fixed_brightness_verified":True})
    assert len({str(p) for p,_ in videos})==len(videos)
    all_mp4=list((P3/"eval").rglob("*.mp4"))
    complete=sum(r["status"]=="COMPLETE" for r in rows)
    assert len(videos)==complete*2
    if not s["technical_errors"]:assert len(all_mp4)==len(videos)==1920
    print(json.dumps({"stage":"rows_inputs_frames","verified_trials":len(input_checks),"videos_to_check":len(videos)}),flush=True)
    # 動画検査は評価終了後に実施し、評価中のGPUやCPU速度に影響させない。
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        video_checks=[]
        for number,record in enumerate(pool.map(probe,videos),1):
            video_checks.append(record)
            if number%200==0:print(json.dumps({"stage":"video_decode","verified":number,"total":len(videos)}),flush=True)
    wave_cells=[];cells=[]
    for spec in f["execution_order"]:
        part=[r for r in rows if r["run_id"]==spec["model_id"] and r["condition"]==spec["condition"] and r["wave"]==spec["wave"]]
        wave_cells.append({"model_id":spec["model_id"],"condition":spec["condition"],"wave":spec["wave"],"planned":10,
          "recorded":len(part),"successes":sum(r["success"] is True for r in part),
          "completed":sum(r["status"]=="COMPLETE" for r in part),"unestablished":sum(r["status"]!="COMPLETE" for r in part)})
    with (P3/"cell_counts.csv").open(newline="") as inp:count_csv=list(csv.DictReader(inp))
    assert [{k:str(v) for k,v in x.items()} for x in wave_cells]==count_csv
    for m in models:
        for c in ["clean","dim","bright"]:
            part=[r for r in rows if r["run_id"]==m and r["condition"]==c]
            assert len(part)==20
            cells.append({"model_id":m,"condition":c,"planned":20,
              "successes":sum(r["success"] is True for r in part),
              "completed":sum(r["status"]=="COMPLETE" for r in part),
              "unestablished":sum(r["status"]!="COMPLETE" for r in part)})
    assert len(s["model_gpu_checks"])==16
    assert all(not x["compute_processes"] for x in s["model_gpu_checks"])
    assert all(not x["compute_processes_after"] for x in s["wave_attempts"])
    gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
    assert not gpu
    timing={k:sum(t[k] for t in inner_timings) for k in ["env_build_and_reset_seconds","model_load_seconds",
         "inference_and_simulation_seconds","video_save_seconds","total_seconds"]}
    timing["wave_process_wall_seconds"]=sum(t["elapsed_seconds"] for t in s["wave_attempts"])
    timing["process_startup_cleanup_other_seconds"]=timing["wave_process_wall_seconds"]-timing["total_seconds"]
    timing["orchestrator_between_waves_seconds"]=s["elapsed_seconds"]-timing["wave_process_wall_seconds"]
    timing["evaluation_wall_seconds"]=s["elapsed_seconds"]
    save(P3/"_evidence/paired_hash_validation.json",{"status":"PASS" if not mismatches else "MISMATCH",
         "indices":list(range(30,50)),"observations":len(audits),"mismatches":mismatches,
         "per_index_observations":dict(collections.Counter(a["initial_state_index"] for a in audits)),
         "reference_by_index":references})
    save(P3/"_evidence/video_validation.json",{"status":"PASS","videos":len(videos),"checks":video_checks})
    save(P3/"_evidence/input_frame_validation.json",{"status":"PASS","trials":len(input_checks),"checks":input_checks,
                                                   "control_frame_metadata_count":frame_count})
    save(P3/"_evidence/final_originals_check.json",{"status":"PASS","previous_artifact_count":len(original_checks),
         "source_file_count":len(source_checks),"previous_artifacts":original_checks,"source_files":source_checks,
         "frozen_code_and_checkpoint_files":"PASS","dataset_shards":"PASS","lerobot_head":head,"compute_processes":gpu})
    result={"status":"P3_COMPLETE" if not mismatches else "P3_EVALUATED_HASH_REVIEW_REQUIRED",
       "freeze_sha256":sha(P3/"freeze.json"),"results_all_csv_sha256":sha(P3/"results_all.csv"),
       "cell_counts_csv_sha256":sha(P3/"cell_counts.csv"),"planned_trials":960,"completed_trials":complete,
       "unestablished_trials":960-complete,"technical_error_attempt_records":len(s["technical_errors"]),
       "retry_trial_count":s["retry_trial_count"],"cell_counts":cells,
       "paired_hash_mismatches":mismatches,"paired_hash_checked_observations":len(audits),"videos":len(videos),
       "total_mp4_including_retry_context":len(all_mp4),"input_representative_trial_count":len(input_checks),
       "started_at_utc":s["started_at_utc"],"finished_at_utc":s["finished_at_utc"],
       "validated_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "validation_seconds":time.monotonic()-started,"timing_seconds":timing,
       "max_cuda_allocated_MiB":max(t["max_cuda_allocated_bytes"] for t in inner_timings)/1024**2,
       "max_cuda_reserved_MiB":max(t["max_cuda_reserved_bytes"] for t in inner_timings)/1024**2,
       "peak_total_gpu_memory_MiB":max(t["peak_total_gpu_memory_MiB"] for t in s["wave_attempts"]),
       "free_GiB":shutil.disk_usage(ROOT).free/1024**3,
       "next_phase_criteria":{"evaluation_pairing_correct":not mismatches,"technical_error_handling_explicit":True},
       "loss_interpretation":f["loss_interpretation"],
       "P4_not_performed":True,"training_after_final_evaluation":False}
    save(P3/"validated_summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ["cell_counts","timing_seconds"]}))
if __name__=="__main__":main()

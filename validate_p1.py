"""P1の保存結果を機械的に再確認し、90開発試行と10再現性試行を分けて集計する。"""
from pathlib import Path
import csv,json,subprocess,hashlib
from p1_runtime import ROOT,save,short
def main():
    folder=ROOT/"dev/eval"
    waves={}
    model_ids=["OLD46","Z20","N20_200"]
    conditions=["clean","dim","bright"]
    expected=[f"{m}_{c}_repeat0" for m in model_ids for c in conditions]+["OLD46_clean_repeat1"]
    baseline=json.loads((folder/"OLD46_clean_repeat0/initial_audit.json").read_text())
    all_rows=[];video_checks=[];timings={}
    for name in expected:
        root=folder/name
        audit=json.loads((root/"initial_audit.json").read_text())
        assert audit==baseline,"STOP initial or settled states/images differ"
        rows=json.loads((root/"results.json").read_text())
        assert len(rows)==10 and [x["initial_state_index"] for x in rows]==list(range(10))
        clips=json.loads((root/"clipping.json").read_text())
        for row in rows:
            index=row["initial_state_index"];count=row["control_steps"]
            assert 1<=count<=280
            frames=json.loads(Path(row["frame_metadata_path"]).expanduser().read_text())
            assert len(frames)==count
            for i,f in enumerate(frames):
                assert f["frame"]==f["control_step"]==i
                assert abs(f["simulation_time_seconds"]-i/20)<1e-12
                assert abs(f["mujoco_time_seconds"]-.5-i/20)<1e-8
            for key in ["video_path","wrist_video_path"]:
                video=Path(row[key]).expanduser()
                info=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
                  "-show_entries","stream=nb_frames,avg_frame_rate,duration","-of","json",str(video)]))["streams"][0]
                assert int(info["nb_frames"])==count and info["avg_frame_rate"]=="80/1"
                assert abs(float(info["duration"])-count/80)<.002
                video_checks.append({"path":short(video),"frames":count,"fps":80,"status":"PASS"})
            row=dict(row)
            for camera in ["observation.images.image","observation.images.image2"]:
                actual=[c for c in clips if c["control_step"]<count]
                assert len(actual)==count
                assert all(abs(c["k"][index]-row["brightness_factor"])<1e-7 for c in actual)
                row[camera+"_clipped_pixel_fraction_mean"]=sum(c["clipped_pixel_fraction"][camera][index] for c in actual)/count
            all_rows.append(row)
        timing=json.loads((root/"timing.json").read_text())
        guard_name="eval_OLD46_clean_repeat0_retry" if name=="OLD46_clean_repeat0" else "eval_"+name
        guard=json.loads((ROOT/"dev/evidence"/(guard_name+".timing.json")).read_text())
        assert guard["returncode"]==0 and not guard["timed_out"]
        timing["process_seconds"]=guard["elapsed_seconds"]
        timing["process_startup_cleanup_other_seconds"]=guard["elapsed_seconds"]-sum(timing[k] for k in [
            "env_build_and_reset_seconds","model_load_seconds","inference_and_simulation_seconds","video_save_seconds"])
        timing["gpu_total_device_peak_MiB_including_display_and_renderers"]=guard["peak_total_gpu_memory_MiB"]
        timings[name]=timing
        waves[name]={"trials":10,"successes":sum(x["success"] for x in rows),"steps":[x["control_steps"] for x in rows]}
    # 最終P3の6モデルのうち4学習モデルを、N20_200の実測で代用して見積もる。
    parts=["env_build_and_reset_seconds","model_load_seconds","inference_and_simulation_seconds",
           "video_save_seconds","process_startup_cleanup_other_seconds","process_seconds"]
    p3={key:sum(2*sum(weight*timings[f"{model}_{cond}_repeat0"][key] for model,weight in
        [("OLD46",1),("Z20",1),("N20_200",4)]) for cond in conditions) for key in parts}
    first=[x for x in all_rows if x["repeat"]==0 and x["run_id"]=="OLD46" and x["condition"]=="clean"]
    repeated=[x for x in all_rows if x["repeat"]==1]
    assert all((a["success"],a["control_steps"])==(b["success"],b["control_steps"]) for a,b in zip(first,repeated))
    summary={"status":"PASS","development_trials":90,"repeatability_trials":10,"total_trials":100,
       "total_camera_videos":len(video_checks),"all_10_initial_states_all_waves_identical":True,
       "same_condition_repeat_outcomes_and_steps_identical":True,"waves":waves,"timings":timings,
       "p3_360_trial_estimate":p3,"p3_formula":"For each image condition: 2 waves * (OLD46 + Z20 + 4 * N20_200); 36 waves total",
       "p3_limitations":"P1 states 0..9 and 200-update model used as proxy; 30..49 not loaded. Success timing and thermal behavior may change in P3.",
       "vram_note":"max_cuda_allocated_bytes is the policy process PyTorch allocator peak. GPU total device peak also includes renderer processes and Xorg; no subtraction."}
    save(ROOT/"dev/evaluation_summary.json",summary)
    save(ROOT/"dev/video_frame_validation.json",video_checks)
    save(ROOT/"dev/all_trial_results.json",all_rows)
    with (ROOT/"dev/all_trial_results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

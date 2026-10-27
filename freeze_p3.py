"""評価前の照合と凍結。モデルや評価結果を参照して設定を選ばない。"""
from pathlib import Path
import ast,datetime,hashlib,json,subprocess,difflib,shutil
from p1_runtime import ROOT,SOURCE,BASE,BASE_SHA,OLD,sha,save,short
P3=ROOT/"p3"
def processor_hash(folder):
    records=[]
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and (p.name.startswith("policy_") or "tokenizer" in p.parts):
            records.append([str(p.relative_to(folder)),sha(p)])
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),records
def main():
    assert not (P3/"freeze.json").exists(),"A freeze cannot be overwritten"
    before=json.loads((P3/"_evidence/experiment_conditions.before_p3.json").read_text())
    e=before["evaluation"]
    assert e["proposed_final_state_indices"]==list(range(30,50))
    assert e["brightness_factors"]=={"clean":1,"dim":.65,"bright":1.35}
    assert e["state_rng_base"]==20263010 and e["policy_rng_base"]==20264010
    assert e["hard_reset"] and e["control_frequency_hz"]==20 and e["max_control_steps"]==280
    assert e["initial_settling_steps"]==10 and e["with_optional_replicates"]["total_final_rollouts"]==960
    p2=json.loads((ROOT/"p2/validated_summary.json").read_text());assert p2["status"]=="P2_COMPLETE"
    expected={m["run_id"]:m for m in p2["runs"]};assert len(expected)==12
    # Gitコマンドは使わずHEAD参照ファイルを読み取るだけ。
    git=SOURCE/"lerobot/.git";head=(git/"HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref=head[5:];refpath=git/ref
        if refpath.exists():head=refpath.read_text().strip()
        else:head=next(x.split()[0] for x in (git/"packed-refs").read_text().splitlines() if x.endswith(" "+ref))
    assert head=="7427f31801f944f5ec9ce53cda02862ce7d1638b"
    assert sha(BASE/"model.safetensors")==BASE_SHA
    pf=json.loads((ROOT/"p2/R3/N46/preflight.json").read_text())
    assert pf["dataset_revision"]=="86958911c0f959db2bbbdb107eb3e17c5f9c798e"
    shards=[]
    for row in pf["source_shards"]:
        actual=sha(Path(row["path"]).expanduser());assert actual==row["sha256"],row["path"]
        shards.append({"path":row["path"],"sha256":actual})
    models=[]
    order=[(r,c) for r in ["R1","R2","R3"] for c in ["N20","N20B","N10","N46"]]
    order += [(r,"Z20") for r in ["R1","R2","R3"]]+[("common","OLD46")]
    for rep,c in order:
        mid=rep+c if c!="OLD46" else c
        folder=OLD if c=="OLD46" else ROOT/"p2"/rep/("N20" if c=="Z20" else c)/("checkpoint_000000" if c=="Z20" else "checkpoint_030000")
        weight=folder/("adapter_model.safetensors" if c=="OLD46" else "model.safetensors")
        digest=sha(weight)
        want=("5b0f3c3ecafae0dc1b877a1e2537fdfec09547ecc37028c2adcb3957d47b814a" if c=="OLD46" else BASE_SHA if c=="Z20" else expected[mid]["model_safetensors_sha256"])
        assert digest==want,mid
        ph,records=processor_hash(folder)
        files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob("*")) if p.is_file() and
               (p.name in ["config.json","adapter_config.json",weight.name] or p.name.startswith("policy_") or "tokenizer" in p.parts)}
        normalizer_files={k:v for k,v in files.items() if "normalizer" in k}
        assert normalizer_files,mid
        normalizer_tensor=None if c=="OLD46" else json.loads((folder/"checkpoint_audit.json").read_text())["normalizer_tensor_sha256"]
        models.append({"model_id":mid,"replicate":rep,"training_condition":c,"loader":"OLD46" if c=="OLD46" else "Z20" if c=="Z20" else "P2",
           "checkpoint":short(folder),"weights_sha256":digest,"weight_kind":"LoRA adapter on frozen base" if c=="OLD46" else "full model",
           "processor_sha256":ph,"processor_files":records,"normalizer_files":normalizer_files,
           "normalizer_tensor_sha256":normalizer_tensor,"checkpoint_files":files,
           "stats_source":"original distributed processor" if c=="OLD46" else f"P0 {rep}/N20 subset stats" if c=="Z20" else f"P0 {rep}/{c} subset stats"})
    files=["eval_paired.py","eval_p3.py","brightness_transform.py","p1_runtime.py","run_p3.py","prepare_p3.py","freeze_p3.py"]
    for name in files:ast.parse((ROOT/name).read_text(),filename=name)
    code={name:sha(ROOT/name) for name in files}
    # P1からの差分を凍結資料としてそのまま残す。
    diff="".join(difflib.unified_diff((ROOT/"eval_paired.py").read_text().splitlines(True),
           (ROOT/"eval_p3.py").read_text().splitlines(True),fromfile="eval_paired.py(P1/P2 unchanged)",tofile="eval_p3.py"))
    (P3/"_evidence/eval_p3_vs_p1.diff").write_text(diff)
    for name in files:shutil.copy2(ROOT/name,P3/"_evidence/frozen_code"/name)
    active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
    assert not active
    free=shutil.disk_usage(ROOT).free;assert free>=50*1024**3
    freeze={"version":1,"phase":"P3","frozen_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "plan_md5":"6ccbb83f0f1449bcce0f1fd94556b81b","models":models,"code_sha256":code,
       "evaluation_json_original":e,"conditions":{"indices":list(range(30,50)),"waves":{"A":list(range(30,40)),"B":list(range(40,50))},
         "brightness_factors":e["brightness_factors"],"hard_reset":True,"initial_settling_steps":10,
         "control_frequency_hz":20,"max_control_steps":280,"success_rule":e["success_rule"],
         "state_rng_base":20263010,"policy_rng_base":20264010,
         "environment_order":"ascending global index within each wave; local slots 0..9",
         "environment_seed_formula":"20263010 + global initial_state_index",
         "policy_seed_timing":"after model loading, before model.reset; identical in every wave",
         "environment_lifecycle":"new 10 worker processes for every wave; first rollout only",
         "image_processing":"same fixed k for both cameras throughout trial; clip(k*RGB,0,1) before policy preprocessing",
         "video_fps":80,"video_timebase":"one saved frame = one 20 Hz control interval; P4 must correct playback timebase",
         "completed_environment_batch_slots":"retain last observation as in P1; no automatic reset"},
       "execution_order":[{"model_id":m["model_id"],"condition":c,"wave":w,
           "indices":list(range(30,40)) if w=="A" else list(range(40,50))}
           for m in models for c in ["clean","dim","bright"] for w in ["A","B"]],
       "planned_trials":960,"pruned_init_sha256":e["init_state_file_sha256"],
       "pruned_init_verification":"full file hash checked after this freeze and before extraction/use; no pre-freeze final-state evaluation",
       "lerobot_head":head,"base_revision":BASE.name,"base_weights_sha256":BASE_SHA,
       "dataset_snapshot":pf["dataset_snapshot"],"dataset_revision":pf["dataset_revision"],"dataset_shards":shards,
       "preflight":{"compute_processes":active,"Xorg_display_excluded":True,"free_GiB":free/1024**3,
                    "p2_manifest_sha256":sha(P3/"_evidence/SHA256SUMS.before_p3")},
       "changes_from_P1_P2":["route final states30..49 as two ten-state arrays with global-index seeds",
         "route 12 final checkpoints + 3 zero-update N20 checkpoints + OLD46 original processor",
         "record required trial metadata and first model-input tensors; no RNG calls added",
         "compare hashes by global index and report mismatches without abort, as instructed",
         "freeze verification, error accounting and resource monitoring outside evaluation logic"],
       "control_logic_unchanged":["hard reset/settle10/control20Hz/max280/check_success","batch10 order and policy RNG reset",
         "preprocess->env_pre->brightness->policy_pre->select_action->policy_post->env_post",
         "active-environment stepping; retain completed slots; raw camera orientation and video encoding"],
       "technical_error_policy":{"max_retries_per_failed_trial":1,"task_failure_retry":False,
         "batch_retry":"reconstruct identical full wave to preserve policy batch RNG; accept only failed trial indices; other rows retained from attempt1",
         "wave_process_failure":"mark all ten unfinished result rows technical_error; stop threshold takes precedence over retry",
         "stop_cell_unique_error_trials":3,"stop_overall_unique_error_trials":20,
         "unresolved":"leave success null; reduce completed denominator, retain planned count"},
       "stop_rules":["any frozen asset mismatch","evaluation logic change required","GPU OOM",
         "projected evaluation over10800seconds","free disk below50GiB","technical error thresholds"],
       "representative_input_rule":"control_step0, after policy preprocessor, two exact float CHW tensors + PNG previews; capture does not consume RNG",
       "state_scope_label":e["historical_usage_review"]["default_label"],
       "training_demonstration_disjointness":"NOT_ESTABLISHED",
       "loss_interpretation":"subset-dependent normalization gives different loss scales; training loss and development0..9 are not final success evidence"}
    assert len(freeze["execution_order"])==96
    # 排他的作成で凍結の上書きを拒否する。
    with (P3/"freeze.json").open("x") as out:out.write(json.dumps(freeze,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":"FROZEN","models":len(models),"waves":96,"planned_trials":960,
                      "freeze_sha256":sha(P3/"freeze.json"),"frozen_at_utc":freeze["frozen_at_utc"]}))
if __name__=="__main__":main()

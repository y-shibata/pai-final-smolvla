"""固定SmolVLAをP0 subsetで学習する。P1は200更新で停止し全出力をdevに隔離。"""
import argparse,json,time,shutil,hashlib,datetime,math
from pathlib import Path
from p1_runtime import ROOT,SOURCE,BASE,BASE_SHA,OLD,setup,save,sha,short
setup()
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from brightness_transform import BrightnessTransform,CAMERAS
from subset_dataset import SubsetDataset

def tensor_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        t=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def normalizer_hash(pre,post):
    """保存前と別プロセス再読込後の、実際の正規化tensorを照合する。"""
    tensors={}
    for label,pipe in (("pre",pre),("post",post)):
        for step in pipe.steps:
            if step.__class__.__name__ not in ("NormalizerProcessorStep","UnnormalizerProcessorStep"):continue
            for feature,vectors in step._tensor_stats.items():
                for kind,value in vectors.items():
                    tensors[f"{label}:{feature}:{kind}"]=torch.as_tensor(value)
    assert tensors
    return tensor_hash(tensors)

def make_config(device="cuda"):
    cfg=PreTrainedConfig.from_pretrained(OLD,local_files_only=True)
    cfg.device=device;cfg.use_peft=False;cfg.pretrained_path=BASE
    cfg.pretrained_revision="c83c3163b8ca9b7e67c509fffd9121e66cb96205"
    return cfg

def check_normalizers(pre,post,stats):
    checks={}
    for label,pipe in (("pre",pre),("post",post)):
        found=False
        for step in pipe.steps:
            if step.__class__.__name__ not in ("NormalizerProcessorStep","UnnormalizerProcessorStep"):continue
            found=True
            for feature,vectors in step._tensor_stats.items():
                if feature not in stats: continue
                for kind,expected in stats[feature].items():
                    actual=torch.as_tensor(vectors[kind]).detach().cpu()
                    target=torch.as_tensor(expected,dtype=actual.dtype)
                    error=float((actual-target).abs().max())
                    assert torch.equal(actual,target),f"STOP stats override: {feature}/{kind}"
                    checks[f"{label}:{feature}:{kind}"]=error
        assert found
    return checks

def effective_state(policy,original_keys):
    """LoRAのA/BをCPUで合成し、baseと同じキー集合の全重みを保存する。"""
    raw=policy.get_base_model().state_dict()
    modules=dict(policy.get_base_model().named_modules())
    out={}
    for key in original_keys:
        if key in raw: out[key]=raw[key].detach().cpu().contiguous().clone();continue
        stem,kind=key.rsplit(".",1); module=modules[stem]
        if kind=="bias":
            out[key]=module.base_layer.bias.detach().cpu().contiguous().clone();continue
        assert kind=="weight",key
        weight=module.base_layer.weight.detach().cpu().clone()
        weight+=module.get_delta_weight("default").detach().cpu().to(weight.dtype)
        out[key]=weight.contiguous()
    return out

def checkpoint(policy,cfg,pre,post,stats,out,step,original_keys,base_tensor_hash,started=None):
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cp=out/f"checkpoint_{step:06d}";cp.mkdir()
    effective=effective_state(policy,original_keys)
    digest=tensor_hash(effective)
    cfg.use_peft=False;cfg.save_pretrained(cp)
    if step==0:
        assert digest==base_tensor_hash,"Update-0 effective weights changed"
        shutil.copyfile(BASE/"model.safetensors",cp/"model.safetensors")
    else: save_file(effective,cp/"model.safetensors")
    pre.save_pretrained(cp);post.save_pretrained(cp)
    save(cp/"subset_stats.json",stats)
    policy.save_pretrained(cp/"adapter")
    checks=check_normalizers(pre,post,stats)
    record={"step":step,"checkpoint":short(cp),"model_file_sha256":sha(cp/"model.safetensors"),
      "effective_tensor_sha256":digest,"base_tensor_sha256":base_tensor_hash,
      "normalizer_max_abs_errors_vs_stats_cast":checks,
      "normalizer_tensor_sha256":normalizer_hash(pre,post),
      "elapsed_seconds":None if started is None else time.perf_counter()-started,
      "saved_at_utc":utc_now()}
    save(cp/"checkpoint_audit.json",record);cfg.use_peft=True
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
    return record

def reload_checkpoint(path,device):
    cp=Path(path).expanduser()
    if not cp.is_absolute(): cp=ROOT/cp
    assert cp.is_dir(),cp
    cfg=PreTrainedConfig.from_pretrained(cp,local_files_only=True)
    cfg.device=device
    model=SmolVLAPolicy.from_pretrained(cp,config=cfg,local_files_only=True,strict=True)
    pre,post=make_pre_post_processors(cfg,pretrained_path=cp,
       preprocessor_overrides={"device_processor":{"device":device}},
       postprocessor_overrides={"device_processor":{"device":"cpu"}})
    stats=json.loads((cp/"subset_stats.json").read_text())
    checks=check_normalizers(pre,post,stats)
    expected=json.loads((cp/"checkpoint_audit.json").read_text())
    actual=tensor_hash(model.state_dict())
    assert actual==expected["effective_tensor_sha256"],"STOP standalone weight reload mismatch"
    assert sha(cp/"model.safetensors")==expected["model_file_sha256"]
    normalizer_digest=normalizer_hash(pre,post)
    if "normalizer_tensor_sha256" in expected:
        assert normalizer_digest==expected["normalizer_tensor_sha256"],"STOP standalone normalizer hash mismatch"
    result={"normalizer_tensor_sha256":normalizer_digest,"verified_at_utc":utc_now(),
            "status":"PASS","same_weights":True,"same_normalizers":True,"tensor_sha256":actual,
            "normalizer_max_abs_errors":checks}
    save(cp/"standalone_reload.json",result)
    print(json.dumps(result))
    return model,pre,post

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--condition",choices=["N10","N20","N46","N20B"],default="N20")
    parser.add_argument("--scope",choices=["p1","p2"],default="p1")
    parser.add_argument("--replicate",choices=["R1","R2","R3"],default="R1")
    parser.add_argument("--updates",type=int,default=200)
    parser.add_argument("--reload");parser.add_argument("--device",default="cuda")
    parser.add_argument("--audit-loader",action="store_true")
    a=parser.parse_args()
    if a.reload: reload_checkpoint(a.reload,a.device);return
    assert a.updates==(200 if a.scope=="p1" else 30000)
    assert a.scope=="p2" or a.replicate=="R1"
    # 組だけで決まる固定seedを使う。成績を見たseed変更や継続学習は行わない。
    replicate_index=int(a.replicate[1:])-1
    training_seed=20261011+replicate_index
    augmentation_seed=20262011+replicate_index
    subset=ROOT/"subsets"/a.replicate/a.condition
    out=ROOT/"dev"/f"train_smoke_R1{a.condition}" if a.scope=="p1" else ROOT/"p2"/a.replicate/a.condition
    if a.scope=="p2":
        assert not (out/"run_config.json").exists(),"Do not overwrite an existing training run"
        assert (out/"preflight.json").exists(),"P2 preflight must be saved before launch"
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    set_seed(training_seed)
    started=time.perf_counter();started_at=utc_now()
    dataset=SubsetDataset(subset)
    # 学習と同じ__getitem__で1epoch全フレームを検査し、選択外0件を確かめる。
    audit_loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4)
    ids=[]; frames=0
    for batch in audit_loader:
        ids.extend(batch["episode_index"].tolist());frames+=len(batch["episode_index"])
    expected=dataset.selection["episode_ids"]
    extras=sorted(set(ids)-set(expected));assert not extras and sorted(set(ids))==sorted(expected)
    save(out/"loader_audit.json",{"frames":frames,"expected_episode_ids":expected,
      "seen_episode_ids_one_epoch":sorted(set(ids)),"selected_outside_count":len(extras),
      "epoch_frame_counts":{str(i):ids.count(i) for i in sorted(set(ids))}})
    if a.audit_loader: print("Loader PASS",frames,"frames");return
    assert sha(BASE/"model.safetensors")==BASE_SHA
    stats=json.loads((subset/"stats.json").read_text())
    # CPU loader検査が乱数を消費しても、学習開始seedは必ず同じ値に戻す。
    set_seed(training_seed)
    cfg=make_config(a.device)
    assert (cfg.chunk_size,cfg.n_action_steps)==(50,50)
    assert cfg.optimizer_lr==.001 and cfg.scheduler_warmup_steps==1000
    assert cfg.scheduler_decay_steps==30000 and cfg.scheduler_decay_lr==.0001
    assert str(cfg.normalization_mapping["VISUAL"]).endswith("IDENTITY")
    assert tuple(cfg.optimizer_betas)==(.9,.95) and cfg.optimizer_eps==1e-8
    assert cfg.optimizer_weight_decay==1e-10 and cfg.optimizer_grad_clip_norm==10
    torch.cuda.reset_peak_memory_stats()
    policy=SmolVLAPolicy.from_pretrained(BASE,config=cfg,local_files_only=True,strict=True)
    original_keys=list(policy.state_dict())
    base_tensor_hash=tensor_hash(policy.state_dict())
    pre,post=make_smolvla_pre_post_processors(cfg,stats)
    check_normalizers(pre,post,stats)
    policy=policy.wrap_with_peft(peft_cli_overrides={"method_type":"LORA","r":16,"lora_alpha":16})
    records=[checkpoint(policy,cfg,pre,post,stats,out,0,original_keys,base_tensor_hash,started)]
    optimizer=cfg.get_optimizer_preset().build(policy.parameters())
    # 実際の停止点が200でも、schedulerには本学習の30,000を指定。短縮補正を避ける。
    scheduler=cfg.get_scheduler_preset().build(optimizer,num_training_steps=30000)
    policy.train()
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=4,
       pin_memory=True,drop_last=False,generator=torch.Generator().manual_seed(training_seed))
    iterator=iter(loader)
    transform=BrightnessTransform(seed=augmentation_seed) if a.condition=="N20B" else None
    config={"scope":a.scope,"updates_requested":a.updates,"replicate":a.replicate,"selection_seed":101*(replicate_index+1),
      "training_seed":training_seed,"augmentation_seed":augmentation_seed,"started_at_utc":started_at,
      "condition":a.condition,"selection_sha256":sha(subset/"selection.json"),"stats_sha256":sha(subset/"stats.json"),
      "base_file_sha256":BASE_SHA,"batch_size":1,"num_workers":4,"lora_r":16,"lora_alpha":16,
      "optimizer":{"lr":.001,"betas":[.9,.95],"eps":1e-8,"weight_decay":1e-10,"grad_clip_norm":10},
      "scheduler":{"warmup":1000,"decay_steps":30000,"peak_lr":.001,"decay_lr":.0001},
      "chunk_size":50,"n_action_steps":50,"brightness_enabled":bool(transform)}
    save(out/"run_config.json",config)
    times=[];steps=[];augmentation=[]
    # 1更新ずつディスクへ記録するので、中断しても損失と学習率を失わない。
    live_log=(out/"steps.jsonl").open("x",buffering=1)
    for update in range(1,a.updates+1):
        torch.cuda.synchronize(); t=time.perf_counter()
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        for key in CAMERAS:batch[key]=batch[key].float()/255.
        if transform:
            batch,record=transform(batch);augmentation.append({"update":update,**record})
        batch=pre(batch)
        optimizer.zero_grad(set_to_none=True)
        loss,details=policy(batch)
        if not torch.isfinite(loss).all():
            save(out/"stop_reason.json",{"reason":"LOSS_NAN_OR_INF","update":update,"at_utc":utc_now()})
            raise RuntimeError("STOP: loss became NaN/inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),10)
        optimizer.step();scheduler.step()
        torch.cuda.synchronize();elapsed=time.perf_counter()-t;times.append(elapsed)
        row={"update":update,"loss":float(loss),"seconds":elapsed,"lr":scheduler.get_last_lr()[0],
             "elapsed_seconds":time.perf_counter()-started,"at_utc":utc_now(),
             "max_cuda_allocated_bytes":torch.cuda.max_memory_allocated()}
        steps.append(row);live_log.write(json.dumps(row,allow_nan=False)+"\n")
        if update%20==0:
            print(json.dumps(row),flush=True)
            save(out/"progress.json",row)
        if time.perf_counter()-started>5400:raise RuntimeError("STOP: run exceeded 90 minutes")
        if update in (3000,12000,a.updates):
            records.append(checkpoint(policy,cfg,pre,post,stats,out,update,original_keys,base_tensor_hash,started))
    live_log.close()
    estimate=float(np.mean(times))*30000
    steady=float(np.mean(times[10:]))*30000
    result={"status":"PASS" if estimate<=5400 else "STOP_ESTIMATE_OVER_90_MIN",
      "started_at_utc":started_at,"finished_at_utc":utc_now(),"final_loss":steps[-1]["loss"],
      "completed_updates":a.updates,"replicate":a.replicate,"condition":a.condition,
      "loss_first_100_mean":float(np.mean([x["loss"] for x in steps[:100]])),
      "loss_last_100_mean":float(np.mean([x["loss"] for x in steps[-100:]])),
      "checkpoint_records":records,"seconds_per_update_mean":float(np.mean(times)),
      "seconds_per_update_after_first_10":float(np.mean(times[10:])),
      "estimated_30000_seconds":estimate,"estimated_30000_steady_seconds":steady,
      "historical_30000_seconds":2065,"elapsed_total_seconds":time.perf_counter()-started,
      "max_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),
      "max_cuda_reserved_bytes":torch.cuda.max_memory_reserved()}
    save(out/"steps.json",steps);save(out/"augmentation.json",augmentation);save(out/"result.json",result)
    print(json.dumps(result),flush=True)
    assert estimate<=5400,"STOP: 30,000 update estimate exceeds 90 minutes"
if __name__=="__main__":main()

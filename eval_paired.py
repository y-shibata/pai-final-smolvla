"""開発用0〜9だけを使うpaired評価。各モデル×画像条件で10環境を新規生成する。"""
import argparse,hashlib,json,time,traceback,multiprocessing as mp,subprocess
from pathlib import Path
import numpy as np
from p1_runtime import ROOT,SOURCE,BASE,OLD,BASE_SHA,setup,save,sha,short
FACTORS={"clean":1.0,"dim":0.65,"bright":1.35}
def array_hash(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def stack_obs(items):
    if isinstance(items[0],dict):return {k:stack_obs([x[k] for x in items]) for k in items[0]}
    return np.stack(items)
def write_video(path,frames,fps=80):
    h,w=frames[0].shape[:2]
    proc=subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo",
       "-pix_fmt","rgb24","-s",f"{w}x{h}","-r",str(fps),"-i","pipe:0","-an",
       "-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p",str(path)],stdin=subprocess.PIPE)
    for frame in frames:proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close();assert proc.wait()==0
def worker(conn,index):
    env=None
    try:
        setup()
        import random
        # robosuiteの固定ログ名だけを自分の一時領域へ向ける。
        # 依存ライブラリのファイルは変更せず、import時だけFileHandlerを包む。
        import logging,os
        original_handler=logging.FileHandler
        class WorkspaceLogHandler(original_handler):
            def __init__(self,filename,*args,**kwargs):
                if str(filename)=="/tmp/robosuite.log":
                    filename=ROOT/f"tmp/robosuite_{os.getpid()}.log"
                super().__init__(filename,*args,**kwargs)
        logging.FileHandler=WorkspaceLogHandler
        try:
            from lerobot.envs.libero import LiberoEnv,_get_suite
        finally:
            logging.FileHandler=original_handler
        # 環境構築時に使う乱数も固定。モデルの乱数とは別プロセスである。
        seed=20263010+index;random.seed(seed);np.random.seed(seed)
        states=np.load(ROOT/"dev/init_states_0_9.npy",allow_pickle=False)
        assert states.shape==(10,92) and 0<=index<10
        env=LiberoEnv(_get_suite("libero_spatial"),task_id=2,task_suite_name="libero_spatial",
          obs_type="pixels_agent_pos",init_states=False,episode_index=index,n_envs=10,
          hard_reset=True,num_steps_wait=10,control_freq=20,episode_length=280)
        # 元ファイルを読む既存constructor分岐を通らず、隔離した10件だけを渡す。
        env.init_states=True;env._init_states=states
        obs,_=env.reset(seed=seed)
        audit={"initial_state_index":index,"state_seed":seed,"raw_initial_state_sha256":array_hash(states[index]),
          "post_settle_state_sha256":array_hash(env._env.env.sim.get_state().flatten()),
          "raw_image_sha256":{k:array_hash(v) for k,v in obs["pixels"].items()},
          "post_settle_simulation_time":float(env._env.env.sim.data.time)}
        conn.send(("ready",obs,audit))
        step=0;done=False
        while True:
            command,data=conn.recv()
            if command=="close":break
            assert command=="step" and not done and step<280
            obs,reward,terminated,truncated,info=env.step(data);step+=1
            done=bool(terminated or truncated or step>=280)
            conn.send(("step",obs,done,bool(info["is_success"]),float(env._env.env.sim.data.time)))
    except BaseException:conn.send(("error",traceback.format_exc()))
    finally:
        if env is not None:env.close()
        conn.close()
def receive(conn):
    data=conn.recv()
    if data[0]=="error":raise RuntimeError(data[1])
    return data
def processor_hash(folder):
    records=[]
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and (p.name.startswith("policy_") or "tokenizer" in p.parts):
            records.append([str(p.relative_to(folder)),sha(p)])
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),records
def load_policy(which,checkpoint_path=None):
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    if which=="OLD46":
        from peft import PeftModel
        cfg=PreTrainedConfig.from_pretrained(OLD,local_files_only=True)
        cfg.device="cuda";cfg.use_peft=False;cfg.pretrained_path=BASE
        model=SmolVLAPolicy.from_pretrained(BASE,config=cfg,local_files_only=True,strict=True)
        model=PeftModel.from_pretrained(model,OLD,is_trainable=False,local_files_only=True)
        folder=OLD;weight_path=OLD/"adapter_model.safetensors"
    else:
        step=0 if which=="Z20" else 200
        folder=Path(checkpoint_path) if checkpoint_path else ROOT/f"dev/train_smoke_R1N20/checkpoint_{step:06d}"
        cfg=PreTrainedConfig.from_pretrained(folder,local_files_only=True)
        cfg.device="cuda"
        model=SmolVLAPolicy.from_pretrained(folder,config=cfg,local_files_only=True,strict=True)
        weight_path=folder/"model.safetensors"
    # OLD46は保存済みの元前処理をそのまま使う。subset statsへの置換はしない。
    pre,post=make_pre_post_processors(cfg,pretrained_path=folder)
    digest,records=processor_hash(folder)
    model.eval()
    return model,pre,post,{"weights_sha256":sha(weight_path),"base_weights_sha256":BASE_SHA,
       "processor_sha256":digest,"processor_files":records,"checkpoint":short(folder)},cfg

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",choices=["OLD46","Z20","N20_200","P2"],required=True)
    p.add_argument("--condition",choices=list(FACTORS),required=True);p.add_argument("--repeat",type=int,default=0)
    p.add_argument("--replicate",choices=["R1","R2","R3"],default="R1")
    p.add_argument("--training-condition",choices=["N10","N20","N46","N20B"])
    a=p.parse_args();setup()
    checkpoint_path=None
    if a.model=="P2":
        assert a.condition=="clean" and a.repeat==0 and a.training_condition
        checkpoint_path=ROOT/"p2"/a.replicate/a.training_condition/"checkpoint_030000"
        assert json.loads((checkpoint_path/"standalone_reload.json").read_text())["status"]=="PASS"
        assert json.loads((checkpoint_path/"checkpoint_audit.json").read_text())["step"]==30000
    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoConfig
    from lerobot.envs.utils import preprocess_observation
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.utils.random_utils import set_seed
    from brightness_transform import BrightnessTransform
    torch.set_num_threads(4)
    wave=f"{a.model}_{a.condition}_repeat{a.repeat}"
    out=ROOT/"dev/eval"/wave if a.model!="P2" else ROOT/"p2"/a.replicate/a.training_condition/"devcheck"
    out.mkdir(parents=True,exist_ok=False)
    (out/"videos").mkdir()
    # 環境ごとにプロセスを作り、各プロセスで最初のrolloutだけを使用する。
    ctx=mp.get_context("spawn");children=[];pipes=[]
    t=time.perf_counter();start=t
    for index in range(10):
        parent,child=ctx.Pipe();proc=ctx.Process(target=worker,args=(child,index))
        proc.start();child.close();children.append(proc);pipes.append(parent)
    try:
        ready=[receive(c) for c in pipes]
        obs=[x[1] for x in ready];audits=[x[2] for x in ready]
        env_seconds=time.perf_counter()-t
        save(out/"initial_audit.json",audits)
        # 過去の開発waveと同じseed・同じindexなら、状態と元画像が完全一致するか確認。
        comparisons=[]
        for previous in sorted((ROOT/"dev/eval").glob("*/initial_audit.json")):
            if previous.parent==out:continue
            old=json.loads(previous.read_text())
            for before,now in zip(old,audits,strict=True):
                for key in ("raw_initial_state_sha256","post_settle_state_sha256","raw_image_sha256"):
                    matched=before[key]==now[key]
                    comparisons.append({"reference":short(previous),"index":now["initial_state_index"],"key":key,"equal":matched})
        save(out/"paired_hash_checks.json",comparisons)
        assert all(x["equal"] for x in comparisons),"STOP: paired initial or post-settle hash mismatch"
        torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter();model,pre,post,model_info,cfg=load_policy(a.model,checkpoint_path)
        model_seconds=time.perf_counter()-t
        env_pre,env_post=make_env_pre_post_processors(
            LiberoConfig(task="libero_spatial",task_ids=[2],obs_type="pixels_agent_pos"),cfg)
        # モデル生成が乱数を消費した後でpolicy seedを戻す。全waveで同一。
        set_seed(20264010);model.reset()
        transform=BrightnessTransform(fixed_k=FACTORS[a.condition])
        done=[False]*10;success=[False]*10;counts=[0]*10
        frames=[[] for _ in range(10)];wrist=[[] for _ in range(10)];frame_times=[[] for _ in range(10)]
        terminal=[None]*10;clipping=[]
        t=time.perf_counter()
        with torch.inference_mode():
            for control_step in range(280):
                # 1フレームはこの後の1制御区間(1/20秒)に対応。終端画像は別ファイルにする。
                for i in range(10):
                    if done[i]:continue
                    frames[i].append(obs[i]["pixels"]["image"][::-1,::-1].copy())
                    wrist[i].append(obs[i]["pixels"]["image2"][::-1,::-1].copy())
                    frame_times[i].append({"frame":counts[i],"control_step":counts[i],
                        "simulation_time_seconds":counts[i]/20,"saved_fps":80,
                        "mujoco_time_seconds":audits[i]["post_settle_simulation_time"]+counts[i]/20})
                batch=preprocess_observation(stack_obs(obs))
                batch["task"]=["pick up the black bowl from table center and place it on the plate"]*10
                batch=env_pre(batch);batch,clip=transform(batch);clipping.append({"control_step":control_step,**clip})
                action=post(model.select_action(pre(batch)))
                action=env_post({"action":action})["action"].cpu().numpy()
                for i,c in enumerate(pipes):
                    if not done[i]:c.send(("step",action[i]))
                for i,c in enumerate(pipes):
                    if done[i]:continue
                    message=receive(c);obs[i]=message[1];counts[i]+=1
                    done[i]=message[2];success[i]=message[3]
                    assert abs(message[4]-audits[i]["post_settle_simulation_time"]-counts[i]/20)<1e-8
                    if done[i]:terminal[i]=obs[i]["pixels"]["image"][::-1,::-1].copy()
                if all(done):break
        torch.cuda.synchronize();inference_seconds=time.perf_counter()-t
        t=time.perf_counter()
        from PIL import Image
        rows=[]
        for i in range(10):
            path=out/f"videos/state_{i:02d}_agentview_raw80.mp4"
            hand=out/f"videos/state_{i:02d}_wrist_raw80.mp4"
            write_video(path,frames[i]);write_video(hand,wrist[i])
            Image.fromarray(terminal[i]).save(out/f"videos/state_{i:02d}_terminal.png")
            save(out/f"videos/state_{i:02d}_frames.json",frame_times[i])
            rows.append({"run_id":a.model if a.model!="P2" else a.replicate+a.training_condition,"condition":a.condition,"replicate":a.replicate,"repeat":a.repeat,
              "initial_state_index":i,"brightness_factor":FACTORS[a.condition],"success":success[i],
              "control_steps":counts[i],"weights_sha256":model_info["weights_sha256"],
              "processor_sha256":model_info["processor_sha256"],"video_path":short(path),"wrist_video_path":short(hand),
              "frame_metadata_path":short(out/f"videos/state_{i:02d}_frames.json"),
              "policy_seed":20264010,"state_seed":20263010+i,"scope":"DEVELOPMENT_ONLY"})
        video_seconds=time.perf_counter()-t
        timing={"env_build_and_reset_seconds":env_seconds,"model_load_seconds":model_seconds,
           "inference_and_simulation_seconds":inference_seconds,"video_save_seconds":video_seconds,
           "total_seconds":time.perf_counter()-start,"max_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),
           "max_cuda_reserved_bytes":torch.cuda.max_memory_reserved()}
        save(out/"results.json",rows);save(out/"clipping.json",clipping)
        save(out/"model_info.json",model_info);save(out/"timing.json",timing)
        print(json.dumps({"wave":wave,"completed":10,"successes":sum(success),"timing":timing}),flush=True)
    finally:
        for c in pipes:
            try:c.send(("close",None))
            except (BrokenPipeError,EOFError):pass
        for proc in children:
            proc.join(timeout=10)
            if proc.is_alive():proc.terminate();proc.join(timeout=5)
if __name__=="__main__":main()

"""各P1処理を30分で停止し、時間とGPUメモリを記録する起動wrapper。"""
from pathlib import Path
import argparse,subprocess,time,os,signal,json
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument("--name",required=True);p.add_argument("--cpu",action="store_true")
    p.add_argument("--allow-xorg",action="store_true");p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args();cmd=a.command
    if cmd and cmd[0]=="--":cmd=cmd[1:]
    folder=ROOT/"dev/evidence";folder.mkdir(exist_ok=True)
    # GPU開始前には計算プロセスを検査。Xorg例外は明示指定がある時だけ許可する。
    gpu_snapshot=subprocess.check_output(["nvidia-smi"],text=True)
    if not a.cpu:
        active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
        assert not active,"GPU already has compute processes"
        assert a.allow_xorg,"Xorg exception requires explicit authorization"
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TMPDIR=str(ROOT/"tmp"))
    if a.cpu:env["CUDA_VISIBLE_DEVICES"]=""
    started=time.monotonic();peak=0;timed_out=False
    with (folder/(a.name+".log")).open("w") as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            if time.monotonic()-started>=1800:
                timed_out=True;os.killpg(proc.pid,signal.SIGTERM);time.sleep(2)
                if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL)
                break
            if not a.cpu:
                text=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
                peak=max(peak,int(text.strip().splitlines()[0]))
            time.sleep(.5)
        rc=proc.wait()
    record={"command":cmd,"elapsed_seconds":time.monotonic()-started,"peak_total_gpu_memory_MiB":peak,
      "cpu_only":a.cpu,"timed_out":timed_out,"returncode":rc,"gpu_before":gpu_snapshot,
      "allow_xorg":a.allow_xorg,"limit_seconds":1800}
    (folder/(a.name+".timing.json")).write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k not in ("gpu_before","command")}),flush=True)
    raise SystemExit(rc or (124 if timed_out else 0))
if __name__=="__main__":main()

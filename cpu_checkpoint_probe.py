"""GPUを使わず、更新0回の保存・単独再読込のコード経路を検証する。"""
from train_experiment import *
def main():
    torch.set_num_threads(4);set_seed(20261011)
    out=ROOT/"dev/cpu_checkpoint_probe";out.mkdir(exist_ok=True)
    stats=json.loads((ROOT/"subsets/R1/N20/stats.json").read_text())
    cfg=make_config("cpu")
    policy=SmolVLAPolicy.from_pretrained(BASE,config=cfg,local_files_only=True,strict=True)
    keys=list(policy.state_dict());digest=tensor_hash(policy.state_dict())
    pre,post=make_smolvla_pre_post_processors(cfg,stats)
    policy=policy.wrap_with_peft(peft_cli_overrides={"method_type":"LORA","r":16,"lora_alpha":16})
    record=checkpoint(policy,cfg,pre,post,stats,out,0,keys,digest)
    print(json.dumps(record))
if __name__=="__main__":main()

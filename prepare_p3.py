"""凍結保存後にだけ最終初期状態を抽出する。pickleコードは実行しない。"""
from pathlib import Path
import datetime,struct,json,hashlib
from p1_runtime import ROOT,sha,save,short
def main():
    freeze=json.loads((ROOT/"p3/freeze.json").read_text())
    for name,digest in freeze["code_sha256"].items():assert sha(ROOT/name)==digest
    out=ROOT/"p3/init_states_30_49.npy";assert not out.exists()
    p0=json.loads((ROOT/"evidence/initial_state_check.json").read_text())
    source=Path(p0["path"]).expanduser()
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    assert started>freeze["frozen_at_utc"]
    actual=sha(source);assert actual==freeze["pruned_init_sha256"]==p0["sha256"]
    with source.open("rb",buffering=0) as f:
        header=f.read(30);assert header[:4]==b"PK\x03\x04"
        values=struct.unpack("<4s5H3I2H",header);assert values[3]==0
        name=f.read(values[-2]);extra=f.read(values[-1]);assert name==b"archive/data.pkl"
        prefix=f.read(200);marker=b"\x89h\x03X";offset=prefix.index(marker)+len(marker)
        length=struct.unpack("<I",prefix[offset:offset+4])[0]
        assert b"K2K\\\x86" in prefix and b"f8" in prefix and length>36800
        start=30+values[-2]+values[-1]+offset+4;f.seek(start)
        raw=bytearray()
        # 原ファイルのhashは全体で照合した。ここで保持する配列は30〜49だけ。
        for i in range(50*92*8):
            b=f.read(1)
            if b[0]>=128:
                assert 0xC2<=b[0]<=0xC3
                b=bytes([ord((b+f.read(1)).decode("utf8"))])
            if i>=30*92*8:raw+=b
        end=f.tell()
    import numpy as np
    states=np.frombuffer(raw,dtype="<f8").reshape(20,92).copy()
    hashes=[hashlib.sha256(row.tobytes()).hexdigest() for row in states]
    assert hashes==p0["row_sha256"][30:50]
    np.save(out,states,allow_pickle=False)
    save(ROOT/"p3/_evidence/initial_state_extraction.json",{
       "status":"PASS","started_at_utc":started,"finished_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "freeze_sha256":sha(ROOT/"p3/freeze.json"),"source":short(source),"source_sha256":actual,
       "saved_indices":list(range(30,50)),"shape":[20,92],"row_sha256":hashes,
       "saved_array_sha256":sha(out),"source_payload_byte_range":[start,end],
       "method":"known protocol2 byte decoding; no unpickling; retain rows30..49 only",
       "source_full_file_read_for_sha256_after_freeze":True})
    print(json.dumps({"status":"PASS","freeze_sha256":sha(ROOT/"p3/freeze.json"),"saved_state_sha256":sha(out)}))
if __name__=="__main__":main()

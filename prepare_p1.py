"""初期状態0〜9のバイトだけを読み、P0の行hashと照合して隔離保存する。"""
from pathlib import Path
import struct,hashlib,json,shutil
from p1_runtime import ROOT, SOURCE, save, short
def main():
    (ROOT/"dev/evidence").mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/"SHA256SUMS", ROOT/"dev/evidence/P0_SHA256SUMS.before_p1")
    p0=json.loads((ROOT/"evidence/initial_state_check.json").read_text())
    source=Path(p0["path"]).expanduser()
    reads=[]
    with source.open("rb",buffering=0) as f:
        def read(n):
            start=f.tell(); data=f.read(n); reads.append([start,len(data)]); return data
        header=read(30)
        assert header[:4]==b"PK\x03\x04"
        values=struct.unpack("<4s5H3I2H",header)
        assert values[3]==0, "Compressed payload is not accepted"
        name=read(values[-2]); extra=read(values[-1])
        assert name==b"archive/data.pkl"
        prefix=read(200)
        # 既知ファイルのNumPy pickle構造だけを許可。巨大配列のデシリアライズはしない。
        marker=b"\x89h\x03X"; offset=prefix.index(marker)+len(marker)
        length=struct.unpack("<I",prefix[offset:offset+4])[0]
        assert b"K2K\\\x86" in prefix and b"f8" in prefix and length>36800
        payload_start=30+values[-2]+values[-1]+offset+4
        f.seek(payload_start)
        raw=bytearray()
        # protocol 2はLatin-1の各バイトをUTF-8で格納。10*92*8バイトで必ず終了。
        for _ in range(10*92*8):
            b=read(1)
            if b[0]<128: raw+=b
            else:
                assert 0xC2<=b[0]<=0xC3
                raw+=bytes([ord((b+read(1)).decode("utf8"))])
        end=f.tell()
    import numpy as np
    states=np.frombuffer(raw,dtype="<f8").reshape(10,92).copy()
    hashes=[hashlib.sha256(row.tobytes()).hexdigest() for row in states]
    assert hashes==p0["row_sha256"][:10]
    np.save(ROOT/"dev/init_states_0_9.npy",states,allow_pickle=False)
    # 各1バイト読込を連続範囲へまとめ、実際に触れた範囲を監査できるようにする。
    save(ROOT/"dev/evidence/initial_state_extraction.json", {
      "status":"PASS","source":short(source),"source_full_hash_not_recomputed_in_P1":True,
      "indices_read":list(range(10)),"shape":list(states.shape),"row_sha256":hashes,
      "byte_ranges_read_including_overlap":[[0,payload_start],[payload_start,end]],
      "payload_bytes_decoded":len(raw),"last_file_offset_read_exclusive":end,
      "states_10_through_49_not_read":True,"method":"unbuffered prefix extraction; no torch.load or ZIP whole-file read"})
    print("P1 initial state extraction PASS; only indices 0..9")
if __name__=="__main__":main()

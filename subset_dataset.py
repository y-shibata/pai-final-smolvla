"""P0の選択を使うDataset。actionの50ステップと末尾paddingをepisode内で作る。"""
import io,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import pyarrow.dataset as ds
from brightness_transform import CAMERAS
class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.selection=json.loads((Path(subset)/"selection.json").read_text())
        self.rows=[]; self.end_indices=[]
        # Arrowのepisode predicateを指定し、選択行だけをPythonへ返す。
        for ep in self.selection["episodes"]:
            table=ds.dataset(Path(ep["source_shard"]).expanduser(),format="parquet").to_table(
                filter=ds.field("episode_index")==ep["episode_index"])
            rows=table.to_pylist()
            rows.sort(key=lambda r:r["frame_index"])
            assert len(rows)==ep["frame_count"]
            assert {r["episode_index"] for r in rows}=={ep["episode_index"]}
            assert [r["frame_index"] for r in rows]==list(range(len(rows)))
            end=len(self.rows)+len(rows)-1
            self.rows.extend(rows); self.end_indices.extend([end]*len(rows))
        assert len(self.rows)==self.selection["frame_count"]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; end=self.end_indices[i]
        action=torch.tensor(np.stack([self.rows[min(i+j,end)]["action"] for j in range(50)]),dtype=torch.float32)
        out={"action":action,"action_is_pad":torch.arange(50)+i>end,
             "observation.state":torch.tensor(row["observation.state"],dtype=torch.float32),
             "task":self.selection["task"],"episode_index":int(row["episode_index"])}
        for k in CAMERAS:
            # コピーを作るため、元画像バイトも元の配列も変わらない。
            image=np.array(Image.open(io.BytesIO(row[k]["bytes"])).convert("RGB"),copy=True)
            out[k]=torch.from_numpy(image).permute(2,0,1)
        return out

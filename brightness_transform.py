"""モデル前処理の直前に使う明るさwrapper。元データやラベルは変更しない。"""
import torch
CAMERAS=("observation.images.image","observation.images.image2")
def clipping_fractions(scaled):
    """RGB成分単位と、いずれかの成分が飽和した画素単位の両方を返す。"""
    clipped=(scaled<0)|(scaled>1)
    return clipped.float().mean(dim=(1,2,3)).tolist(), clipped.any(dim=1).float().mean(dim=(1,2)).tolist()

class BrightnessTransform:
    def __init__(self, seed=20262011, fixed_k=None):
        self.rng=torch.Generator(device="cpu").manual_seed(seed)
        self.fixed_k=fixed_k
    def __call__(self,batch):
        out=dict(batch)
        first=batch[CAMERAS[0]]
        assert first.ndim==4 and first.is_floating_point()
        n=first.shape[0]
        k=(torch.full((n,),float(self.fixed_k)) if self.fixed_k is not None
           else torch.rand(n,generator=self.rng)*0.6+0.7)
        ratios={};pixels={}
        for key in CAMERAS:
            src=batch[key]; assert src.ndim==4 and src.shape[0]==n
            assert bool((src>=0).all() and (src<=1).all())
            scaled=src*k.to(src.device).view(n,1,1,1)
            # 飽和率はクリップが値を変えた画素成分の割合。元から0/1の画素と区別する。
            ratios[key],pixels[key]=clipping_fractions(scaled)
            out[key]=scaled.clamp(0,1)
        return out, {"k":k.tolist(),"clipped_fraction":ratios,"clipped_pixel_fraction":pixels}

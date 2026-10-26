"""walkscene.background with the near plane at 8 m: the orbit flies 13 m up, where floaters hang near the camera."""
import sys
import numpy as np
import sim2chip.walkscene.background as b


def render(sp, w2cs, K, w, h):
    import torch
    from gsplat import rasterization
    dev = sp["means"].device
    m = torch.as_tensor(np.asarray(w2cs), dtype=torch.float32, device=dev)
    k = torch.as_tensor(K, dtype=torch.float32, device=dev)[None].repeat(len(m), 1, 1)
    with torch.no_grad():
        out, alpha, _ = rasterization(**sp, viewmats=m, Ks=k, width=w, height=h, sh_degree=0,
                                      render_mode="RGB+ED", near_plane=8.0, far_plane=500, packed=False)
    return ((out[..., :3].clamp(0, 1) * 255).byte().cpu().numpy(), out[..., 3].cpu().numpy(),
            alpha[..., 0].cpu().numpy())


b.render = render
b.main()

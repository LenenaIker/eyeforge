"""Peak decoding of count grids (experiment): one point per local maximum."""
import numpy as np


def peaks(g, labels, th):
    """One point per cell that reaches th and is the maximum of its 3x3 neighbourhood (ties all kept)."""
    p = 1 - g[..., 0]
    pad = np.pad(p, 1, constant_values=-1)
    win = np.stack([pad[1 + dy:1 + dy + p.shape[0], 1 + dx:1 + dx + p.shape[1]]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    keep = (p >= th) & (p >= win.max(0))
    rows, cols = p.shape
    cls = g[..., 1:].argmax(-1) + 1
    return [((j + .5) / cols, (i + .5) / rows, labels[cls[i, j]]) for i, j in zip(*np.nonzero(keep))]


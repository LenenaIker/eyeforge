// Pillow's bilinear resize (Image.resize(..., BILINEAR)), which ingest uses on the training and test photos: a
// triangle filter whose support grows with the reduction, horizontal pass then vertical, each rounded to 8 bits.
// The browser's own canvas scaling samples differently and moved some counts, so the live test uses this.
// Checked against Pillow in resize.test.ts (fixtures from scripts/decode_fixtures.py).

type Coeffs = { start: Int32Array; size: Int32Array; w: Float64Array; ksize: number };

function coeffs(inSize: number, outSize: number): Coeffs {
  const scale = inSize / outSize;
  const fscale = Math.max(scale, 1);
  const support = fscale;  // bilinear filter support 1.0, times the scale when reducing
  const ksize = Math.ceil(support) * 2 + 1;
  const start = new Int32Array(outSize), size = new Int32Array(outSize), w = new Float64Array(outSize * ksize);
  for (let x = 0; x < outSize; x++) {
    const center = (x + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let total = 0;
    for (let i = 0; i < xmax; i++) {
      const t = Math.abs((i + xmin - center + 0.5) / fscale);
      const v = t < 1 ? 1 - t : 0;
      w[x * ksize + i] = v;
      total += v;
    }
    for (let i = 0; i < xmax; i++) if (total) w[x * ksize + i] /= total;
    start[x] = xmin; size[x] = xmax;
  }
  return { start, size, w, ksize };
}

// Pillow rounds with fixed-point coefficients (22 bits) and clips to 0..255 after each pass.
const PREC = 1 << 22;
function fixed(c: Coeffs): Int32Array {
  const k = new Int32Array(c.w.length);
  for (let i = 0; i < k.length; i++) k[i] = Math.trunc(c.w[i] < 0 ? c.w[i] * PREC - 0.5 : c.w[i] * PREC + 0.5);
  return k;
}
const clip8 = (v: number) => { const r = Math.trunc(v / PREC); return r < 0 ? 0 : r > 255 ? 255 : r; };

/** rgba: w x h RGBA bytes. Returns out x out x 3 RGB bytes, stretched (no crop), like ingest's photos. */
export function resizeBilinear(rgba: Uint8ClampedArray, w: number, h: number, out: number): Uint8Array {
  const cx = coeffs(w, out), kx = fixed(cx), cy = coeffs(h, out), ky = fixed(cy);
  const tmp = new Uint8Array(out * h * 3);  // horizontal pass: out x h
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < out; x++) {
      let r = PREC / 2, g = PREC / 2, b = PREC / 2;
      const s0 = cx.start[x], n = cx.size[x], k0 = x * cx.ksize;
      for (let i = 0; i < n; i++) {
        const p = (y * w + s0 + i) * 4, k = kx[k0 + i];
        r += rgba[p] * k; g += rgba[p + 1] * k; b += rgba[p + 2] * k;
      }
      const q = (y * out + x) * 3;
      tmp[q] = clip8(r); tmp[q + 1] = clip8(g); tmp[q + 2] = clip8(b);
    }
  }
  const res = new Uint8Array(out * out * 3);  // vertical pass
  for (let y = 0; y < out; y++) {
    const s0 = cy.start[y], n = cy.size[y], k0 = y * cy.ksize;
    for (let x = 0; x < out; x++) {
      let r = PREC / 2, g = PREC / 2, b = PREC / 2;
      for (let i = 0; i < n; i++) {
        const p = ((s0 + i) * out + x) * 3, k = ky[k0 + i];
        r += tmp[p] * k; g += tmp[p + 1] * k; b += tmp[p + 2] * k;
      }
      const q = (y * out + x) * 3;
      res[q] = clip8(r); res[q + 1] = clip8(g); res[q + 2] = clip8(b);
    }
  }
  return res;
}

import { expect, it } from "vitest";
import { resizeBilinear } from "./resize";
import cases from "./resize.fixtures.json";

// Same bytes as Pillow's Image.resize(..., BILINEAR), the resize ingest gives the training and test photos.
(cases as { w: number; h: number; out: number; rgb: number[]; want: number[] }[]).forEach((c) =>
  it(`${c.w}x${c.h} -> ${c.out}`, () => {
    const rgba = new Uint8ClampedArray(c.w * c.h * 4);
    for (let i = 0; i < c.w * c.h; i++) {
      rgba[i * 4] = c.rgb[i * 3]; rgba[i * 4 + 1] = c.rgb[i * 3 + 1]; rgba[i * 4 + 2] = c.rgb[i * 3 + 2]; rgba[i * 4 + 3] = 255;
    }
    const got = resizeBilinear(rgba, c.w, c.h, c.out);
    let worst = 0;
    got.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - c.want[i])); });
    expect(worst).toBe(0);
  }));

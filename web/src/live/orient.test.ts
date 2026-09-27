import { expect, it } from "vitest";
import { fit, toFrame, turnFromGravity, uprightRgb, uprightTransform, type Turn } from "./orient";

it("reads the turn from gravity", () => {
  expect(turnFromGravity(0.3, 9.7)).toBe(0);
  expect(turnFromGravity(9.7, 0.5)).toBe(270);   // top of the phone to the left
  expect(turnFromGravity(-9.7, 0.5)).toBe(90);   // top to the right
  expect(turnFromGravity(0.2, -9.7)).toBe(180);
});

// A 3x3 frame with one marked pixel: after uprightRgb, the model finds it at (u, v); toFrame must bring that
// back to where the pixel is in the frame, for every turn.
it("turns the image upright and the points back", () => {
  const s = 3;
  for (const turn of [0, 90, 180, 270] as Turn[]) {
    for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
      const img = new Uint8Array(s * s * 3);
      img[(y * s + x) * 3] = 255;
      const up = uprightRgb(img, s, turn);
      const k = up.findIndex((v, i) => i % 3 === 0 && v === 255) / 3;
      const u = ((k % s) + 0.5) / s, v = (Math.floor(k / s) + 0.5) / s;
      const [fx, fy] = toFrame(u, v, turn);
      expect([fx, fy].map((a) => +a.toFixed(4))).toEqual([(x + 0.5) / s, (y + 0.5) / s].map((a) => +a.toFixed(4)));
    }
  }
});

it("turning the phone clockwise turns the scene back clockwise", () => {
  // The frame of a phone turned 90 degrees clockwise shows the scene's top on the frame's left side.
  const s = 2, img = new Uint8Array(s * s * 3);
  img[(1 * s + 0) * 3] = 255;  // scene's top-left is at the frame's bottom-left
  const up = uprightRgb(img, s, 90);
  expect(up[0]).toBe(255);     // upright: top-left
});

it("fits a frame to the screen", () => {
  expect(fit(640, 480, 400, 800, "cover")).toEqual({ left: -333.33333333333337, top: 0, width: 1066.6666666666667, height: 800 });
  expect(fit(640, 480, 400, 800, "contain")).toEqual({ left: 0, top: 250, width: 400, height: 300 });
});

it("saves frames upright the same way the model sees them", () => {
  const s = 3;
  for (const turn of [0, 90, 180, 270] as Turn[]) {
    const { width, height, m: [a, b, c, d, e, f] } = uprightTransform(s, s, turn);
    expect([width, height]).toEqual([s, s]);
    for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
      const img = new Uint8Array(s * s * 3);
      img[(y * s + x) * 3] = 255;
      const k = uprightRgb(img, s, turn).findIndex((v, i) => i % 3 === 0 && v === 255) / 3;
      const [cx, cy] = [x + 0.5, y + 0.5];  // the pixel's centre through the canvas transform
      expect([a * cx + c * cy + e - 0.5, b * cx + d * cy + f - 0.5]).toEqual([k % s, Math.floor(k / s)]);
    }
  }
  expect(uprightTransform(640, 480, 90)).toMatchObject({ width: 480, height: 640 });
});

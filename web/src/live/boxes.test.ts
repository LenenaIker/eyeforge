import { describe, expect, it } from "vitest";
import { anchors, decodeBoxes } from "./boxes";
import cases from "./boxes.fixtures.json";

// Same points as sim2chip.detector.decode on the same outputs (regenerate with scripts/decode_fixtures.py).
type Case = { labels: string[]; threshold: number; size: number; nms_iou: number; max_objects: number;
              boxes: number[][]; scores: number[][]; points: [number, number, string][] };

describe("decodeBoxes matches the Python one", () => {
  (cases as Case[]).forEach((c, k) => it(`case ${k}`, () => {
    const got = decodeBoxes(c.boxes.flat(), c.scores.flat(), c.size, c.labels, c.threshold, c.nms_iou, c.max_objects);
    expect(got.length).toBe(c.points.length);
    got.forEach((p, i) => {
      expect(p[2]).toBe(c.points[i][2]);
      expect(Math.abs(p[0] - c.points[i][0])).toBeLessThanOrEqual(1e-4);
      expect(Math.abs(p[1] - c.points[i][1])).toBeLessThanOrEqual(1e-4);
    });
  }));
  it("counts the anchors of PicoDet inputs", () => {
    expect(anchors(320)).toBe(2125);
    expect(anchors(64)).toBe(85);
  });
});

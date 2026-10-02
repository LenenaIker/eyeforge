// A count detector's outputs to points: a TypeScript copy of sim2chip/detector.py `decode`, kept equal to it by
// boxes.test.ts (fixtures from scripts/decode_fixtures.py), so the phone counts like the evaluation.
// Per class, the boxes scoring at least the threshold, best first, each kept unless it overlaps a kept one of its
// class by more than nmsIou; at most maxObjects, best first. Every kept box is one object at its centre.
import type { LivePoint } from "./decode";

const round4 = (v: number) => Math.round(v * 1e4) / 1e4;

/** Anchors of a PicoDet input: one per cell of its four output grids (strides 8 to 64). */
export function anchors(size: number): number {
  return [8, 16, 32, 64].reduce((n, s) => n + Math.ceil(size / s) ** 2, 0);
}

/** boxes: anchors x 4 (x0, y0, x1, y1, input pixels); scores: classes x anchors, labels order without the
 * background (labels[0]). */
export function decodeBoxes(boxes: ArrayLike<number>, scores: ArrayLike<number>, size: number, labels: string[],
                            threshold: number, nmsIou: number, maxObjects: number): LivePoint[] {
  const n = boxes.length / 4;
  const area = (i: number) => (boxes[i * 4 + 2] - boxes[i * 4]) * (boxes[i * 4 + 3] - boxes[i * 4 + 1]);
  const iou = (i: number, j: number) => {
    const w = Math.min(boxes[i * 4 + 2], boxes[j * 4 + 2]) - Math.max(boxes[i * 4], boxes[j * 4]);
    const h = Math.min(boxes[i * 4 + 3], boxes[j * 4 + 3]) - Math.max(boxes[i * 4 + 1], boxes[j * 4 + 1]);
    const inter = Math.max(0, w) * Math.max(0, h);
    const union = area(i) + area(j) - inter;
    return union > 0 ? inter / union : 0;
  };
  const kept: { score: number; i: number; c: number }[] = [];
  for (let c = 0; c < labels.length - 1; c++) {
    const idx: number[] = [];
    for (let i = 0; i < n; i++) if (scores[c * n + i] >= threshold) idx.push(i);
    idx.sort((a, b) => scores[c * n + b] - scores[c * n + a]);  // stable: ties keep anchor order, like Python
    const chosen: number[] = [];
    for (const i of idx) if (chosen.every((j) => iou(i, j) <= nmsIou)) chosen.push(i);
    for (const i of chosen) kept.push({ score: scores[c * n + i], i, c });
  }
  kept.sort((a, b) => b.score - a.score);
  return kept.slice(0, maxObjects).map(({ i, c }): LivePoint => [
    round4((boxes[i * 4] + boxes[i * 4 + 2]) / 2 / size), round4((boxes[i * 4 + 1] + boxes[i * 4 + 3]) / 2 / size),
    labels[c + 1]]);
}

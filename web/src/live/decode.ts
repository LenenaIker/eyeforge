// The count model's output grid to points: a TypeScript copy of sim2chip/counting.py `decode`, kept equal to
// it by decode.test.ts (fixtures from scripts/decode_fixtures.py), so the phone counts like the evaluation.
// A cell is an object when its best non-background class reaches the threshold; neighbouring object cells of
// the same class are one object, at their probability-weighted centre, in 0..1 of the image.

export type LivePoint = [x: number, y: number, label: string];

const round4 = (v: number) => Math.round(v * 1e4) / 1e4;

/** probs: rows x cols x classes, row-major, background first (the model's output as it comes). */
export function decode(probs: ArrayLike<number>, rows: number, cols: number, labels: string[],
                       threshold: number): LivePoint[] {
  const n = labels.length;
  const at = (i: number, j: number, c: number) => probs[(i * cols + j) * n + c];
  const best = new Int32Array(rows * cols);
  const on = new Uint8Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let b = 1;
      for (let c = 2; c < n; c++) if (at(i, j, c) > at(i, j, b)) b = c;  // first best on ties, like max()
      best[i * cols + j] = b;
      on[i * cols + j] = at(i, j, b) >= threshold ? 1 : 0;
    }
  }
  const seen = new Uint8Array(rows * cols);
  const points: LivePoint[] = [];
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const k = i * cols + j;
      if (!on[k] || seen[k]) continue;
      const cls = best[k];
      const todo: [number, number][] = [[i, j]];
      seen[k] = 1;
      let w = 0, sx = 0, sy = 0;
      while (todo.length) {
        const [a, b] = todo.pop()!;
        const p = at(a, b, cls);
        w += p; sx += (b + 0.5) * p; sy += (a + 0.5) * p;
        for (const [da, db] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const y = a + da, x = b + db, q = y * cols + x;
          if (y >= 0 && y < rows && x >= 0 && x < cols && on[q] && !seen[q] && best[q] === cls) {
            seen[q] = 1;
            todo.push([y, x]);
          }
        }
      }
      points.push([round4(sx / w / cols), round4(sy / w / rows), labels[cls]]);
    }
  }
  return points;
}

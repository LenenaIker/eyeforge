import { describe, expect, it } from "vitest";
import { decode } from "./decode";
import cases from "./decode.fixtures.json";

// Same points as sim2chip.counting.decode on the same grids (regenerate with scripts/decode_fixtures.py).
describe("decode matches the Python one", () => {
  (cases as { labels: string[]; threshold: number; grid: number[][][]; points: [number, number, string][] }[])
    .forEach((c, k) => it(`case ${k}`, () => {
      const got = decode(c.grid.flat(2), c.grid.length, c.grid[0].length, c.labels, c.threshold);
      expect(got.length).toBe(c.points.length);
      got.forEach((p, i) => {
        expect(p[2]).toBe(c.points[i][2]);
        expect(Math.abs(p[0] - c.points[i][0])).toBeLessThanOrEqual(1e-4);
        expect(Math.abs(p[1] - c.points[i][1])).toBeLessThanOrEqual(1e-4);
      });
    }));
});

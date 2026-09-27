// Camera-app behaviour for the live test: the screen stays portrait while the phone turns, so the frame is turned
// upright before the model sees it (the model was trained on upright scenes), points are turned back onto the
// frame, and the labels rotate to face the user. Plus the geometry of a frame shown filling the screen.

/** How far the phone is turned from upright portrait, clockwise, in degrees. */
export type Turn = 0 | 90 | 180 | 270;

/** The turn from gravity as the device reports it (x to the right of the screen, y up it; Android sign
 * convention: upright gives y > 0). Hysteresis-free quadrants: good enough for 90 degree steps. */
export function turnFromGravity(x: number, y: number): Turn {
  if (Math.abs(y) >= Math.abs(x)) return y >= 0 ? 0 : 180;
  return x > 0 ? 270 : 90;  // top of the phone to the left (x up) is a counter-clockwise turn
}

/** An s x s RGB image turned so the scene is upright: undoes a phone turn of `turn` degrees clockwise. */
export function uprightRgb(src: Uint8Array, s: number, turn: Turn): Uint8Array {
  if (turn === 0) return src;
  const dst = new Uint8Array(src.length);
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      // A phone turned clockwise sees the scene turned counter-clockwise: turn the image clockwise back.
      const [r, c] = turn === 90 ? [x, s - 1 - y] : turn === 270 ? [s - 1 - x, y] : [s - 1 - y, s - 1 - x];
      const i = (y * s + x) * 3, j = (r * s + c) * 3;
      dst[j] = src[i]; dst[j + 1] = src[i + 1]; dst[j + 2] = src[i + 2];
    }
  }
  return dst;
}

/** A point (0..1) found in the upright image, back on the frame as it came from the camera. */
export function toFrame(u: number, v: number, turn: Turn): [number, number] {
  if (turn === 90) return [v, 1 - u];
  if (turn === 270) return [1 - v, u];
  if (turn === 180) return [1 - u, 1 - v];
  return [u, v];
}

/** Where a frame of fw x fh lands in a view of vw x vh: "cover" fills the view (cropping), "contain" fits it. */
export function fit(fw: number, fh: number, vw: number, vh: number, mode: "cover" | "contain") {
  const k = mode === "cover" ? Math.max(vw / fw, vh / fh) : Math.min(vw / fw, vh / fh);
  const w = fw * k, h = fh * k;
  return { left: (vw - w) / 2, top: (vh - h) / 2, width: w, height: h };
}

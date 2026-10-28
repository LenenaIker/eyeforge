// What the web flashes onto a board and how it reads the board's answers. Pure code (no USB): the flashing itself
// lives in flasher.ts. Each board profile has a folder under /firmware with the prebuilt firmware and a
// manifest.json saying where every part goes (web/public/firmware/esp32/manifest.json).

/** Board folder per device profile (sim2chip/hardware.py). Only the classic ESP32 (ESP32-CAM) for now; an
 * ESP32-S3 would get its own folder (bootloader at 0x0, native USB). */
export const BOARDS: Record<string, string> = { esp32cam: "/firmware/esp32" };

export type Part = { name: string; offset: number; file: string };
export type Manifest = {
  firmware?: string;      // FIRMWARE_VERSION of firmware/esp32cam/src/main.cpp, which the board says when it starts
  board: string;          // what the user plugs in, e.g. "ESP32-CAM"
  chip: string;           // esptool-js CHIP_NAME the board must report, e.g. "ESP32"
  usb: "usb-serial" | "native";  // a USB to serial chip (CH340) or the chip's own USB
  parts: Part[];          // firmware parts, written before the model
  model: { offset: number; maxSize: number; magic: string };
  baud: number;           // flashing speed
  monitorBaud: number;    // the firmware's own output
};

const num = (v: unknown) => (typeof v === "number" ? v : parseInt(String(v), 16));

export function parseManifest(raw: any): Manifest {
  if (raw?.format !== 1 || !Array.isArray(raw.parts)) throw new Error("Unknown firmware manifest format.");
  return {
    firmware: raw.firmware, board: raw.board, chip: raw.chip, usb: raw.usb ?? "usb-serial",
    parts: raw.parts.map((p: any) => ({ name: p.name, offset: num(p.offset), file: p.file })),
    model: { offset: num(raw.model.offset), maxSize: num(raw.model.max_size), magic: raw.model.magic },
    baud: raw.baud ?? 460800, monitorBaud: raw.monitor_baud ?? 115200,
  };
}

export const flashable = (hardware: string | undefined, deviceBlob: string | null | undefined) =>
  !!hardware && hardware in BOARDS && deviceBlob === "model.bin";

/** Why this model.bin cannot go on the board, or null when it can. */
export function checkModel(bin: Uint8Array, m: Manifest["model"]): string | null {
  const magic = new TextDecoder().decode(bin.subarray(0, 4));
  if (bin.length < 24 || magic !== m.magic) return "This model file is not in the board format. Retrain the model to get one.";
  if (bin.length > m.maxSize) {
    return `The model is ${Math.round(bin.length / 1024)} KB and the board keeps at most ${Math.round(m.maxSize / 1024)} KB.`;
  }
  return null;
}

/** The model's fingerprint as the board reports it: the CRC-32 in model.bin's header, 8 hex digits. */
export function modelCrc(bin: Uint8Array): string | null {
  if (bin.length < 16) return null;
  const v = new DataView(bin.buffer, bin.byteOffset, bin.byteLength).getUint32(12, true);
  return v.toString(16).padStart(8, "0");
}

/** Points of a count model: [x, y, class], 0-1 on the picture the model saw. From firmware 1.2.0 on that is the
 * whole frame stretched to a square, so they are 0-1 on the whole frame too; before, the frame's centre square. */
export type BoardPoint = [number, number, string];

/** One line of the firmware's output (firmware/esp32cam/src/main.cpp prints JSON lines; the boot log is text).
 * firmware and crc come from firmware 1.1.0 on (older boards say neither); jpeg is the picture the model saw.
 * frame (width, height) comes from 1.2.0 on: the model sees that whole frame stretched; without it, its centre square. */
export type BoardEvent =
  | { kind: "ready"; input: number; labels: string[]; firmware?: string; crc?: string; task?: string; frame?: [number, number] }
  | { kind: "error"; message: string; firmware?: string }
  | { kind: "label"; label: string; score: number; ms: number; scores: Record<string, number>; jpeg?: string }
  | { kind: "counts"; ms: number; counts: Record<string, number>; points?: BoardPoint[]; jpeg?: string }
  | { kind: "photo"; w: number; h: number; bytes: number; captureMs: number; jpeg: string }
  | { kind: "photo_sent"; bytes: number; sendMs: number }
  | { kind: "text" };

// A line with bytes lost on the way (the USB adapter drops some at 921600 baud: on 2026-10-28 the board's ready
// line came without '"task":"classification","input":96' every time, and the page took the board for silent and
// asked to flash it again and again). What is still whole of a ready or a label line is enough.
function salvage(t: string): BoardEvent {
  const str = (k: string) => t.match(new RegExp(`"${k}":"([^"]*)"`))?.[1];
  const num = (k: string) => { const m = t.match(new RegExp(`"${k}":(-?[0-9.]+)`)); return m ? Number(m[1]) : undefined; };
  if (t.includes('"event":"ready"')) {
    const labels = t.match(/"labels":\[([^\]]*)\]/)?.[1].match(/"([^"]*)"/g)?.map((x) => x.slice(1, -1)) ?? [];
    const frame = t.match(/"frame":\[(\d+),(\d+)\]/);
    const firmware = str("firmware"), crc = str("crc"), task = str("task");
    return { kind: "ready", input: num("input") ?? 96, labels,
             ...(firmware ? { firmware } : {}), ...(crc && /^[0-9a-f]{8}$/.test(crc) ? { crc } : {}),
             ...(task ? { task } : {}), ...(frame ? { frame: [Number(frame[1]), Number(frame[2])] as [number, number] } : {}) };
  }
  const label = str("label"), score = num("score");
  if (label !== undefined && score !== undefined) return { kind: "label", label, score, ms: num("ms") ?? 0, scores: {} };
  return { kind: "text" };
}

export function parseLine(line: string): BoardEvent {
  const t = line.trim();
  if (!t.startsWith("{")) return { kind: "text" };
  let o: any;
  try { o = JSON.parse(t); } catch { return salvage(t); }
  const jpeg = typeof o.jpeg === "string" && wholeJpeg(o.jpeg) ? { jpeg: o.jpeg as string } : {};
  if (o.event === "ready") {
    return { kind: "ready", input: o.input, labels: o.labels ?? [],
             ...(o.firmware ? { firmware: String(o.firmware) } : {}), ...(o.crc ? { crc: String(o.crc) } : {}),
             ...(o.task ? { task: String(o.task) } : {}),
             ...(Array.isArray(o.frame) && o.frame.length === 2 && o.frame.every((v: unknown) => typeof v === "number" && v > 0)
               ? { frame: [o.frame[0], o.frame[1]] as [number, number] } : {}) };
  }
  if (o.event === "error") {
    return { kind: "error", message: String(o.message ?? "unknown error"), ...(o.firmware ? { firmware: String(o.firmware) } : {}) };
  }
  if (o.event === "photo" && typeof o.jpeg === "string") {
    return { kind: "photo", w: o.w, h: o.h, bytes: o.bytes, captureMs: o.capture_ms, jpeg: o.jpeg };
  }
  if (o.event === "photo_sent") return { kind: "photo_sent", bytes: o.bytes, sendMs: o.send_ms };
  if (typeof o.label === "string") return { kind: "label", label: o.label, score: o.score, ms: o.ms, scores: o.scores ?? {}, ...jpeg };
  if (o.counts) {
    return { kind: "counts", ms: o.ms, counts: o.counts, ...(Array.isArray(o.points) ? { points: o.points } : {}), ...jpeg };
  }
  return { kind: "text" };
}

/** What a log shows of a line: frames and photos carry long base64 pictures. */
export const logLine = (line: string, max = 240) =>
  line.length > max ? `${line.slice(0, max)}... (${line.length} characters)` : line;

/** The size the board's photo is asked for when the user saves a frame (see the firmware's "capture"). UXGA
 * (1600x1200, about 75 KB) went through the cable in 1.25 s at 921600 baud on 2026-10-12, 3.1 s from tap to photo. */
export const PHOTO_SIZE = "uxga";

/** Why the live test should flash the board first, or null when it runs this job's model on our firmware.
 * optional: only the firmware version differs, so the board can be tested as it is. */
export function boardMismatch(ready: Extract<BoardEvent, { kind: "ready" }> | null, crc: string | null,
                              firmware?: string): { reason: string; optional: boolean } | null {
  if (!ready) return { reason: "The board did not answer like an EyeForge board with a model.", optional: false };
  if (!ready.crc || !ready.firmware) {
    return { reason: "The board runs an older EyeForge firmware that cannot send what it sees.", optional: false };
  }
  if (crc && ready.crc !== crc) return { reason: "The board has another model on it.", optional: false };
  if (firmware && ready.firmware !== firmware) {
    return { reason: `The board runs EyeForge firmware ${ready.firmware}; this page brings ${firmware}.`, optional: true };
  }
  return null;
}

/** Width over height of the picture the model sees, to show its frames undistorted: the whole frame from firmware
 * 1.2.0 on (its stretched square drawn back at the frame's shape), the centre square before. */
export const boardAspect = (ready: Extract<BoardEvent, { kind: "ready" }> | null) =>
  ready?.frame ? ready.frame[0] / ready.frame[1] : 1;

/** What the board said of its last frame, as the live test's check starts from it, with the points 0-1 on the
 * whole photo (w x h) from "capture", which has the frame's field of view. From firmware 1.2.0 on (ready.frame)
 * they already are; an older board saw the centre square, so they are moved into it. */
export function boardGuess(ev: BoardEvent | null, ready?: Extract<BoardEvent, { kind: "ready" }> | null,
                           photo?: { w: number; h: number }): { points: BoardPoint[]; top: { label: string; p: number } | null } {
  if (ev?.kind === "label") return { points: [], top: { label: ev.label, p: ev.score } };
  if (ev?.kind === "counts") {
    const square = !ready?.frame && photo ? Math.min(photo.w, photo.h) : 0;
    const onPhoto = ([x, y, l]: BoardPoint): BoardPoint => square
      ? [((photo!.w - square) / 2 + x * square) / photo!.w, ((photo!.h - square) / 2 + y * square) / photo!.h, l]
      : [x, y, l];
    return { points: (ev.points ?? []).map(onPhoto), top: null };
  }
  return { points: [], top: null };
}

/** A JPEG that came whole: base64 of the right length, starting and ending with the JPEG markers. A line that lost
 * bytes inside its picture still parses as JSON, and the picture shows as a broken image (2026-10-28, a phone). */
export function wholeJpeg(b64: string): boolean {
  if (b64.length < 8 || b64.length % 4) return false;
  try {
    const bytes = fromBase64(b64);
    const n = bytes.length;
    return bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[n - 2] === 0xff && bytes[n - 1] === 0xd9;
  } catch { return false; }
}

export function fromBase64(b64: string): Uint8Array<ArrayBuffer> {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** What a board says it runs: from its ready line, or only the firmware from an error line (no model yet).
 * null: it did not answer like our firmware (another program, an old firmware, a blank board). */
export type OnBoard = { firmware?: string; crc?: string } | null;

/** Which parts to write: the firmware only when the board runs another version (or did not answer), the model
 * only when its fingerprint differs. force writes everything, for a board in a bad state. */
export function planWrite(onBoard: OnBoard, firmware: string | undefined, crc: string | null, force = false) {
  if (force || !onBoard) return { firmware: true, model: true };
  return { firmware: !firmware || onBoard.firmware !== firmware, model: !crc || onBoard.crc !== crc };
}

/** Splits a byte stream into lines; feed it chunks, it returns the complete lines. */
export function lineSplitter() {
  const dec = new TextDecoder();
  let partial = "";
  return (chunk: Uint8Array): string[] => {
    partial += dec.decode(chunk, { stream: true });
    const lines = partial.split(/\r?\n/);
    partial = lines.pop() ?? "";
    return lines.filter((l) => l.length > 0);
  };
}

export type Phase = "pick" | "connect" | "write" | "monitor";

export const RESET_HINT = "Hold the IO0 button, tap RST, release IO0, then try again.";

/** The error a user reads, in plain words, for what failed in each phase. */
export function friendlyError(e: unknown, phase: Phase): string {
  const err = e as { name?: string; message?: string };
  const msg = String(err?.message ?? e);
  if (phase === "pick") {
    if (err?.name === "NotFoundError" || /no device selected|cancel/i.test(msg)) {
      return "No board was chosen. Tap Flash again and pick the board's USB device (often \"USB Serial\" or \"CH340\").";
    }
    if (err?.name === "SecurityError") return "The browser blocked access to USB devices on this page.";
  }
  if (/claim|access denied|busy|already open|failed to open/i.test(msg)) {
    return "Another program is using the board (a serial monitor, Arduino IDE or another tab). Close it, unplug and plug the board, and try again.";
  }
  if (/lost|disconnect|NetworkError|transfer|unplug|device.*(gone|closed)/i.test(msg) || err?.name === "NetworkError") {
    return phase === "write"
      ? "The board was unplugged or the cable lost contact while writing. Plug it back in and flash again: the board will not run until it is written completely."
      : "The board was unplugged. Plug it back in and try again.";
  }
  if (phase === "connect") return `The board is not answering. ${RESET_HINT}`;
  if (phase === "write") return `Writing stopped: ${msg}. Plug the board in again and flash again.`;
  return msg;
}

// What the web flashes onto a board and how it reads the board's answers. Pure code (no USB): the flashing itself
// lives in flasher.ts. Each board profile has a folder under /firmware with the prebuilt firmware and a
// manifest.json saying where every part goes (web/public/firmware/esp32/manifest.json).

/** Board folder per device profile (sim2chip/hardware.py). Only the classic ESP32 (ESP32-CAM) for now; an
 * ESP32-S3 would get its own folder (bootloader at 0x0, native USB). */
export const BOARDS: Record<string, string> = { esp32cam: "/firmware/esp32" };

export type Part = { name: string; offset: number; file: string };
export type Manifest = {
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
    board: raw.board, chip: raw.chip, usb: raw.usb ?? "usb-serial",
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

/** One line of the firmware's output (firmware/esp32cam/src/main.cpp prints JSON lines; the boot log is text). */
export type BoardEvent =
  | { kind: "ready"; input: number; labels: string[] }
  | { kind: "error"; message: string }
  | { kind: "label"; label: string; score: number; ms: number; scores: Record<string, number> }
  | { kind: "counts"; ms: number; counts: Record<string, number> }
  | { kind: "text" };

export function parseLine(line: string): BoardEvent {
  const t = line.trim();
  if (!t.startsWith("{")) return { kind: "text" };
  let o: any;
  try { o = JSON.parse(t); } catch { return { kind: "text" }; }
  if (o.event === "ready") return { kind: "ready", input: o.input, labels: o.labels ?? [] };
  if (o.event === "error") return { kind: "error", message: String(o.message ?? "unknown error") };
  if (typeof o.label === "string") return { kind: "label", label: o.label, score: o.score, ms: o.ms, scores: o.scores ?? {} };
  if (o.counts) return { kind: "counts", ms: o.ms, counts: o.counts };
  return { kind: "text" };
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

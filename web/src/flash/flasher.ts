// Flash a board from the browser with esptool-js and watch its output on the same cable. Android Chrome has
// WebUSB but no USB Web Serial, so there the CH340 driver of esptool-js talks to the board; desktop Chrome and
// Edge use Web Serial, where the system's own driver owns the USB to serial chip. Tested on a Pixel 7a
// (Chrome 154) with an ESP32-CAM-MB (CH340) on 2026-10-12.
import { BOARDS, checkModel, friendlyError, lineSplitter, parseLine, parseManifest, type Manifest, type OnBoard, type Phase } from "./board";

type Esptool = typeof import("esptool-js");
let esptool: Promise<Esptool> | null = null;
const lib = () => (esptool ??= import("esptool-js").then((m) => { keepReading(m.WebUSBSerialPort); return m; }));  // 100 KB only for people who flash

const IN_FLIGHT = 8;  // USB reads waiting at once on the phone's CH340 path
/** esptool-js 0.7.0 reads the CH340 over WebUSB one 64-byte transfer at a time: between two, the adapter's small
 * buffer overflows and bytes are lost (2026-10-28, a Pixel 7a at 115200 baud: most live pictures came broken).
 * The readable stream is made again with IN_FLIGHT transfers always waiting, taken in order. */
function keepReading(Port: any) {
  if (!Port?.prototype?.createStreams || Port.prototype.__keepsReading) return;
  const make = Port.prototype.createStreams;
  Port.prototype.__keepsReading = true;
  Port.prototype.createStreams = function (this: any) {
    make.call(this);
    const self = this, device = this.usbDevice, endpoint = this.endpointIn, size = 64;
    this.readable = new ReadableStream<Uint8Array>({
      start: (controller) => {
        const queue: Promise<any>[] = [];
        const ask = () => queue.push(device.transferIn(endpoint, size).catch((e: unknown) => ({ error: e })));
        (async () => {
          for (let i = 0; i < IN_FLIGHT; i++) ask();
          while (self.readLoopRunning && device.opened) {
            const r = await queue.shift();
            if (!self.readLoopRunning || !device.opened) break;
            if (r?.error) { controller.error(r.error); return; }
            if (r?.status === "stall") { await device.clearHalt("in", endpoint).catch(() => {}); }
            else if (r?.status === "ok" && r.data?.byteLength) {
              controller.enqueue(new Uint8Array(r.data.buffer, r.data.byteOffset, r.data.byteLength));
            }
            ask();
          }
          try { controller.close(); } catch { /* already closed */ }
        })();
      },
      cancel: () => { self.readLoopRunning = false; },
    });
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const isAndroid = () => /android/i.test(navigator.userAgent);

/** "usb" (WebUSB with the CH340 driver), "serial" (Web Serial) or null when this browser cannot flash. */
export function flashRoute(): "usb" | "serial" | null {
  const usb = "usb" in navigator, serial = "serial" in navigator;
  if (isAndroid()) return usb ? "usb" : serial ? "serial" : null;
  return serial ? "serial" : usb ? "usb" : null;
}

export class FlashError extends Error {
  constructor(message: string, readonly phase: Phase) { super(message); }
}
const wrap = (e: unknown, phase: Phase) => (e instanceof FlashError ? e : new FlashError(friendlyError(e, phase), phase));

export type Files = { manifest: Manifest; parts: { address: number; data: Uint8Array; name: string }[] };

async function bytes(url: string): Promise<Uint8Array> {
  const r = await fetch(url, { cache: "no-cache" });
  // the server answers an unknown path with the app's page: never write that onto the board
  if (!r.ok || r.headers.get("content-type")?.includes("text/html")) throw new Error(`could not download ${url.split("/").pop()}`);
  return new Uint8Array(await r.arrayBuffer());
}

/** The board's firmware parts plus the model, downloaded before the user plugs anything. */
export async function loadFiles(hardware: string, modelUrl: string): Promise<Files & { model: Uint8Array }> {
  const dir = BOARDS[hardware];
  const r = await fetch(`${dir}/manifest.json`, { cache: "no-cache" });
  if (!r.ok) throw new Error("could not download the firmware description");
  const manifest = parseManifest(await r.json());
  const [model, ...fw] = await Promise.all([bytes(modelUrl), ...manifest.parts.map((p) => bytes(`${dir}/${p.file}`))]);
  const bad = checkModel(model, manifest.model);
  if (bad) throw new FlashError(bad, "pick");
  const parts = manifest.parts.map((p, i) => ({ address: p.offset, data: fw[i], name: p.name }));
  parts.push({ address: manifest.model.offset, data: model, name: "model" });
  return { manifest, parts, model };
}

/** Opens the browser's device picker. Must run straight from a tap. */
export async function pickPort(): Promise<any> {
  const route = flashRoute();
  try {
    if (route === "usb") return await (await lib()).WebUSBSerialPort.requestPort();
    if (route === "serial") return await (navigator as any).serial.requestPort();
  } catch (e) { throw wrap(e, "pick"); }
  throw new FlashError("This browser cannot reach USB devices.", "pick");
}

export type Progress = { step: "connect" | "write" | "restart"; part?: string; done: number; total: number };

/** Writes every part in one go, then restarts the board. Calls log with esptool's own lines. */
export async function flash(port: any, files: Files, onProgress: (p: Progress) => void, log: (s: string) => void) {
  const { ESPLoader, Transport } = await lib();
  const terminal = { clean() {}, writeLine: log, write: log };
  const transport = new Transport(port, false);
  const total = files.parts.reduce((n, p) => n + p.data.length, 0);
  let phase: Phase = "connect";
  try {
    onProgress({ step: "connect", done: 0, total });
    const loader = new ESPLoader({ transport, baudrate: files.manifest.baud, terminal });
    const description = await loader.main();
    log(`chip: ${description}`);
    if (loader.chip.CHIP_NAME !== files.manifest.chip) {
      throw new FlashError(`This board has an ${loader.chip.CHIP_NAME} chip, but this model's firmware is for the ${files.manifest.chip} `
        + `(${files.manifest.board}). Plug in the ${files.manifest.board} the model was trained for.`, "connect");
    }
    phase = "write";
    const before = files.parts.map((_, i) => files.parts.slice(0, i).reduce((n, p) => n + p.data.length, 0));
    await loader.writeFlash({
      fileArray: files.parts.map((p) => ({ address: p.address, data: p.data })),
      flashMode: "keep", flashFreq: "keep", flashSize: "keep", eraseAll: false, compress: true,
      // written and size count compressed bytes: scale to the part's own size
      reportProgress: (i: number, written: number, size: number) => onProgress({ step: "write", part: files.parts[i]?.name,
        done: before[i] + Math.round(files.parts[i].data.length * (size ? written / size : 0)), total }),
    });
    onProgress({ step: "restart", done: total, total });
    await loader.after("hard_reset");
  } catch (e) {
    try { await transport.disconnect(); } catch { /* already gone */ }
    throw wrap(e, phase);
  }
  // The flashing transport keeps a reader that would eat the board's text: close it for the monitor.
  try { await transport.disconnect(); } catch { /* already closed */ }
  await sleep(300);
}

/** A board being watched: stop() closes the port; send() writes a command line (firmware 1.1.0 on). */
export type Watching = { stop: () => Promise<void>; send: (line: string) => Promise<void> };

/** Restarts the board and reads its lines until stop() is called. onEnd runs when reading ends on its own. */
export async function watch(port: any, baud: number, onLine: (line: string) => void,
                            onEnd: (error?: string) => void): Promise<Watching> {
  const { Transport } = await lib();
  const transport = new Transport(port, false, false);
  let closed = false;
  const stop = async () => {
    if (closed) return;
    closed = true;
    try { await transport.disconnect(); } catch { /* already closed */ }
  };
  try {
    await transport.connect(baud);
    await transport.setDTR(false);
    await transport.setRTS(true);   // EN low: reset
    await sleep(150);
    await transport.setRTS(false);  // run, IO0 high
  } catch (e) {
    await stop();
    throw wrap(e, "monitor");
  }
  const split = lineSplitter();
  transport.rawRead((data: Uint8Array) => split(data).forEach(onLine), () => closed)
    .then(() => { if (!closed) { closed = true; onEnd("The board stopped sending. Unplugged?"); } else onEnd(); })
    .catch((e: unknown) => { const was = closed; closed = true; onEnd(was ? undefined : friendlyError(e, "monitor")); });
  // Raw bytes, straight to the port: Transport.write would wrap them in SLIP frames for the ROM loader.
  const send = async (line: string) => {
    if (closed || !port.writable) throw new Error("The board is not connected.");
    const writer = port.writable.getWriter();
    try { await writer.write(new TextEncoder().encode(`${line}\n`)); } finally { writer.releaseLock(); }
  };
  return { stop, send };
}

/** Restarts the board and listens for what it runs (OnBoard), for at most ms. Leaves the port closed. */
export async function probe(port: any, baud: number, ms = 6000): Promise<OnBoard> {
  let done: (v: OnBoard) => void = () => {};
  const answer = new Promise<OnBoard>((ok) => { done = ok; });
  const timer = setTimeout(() => done(null), ms);
  const board = await watch(port, baud, (line) => {
    const ev = parseLine(line);
    if (ev.kind === "ready") done({ firmware: ev.firmware, crc: ev.crc });
    else if (ev.kind === "error" && ev.firmware) done({ firmware: ev.firmware });
  }, () => done(null));
  const v = await answer;
  clearTimeout(timer);
  await board.stop();
  await sleep(200);
  return v;
}

// The port picked from the job page's "Try it live" tap (browsers only open the picker from a tap), handed to
// the board's live page, which opens it.
let handed: any = null;
export const handPort = (port: any) => { handed = port; };
export const takePort = () => { const p = handed; handed = null; return p; };

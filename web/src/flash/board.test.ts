import { expect, it } from "vitest";
import raw from "../../public/firmware/esp32/manifest.json";
import { boardAspect, boardGuess, boardMismatch, planWrite, checkModel, flashable, friendlyError, fromBase64, lineSplitter, logLine, modelCrc, parseLine,
         parseManifest } from "./board";

const manifest = parseManifest(raw);

it("reads the ESP32-CAM manifest with the offsets of firmware/esp32cam/partitions.csv", () => {
  expect(manifest.chip).toBe("ESP32");
  expect(manifest.parts.map((p) => [p.file, p.offset])).toEqual([
    ["bootloader.bin", 0x1000], ["partitions.bin", 0x8000], ["ota_data_initial.bin", 0xe000], ["firmware.bin", 0x10000]]);
  expect(manifest.model).toEqual({ offset: 0x190000, maxSize: 0x200000, magic: "EFM1" });
  // the app partition (0x10000, 0x180000 long) ends where the model starts
  expect(0x10000 + 0x180000).toBe(manifest.model.offset);
});

it("offers flashing only for an ESP32-CAM model with a model.bin", () => {
  expect(flashable("esp32cam", "model.bin")).toBe(true);
  expect(flashable("esp32cam", null)).toBe(false);
  expect(flashable("esp32cam", undefined)).toBe(false);
  expect(flashable("esp32s3", "model.bin")).toBe(false);
  expect(flashable("generic", null)).toBe(false);
});

it("checks the model blob", () => {
  const blob = new Uint8Array(64);
  blob.set(new TextEncoder().encode("EFM1"));
  expect(checkModel(blob, manifest.model)).toBeNull();
  expect(checkModel(new Uint8Array(64), manifest.model)).toMatch(/not in the board format/);
  const big = new Uint8Array(manifest.model.maxSize + 1);
  big.set(new TextEncoder().encode("EFM1"));
  expect(checkModel(big, manifest.model)).toMatch(/at most 2048 KB/);
});

it("parses the firmware's lines", () => {
  expect(parseLine('{"event":"ready","input":96,"arena":206000,"labels":["empty","grass"]}'))
    .toEqual({ kind: "ready", input: 96, labels: ["empty", "grass"] });
  expect(parseLine('{"label":"grass","score":0.91,"ms":1210,"scores":{"empty":0.09,"grass":0.91}}'))
    .toEqual({ kind: "label", label: "grass", score: 0.91, ms: 1210, scores: { empty: 0.09, grass: 0.91 } });
  expect(parseLine('{"ms":1300,"counts":{"bottle":3}}')).toEqual({ kind: "counts", ms: 1300, counts: { bottle: 3 } });
  expect(parseLine('{"event":"error","message":"capture failed"}')).toEqual({ kind: "error", message: "capture failed" });
  expect(parseLine("I (312) cpu_start: Starting scheduler").kind).toBe("text");
  expect(parseLine("{broken").kind).toBe("text");
});

it("splits a byte stream into lines across chunks", () => {
  const split = lineSplitter();
  const enc = new TextEncoder();
  expect(split(enc.encode('{"a":'))).toEqual([]);
  expect(split(enc.encode('1}\r\nboot\n\npart'))).toEqual(['{"a":1}', "boot"]);
  expect(split(enc.encode("ial\n"))).toEqual(["partial"]);
});

it("explains errors in plain words", () => {
  const cancelled = Object.assign(new Error("No device selected."), { name: "NotFoundError" });
  expect(friendlyError(cancelled, "pick")).toMatch(/No board was chosen/);
  expect(friendlyError(new Error("Failed to connect with the device"), "connect")).toMatch(/hold the IO0 button/i);
  expect(friendlyError(Object.assign(new Error("The device has been lost."), { name: "NetworkError" }), "write"))
    .toMatch(/unplugged/);
  expect(friendlyError(new Error("Unable to claim interface."), "connect")).toMatch(/Another program/);
});

it("reads the firmware 1.1.0 lines: fingerprint, pictures, points and photos", () => {
  const ready = parseLine('{"event":"ready","firmware":"1.1.0","crc":"0badf00d","task":"count","input":96,"channels":3,"grid":12,"arena":1,"labels":["bg","bottle"]}');
  expect(ready).toEqual({ kind: "ready", input: 96, labels: ["bg", "bottle"], firmware: "1.1.0", crc: "0badf00d", task: "count" });
  expect(parseLine('{"label":"grass","score":0.91,"ms":1210,"scores":{"grass":0.91},"jpeg":"/9j/"}'))
    .toMatchObject({ kind: "label", label: "grass", jpeg: "/9j/" });
  expect(parseLine('{"ms":1300,"counts":{"bottle":2},"points":[[0.25,0.5,"bottle"],[0.75,0.5,"bottle"]],"jpeg":"/9j/"}'))
    .toEqual({ kind: "counts", ms: 1300, counts: { bottle: 2 }, points: [[0.25, 0.5, "bottle"], [0.75, 0.5, "bottle"]], jpeg: "/9j/" });
  expect(parseLine('{"event":"photo","w":800,"h":600,"bytes":3,"capture_ms":700,"jpeg":"AAEC"}'))
    .toEqual({ kind: "photo", w: 800, h: 600, bytes: 3, captureMs: 700, jpeg: "AAEC" });
  expect(parseLine('{"event":"photo_sent","bytes":3,"send_ms":900}')).toEqual({ kind: "photo_sent", bytes: 3, sendMs: 900 });
  expect(Array.from(fromBase64("AAEC"))).toEqual([0, 1, 2]);
  expect(logLine("x".repeat(5000))).toMatch(/^x{240}\.\.\. \(5000 characters\)$/);
});

it("tells whether the board runs this job's model", () => {
  const blob = new Uint8Array(32);
  new DataView(blob.buffer).setUint32(12, 0x0badf00d, true);
  expect(modelCrc(blob)).toBe("0badf00d");
  const ready = { kind: "ready" as const, input: 96, labels: [], firmware: "1.1.0", crc: "0badf00d" };
  expect(boardMismatch(ready, "0badf00d", "1.1.0")).toBeNull();
  expect(boardMismatch(ready, "12345678", "1.1.0")).toMatchObject({ optional: false, reason: /another model/ });
  expect(boardMismatch(ready, "0badf00d", "1.2.0")).toMatchObject({ optional: true });
  expect(boardMismatch({ kind: "ready", input: 96, labels: [] }, "0badf00d", "1.1.0")).toMatchObject({ optional: false, reason: /older/ });
  expect(boardMismatch(null, "0badf00d", "1.1.0")).toMatchObject({ optional: false });
});

it("starts the check from the board's last frame", () => {
  expect(boardGuess({ kind: "label", label: "grass", score: 0.8, ms: 1, scores: {} })).toEqual({ points: [], top: { label: "grass", p: 0.8 } });
  expect(boardGuess({ kind: "counts", ms: 1, counts: {}, points: [[0.1, 0.2, "bottle"]] }))
    .toEqual({ points: [[0.1, 0.2, "bottle"]], top: null });
  expect(boardGuess(null)).toEqual({ points: [], top: null });
});

it("puts the board's points on the whole photo: as they are from 1.2.0, moved from the centre square before", () => {
  const ready = parseLine('{"event":"ready","firmware":"1.2.0","crc":"0badf00d","task":"count","input":96,"frame":[320,240],"labels":["bg","bottle"]}');
  expect(ready).toMatchObject({ kind: "ready", frame: [320, 240] });
  if (ready.kind !== "ready") throw new Error("not a ready line");
  expect(boardAspect(ready)).toBeCloseTo(4 / 3);
  const counts = { kind: "counts" as const, ms: 1, counts: {}, points: [[0.5, 0.5, "bottle"], [0, 1, "bottle"]] as [number, number, string][] };
  const photo = { w: 1600, h: 1200 };
  expect(boardGuess(counts, ready, photo).points).toEqual([[0.5, 0.5, "bottle"], [0, 1, "bottle"]]);
  const old = { kind: "ready" as const, input: 96, labels: [], firmware: "1.1.0", crc: "0badf00d" };
  expect(boardAspect(old)).toBe(1);
  expect(boardGuess(counts, old, photo).points).toEqual([[0.5, 0.5, "bottle"], [0.125, 1, "bottle"]]);
});

it("writes only what the board is missing", () => {
  const both = { firmware: true, model: true };
  expect(planWrite({ firmware: "1.1.0", crc: "0badf00d" }, "1.1.0", "0badf00d")).toEqual({ firmware: false, model: false });
  expect(planWrite({ firmware: "1.1.0", crc: "12345678" }, "1.1.0", "0badf00d")).toEqual({ firmware: false, model: true });
  expect(planWrite({ firmware: "1.0.0", crc: "0badf00d" }, "1.1.0", "0badf00d")).toEqual({ firmware: true, model: false });
  expect(planWrite({ firmware: "1.1.0" }, "1.1.0", "0badf00d")).toEqual({ firmware: false, model: true });  // no model yet
  expect(planWrite({}, "1.1.0", "0badf00d")).toEqual(both);  // firmware before 1.1.0
  expect(planWrite(null, "1.1.0", "0badf00d")).toEqual(both);
  expect(planWrite({ firmware: "1.1.0", crc: "0badf00d" }, "1.1.0", "0badf00d", true)).toEqual(both);
  expect(parseLine('{"event":"error","message":"no model flashed","firmware":"1.1.0"}'))
    .toEqual({ kind: "error", message: "no model flashed", firmware: "1.1.0" });
});

it("a ready line with bytes lost still says which firmware and model the board runs", () => {
  // As the board's line arrived on 2026-10-28, '"task":"classification","input":96' lost on the way.
  const line = '{"event":"ready","firmware":"1.2.0","crc":"b6c43ecd","task":"cla,"frame":[320,240],"channels":3,"grid":0,' +
    '"arena":205184,"labels":["trash_container","street"]}';
  const ev = parseLine(line);
  expect(ev.kind).toBe("ready");
  if (ev.kind !== "ready") return;
  expect(ev.firmware).toBe("1.2.0");
  expect(ev.crc).toBe("b6c43ecd");
  expect(ev.labels).toEqual(["trash_container", "street"]);
  expect(ev.frame).toEqual([320, 240]);
  expect(boardMismatch(ev, "b6c43ecd", "1.2.0")).toBeNull();
  expect(parseLine('{"label":"street","score":0.95,"ms":1211,"scores":{"trash_contaitreet":0.95},"jpeg":"/9j').kind).toBe("label");
});

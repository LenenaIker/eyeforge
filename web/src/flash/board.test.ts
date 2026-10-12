import { expect, it } from "vitest";
import raw from "../../public/firmware/esp32/manifest.json";
import { checkModel, flashable, friendlyError, lineSplitter, parseLine, parseManifest } from "./board";

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

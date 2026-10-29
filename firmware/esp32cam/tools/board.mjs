#!/usr/bin/env node
// Flash and watch the ESP32-CAM with esptool-js, the library the web app uses.
//
//   node board.mjs [--usb] [--baud N] [--port /dev/ttyUSB0] <command>
//     detect                      chip, flash size and MAC
//     backup <out.bin> [bytes]    read the flash (default 4 MB); serial only, see below
//     write <addr>=<file> ...     write files, e.g. 0x10000=firmware.bin 0x190000=model.bin
//     monitor [seconds]           reset the board and print its serial output (default 20 s) at --mbaud
//                                 (default 115200, the firmware's speed after boot; the ROM's lines are garbled)
//     monitor [seconds] --send "capture uxga" --at 15   also send a line to the firmware after --at seconds
//
// Without --usb it goes through Web Serial (what Chrome and Edge use on a computer), here node-serialport.
// With --usb it goes through esptool-js's WebUSB CH340 driver (what Chrome on Android uses), here node-usb.
// Linux's ch341 driver owns the board, as Android has none it is detached for the run and attached again.
// Known WebUSB driver bugs in esptool-js 0.7.0 (2026-10-10): readFlash stalls after 3 blocks, and closing
// fails with "pending request" (harmless: the board is already flashed and reset).
import fs from "node:fs";
import { ESPLoader, Transport, WebUSBSerialPort } from "esptool-js";
import { SerialPort } from "serialport";
import { findByIds, WebUSB } from "usb";
import { NodeSerialPort } from "./webserial_shim.mjs";

const CH340 = [0x1a86, 0x7523];
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  if (i < 0) return fallback;
  const [value] = args.splice(i, 2).slice(1);
  return value;
};
const useUsb = args.includes("--usb") && args.splice(args.indexOf("--usb"), 1);
const baud = Number(flag("--baud", 460800));
const path = flag("--port", "/dev/ttyUSB0");
const monitorBaud = Number(flag("--mbaud", 115200));
const sendLine = flag("--send", null);
const sendAt = Number(flag("--at", 15));
const [command, ...rest] = args;
const terminal = { clean() {}, writeLine: l => console.log(l), write: t => process.stdout.write(t) };
const sleep = ms => new Promise(r => setTimeout(r, ms));

function kernelDriver(attach) {
  const dev = findByIds(...CH340);
  if (!dev) throw new Error("no CH340 board on USB");
  dev.open();
  const iface = dev.interface(0);
  const active = iface.isKernelDriverActive();
  if (attach && !active) iface.attachKernelDriver();
  if (!attach && active) iface.detachKernelDriver();
  dev.close();
  return active;
}

async function reattach() {
  for (let i = 0; i < 10; i++) {
    try {
      kernelDriver(true);
      return;
    } catch {
      await sleep(300);  // the WebUSB adapter may still hold a pending transfer
    }
  }
  console.error("could not give the board back to Linux's ch341 driver: replug it");
}

async function openLoader() {
  let port;
  if (useUsb) {
    kernelDriver(false);
    const dev = (await new WebUSB({ allowAllDevices: true }).getDevices()).find(d => d.vendorId === CH340[0]);
    port = new WebUSBSerialPort(dev);
  } else {
    port = new NodeSerialPort(path, { usbVendorId: CH340[0], usbProductId: CH340[1] });
  }
  const transport = new Transport(port, !!process.env.TRACE);
  const loader = new ESPLoader({ transport, baudrate: baud, terminal });
  console.log("chip:", await loader.main());
  return { loader, transport };
}

async function finish({ loader, transport }) {
  await loader.after("hard_reset");
  try {
    await transport.disconnect();
  } catch (e) {
    console.error(`(close: ${e.message})`);
  }
}

async function monitor(seconds) {
  const port = new SerialPort({ path, baudRate: monitorBaud, hupcl: false, autoOpen: false });
  await new Promise((ok, ko) => port.open(e => (e ? ko(e) : ok())));
  const set = s => new Promise(ok => port.set(s, ok));
  await set({ dtr: false, rts: true });  // pulse EN for a clean boot, IO0 high
  await sleep(120);
  await set({ dtr: false, rts: false });
  // A photo's base64 is a huge line: show only its start.
  let pending = "";
  port.on("data", d => {
    const lines = (pending + d.toString("latin1")).split("\n");
    pending = lines.pop();
    for (const l of lines) console.log(l.replace(/("jpeg":")([A-Za-z0-9+/=]{40})[A-Za-z0-9+/=]*/, "$1$2..."));
  });
  if (sendLine) {
    await sleep(sendAt * 1000);
    console.log(`\n>>> ${sendLine}`);
    await new Promise(ok => port.write(sendLine + "\n", ok));
    await sleep(Math.max(0, seconds - sendAt) * 1000);
  } else {
    await sleep(seconds * 1000);
  }
  await new Promise(ok => port.close(ok));
}

const commands = {
  async detect() {
    const session = await openLoader();
    console.log("flash id:", (await session.loader.readFlashId()).toString(16));
    await finish(session);
  },
  async backup(out, bytes = 4 * 1024 * 1024) {
    if (!out) throw new Error("backup needs an output file");
    if (useUsb) console.warn("warning: readFlash over WebUSB stalls in esptool-js 0.7.0, drop --usb");
    const session = await openLoader();
    const t0 = Date.now();
    let last = 0;
    const image = await session.loader.readFlash(0, Number(bytes), (_p, got, total) => {
      if (got - last >= 512 * 1024) {
        last = got;
        console.log(`read ${got}/${total}`);
      }
    });
    fs.writeFileSync(out, image);
    console.log(`read ${image.length} bytes in ${(Date.now() - t0) / 1000} s`);
    await finish(session);
  },
  async write(...pairs) {
    if (!pairs.length) throw new Error("write needs <addr>=<file> pairs");
    const fileArray = pairs.map(p => {
      const [address, file] = p.split("=");
      return { address: Number(address), data: new Uint8Array(fs.readFileSync(file)) };
    });
    const session = await openLoader();
    const t0 = Date.now();
    await session.loader.writeFlash({
      fileArray, flashMode: "keep", flashFreq: "keep", flashSize: "keep", eraseAll: false, compress: true,
      reportProgress: (i, written, total) => written === total && console.log(`${pairs[i]}: ${total} bytes`),
    });
    console.log(`written in ${(Date.now() - t0) / 1000} s`);
    await finish(session);
  },
  monitor: (seconds = 20) => monitor(Number(seconds)),
};

if (!commands[command]) {
  console.log(fs.readFileSync(new URL(import.meta.url), "utf8").split("\n").slice(1, 9).map(l => l.slice(3)).join("\n"));
  process.exit(command ? 1 : 0);
}
try {
  await commands[command](...rest);
} finally {
  if (useUsb) await reattach();
}
process.exit(0);

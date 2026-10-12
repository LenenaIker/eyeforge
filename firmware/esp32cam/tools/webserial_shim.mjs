// A Web Serial SerialPort on top of node-serialport, so esptool-js's Transport runs in Node
// exactly as it does in a desktop browser.
import { SerialPort } from "serialport";

export class NodeSerialPort {
  constructor(path, info = {}) {
    this.path = path;
    this.info = info;
    this.readable = null;
    this.writable = null;
  }

  getInfo() {
    return this.info;
  }

  async open({ baudRate }) {
    this.port = new SerialPort({ path: this.path, baudRate, autoOpen: false });
    await new Promise((ok, ko) => this.port.open(e => (e ? ko(e) : ok())));
    const port = this.port;
    this.readable = new ReadableStream({
      start(c) {
        port.on("data", d => c.enqueue(new Uint8Array(d)));
        port.on("close", () => {
          try { c.close(); } catch {}
        });
      },
    });
    this.writable = new WritableStream({
      write: chunk => new Promise((ok, ko) => port.write(Buffer.from(chunk), e => (e ? ko(e) : port.drain(ok)))),
    });
  }

  async setBaudRate(baudRate) {
    await new Promise((ok, ko) => this.port.update({ baudRate }, e => (e ? ko(e) : ok())));
  }

  async setSignals(s) {
    const o = {};
    if (s.dataTerminalReady !== undefined) o.dtr = s.dataTerminalReady;
    if (s.requestToSend !== undefined) o.rts = s.requestToSend;
    if (s.break !== undefined) o.brk = s.break;
    await new Promise((ok, ko) => this.port.set(o, e => (e ? ko(e) : ok())));
  }

  async close() {
    await new Promise(ok => this.port.close(() => ok()));
    this.readable = this.writable = null;
  }
}

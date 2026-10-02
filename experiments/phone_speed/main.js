// Every model of export.py on every runtime and accelerator the phone has: load time, then milliseconds per
// photo (median and p90 after a warm-up), including reading the output back, as the live test does. The runtimes
// are loaded like the web app loads LiteRT.js: no cross-origin isolation, so WebAssembly runs on one thread.
import { loadAndCompile, loadLiteRt, Tensor } from "@litertjs/core";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";

const SIZE = 224, WARMUP = 5, BUDGET_MS = 10_000;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1;

// The same input for every model, so their outputs can be compared (same backbone, same untrained head).
const input = new Float32Array(SIZE * SIZE * 3);
for (let y = 0; y < SIZE; y++)
  for (let x = 0; x < SIZE; x++) {
    const i = (y * SIZE + x) * 3;
    input[i] = (x * 255) / SIZE; input[i + 1] = (y * 255) / SIZE; input[i + 2] = ((x ^ y) & 255);
  }

const $ = (id) => document.getElementById(id);
const status = (s) => ($("status").textContent = s);

async function bench(run) {
  const first = performance.now();
  let out = await run();
  const one = performance.now() - first;
  for (let i = 1; i < WARMUP; i++) out = await run();
  const n = Math.max(5, Math.min(30, Math.ceil(BUDGET_MS / Math.max(one, 1))));
  const t = [];
  for (let i = 0; i < n; i++) {
    const s = performance.now();
    out = await run();
    t.push(performance.now() - s);
  }
  t.sort((a, b) => a - b);
  return { median: t[Math.floor(n / 2)], p90: t[Math.floor(n * 0.9)], runs: n, first: one,
           out: Array.from(out, (v) => +v.toFixed(4)) };
}

async function litert(bytes, accelerator) {
  const t0 = performance.now();
  const model = await loadAndCompile(bytes, { accelerator });
  const load = performance.now() - t0;
  const r = await bench(async () => {
    const x = new Tensor(input, [1, SIZE, SIZE, 3]);
    const outs = await model.run(x);
    x.delete();
    const o = accelerator === "wasm" ? outs[0] : await outs[0].moveTo("wasm");
    const data = new Float32Array(o.toTypedArray());
    if (o !== outs[0]) o.delete();
    outs.forEach((t) => t.delete());
    return data;
  });
  model.delete?.();
  return { load, ...r };
}

async function onnx(bytes, ep) {
  const t0 = performance.now();
  const s = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
  const load = performance.now() - t0;
  const x = new ort.Tensor("float32", input, [1, SIZE, SIZE, 3]);
  const r = await bench(async () => (await s.run({ [s.inputNames[0]]: x }))[s.outputNames[0]].data);
  await s.release();
  return { load, ...r };
}

async function main() {
  const device = $("device").value.trim() || "phone";
  localStorage.setItem("device", device);
  $("go").disabled = true;
  const rows = [], gpu = "gpu" in navigator && !!(await navigator.gpu.requestAdapter().catch(() => null));
  await loadLiteRt("/litert/");
  const files = Object.keys(await (await fetch("/models.json")).json());
  const jobs = [];
  for (const f of files) {
    const [name, ext] = f.split(".");
    const accs = gpu ? ["webgpu", "wasm"] : ["wasm"];
    const rt = ext === "tflite" ? "litert" : "ort";
    if (!params.has("only") || params.get("only") === rt) for (const a of accs) jobs.push({ f, name, rt, a });
  }
  for (const [i, j] of jobs.entries()) {
    status(`${i + 1}/${jobs.length}: ${j.name} on ${j.rt} ${j.a}`);
    const row = { model: j.name, runtime: `${j.rt}-${j.a}` };
    try {
      const bytes = new Uint8Array(await (await fetch(`/models/${j.f}`)).arrayBuffer());
      Object.assign(row, j.rt === "litert" ? await litert(bytes, j.a) : await onnx(bytes, j.a));
    } catch (e) {
      row.error = String(e?.message ?? e).slice(0, 300);
    }
    rows.push(row);
    const top = row.out ? row.out.indexOf(Math.max(...row.out)) + ":" + Math.max(...row.out).toFixed(2) : "";
    $("rows").insertAdjacentHTML("beforeend", `<tr><td>${row.model}</td><td>${row.runtime}</td>` +
      (row.error ? `<td colspan="4">${row.error}</td>` :
        `<td>${row.load.toFixed(0)}</td><td>${row.median.toFixed(1)}</td><td>${row.p90.toFixed(1)}</td><td>${top}</td>`) +
      "</tr>");
  }
  const report = { device, time: new Date().toISOString(), userAgent: navigator.userAgent, webgpu: gpu,
                   cores: navigator.hardwareConcurrency, rows };
  await fetch("/result", { method: "POST", body: JSON.stringify(report) });
  status("Done, results saved.");
  $("go").disabled = false;
}

const params = new URLSearchParams(location.search);
try { $("device").value = params.get("device") ?? localStorage.getItem("device") ?? ""; } catch { /* no storage */ }
$("go").onclick = () => main().catch((e) => status(`Failed: ${e?.message ?? e}`));
if (params.has("auto")) $("go").click();

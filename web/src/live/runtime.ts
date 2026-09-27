// The trained model running in the browser with LiteRT.js: WebGPU when the phone has it, else WebAssembly on
// the CPU. The model and its report are kept in the Cache API per job, so the live test works offline after
// the first download (the WebAssembly runtime itself is cached by the service worker).
import { loadAndCompile, loadLiteRt, Tensor, type CompiledModel } from "@litertjs/core";
import { fileUrl, type Report } from "../api";
import { CACHE } from "./saved";

let runtime: Promise<unknown> | null = null;

export type LiveModel = {
  report: Report & { output?: { threshold: number; stride_px: number } };
  size: number;          // input side in pixels (square)
  accelerator: "webgpu" | "wasm";
  run: (rgb: Float32Array) => Promise<Float32Array>;  // size x size x 3, RGB 0-255, row-major
  close: () => void;
};

async function cached(job: string, file: string): Promise<Response> {
  const url = fileUrl(job, file);
  const cache = await caches.open(CACHE);
  const hit = await cache.match(url);
  if (hit) return hit;
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.status === 404 ? "This model is not ready or no longer exists." : r.statusText);
  await cache.put(url, r.clone());
  return r;
}

/** The report from the server when there is a network, else the saved one. A report that changed (the model
 * was trained again) drops the saved model, so the phone never tests an old one. */
async function freshReport(job: string): Promise<any> {
  const cache = await caches.open(CACHE);
  const url = fileUrl(job, "report.json");
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (r.ok) {
      const text = await r.clone().text();
      const old = await cache.match(url);
      if (!old || (await old.text()) !== text) {
        await cache.delete(fileUrl(job, "model.tflite"));
        await cache.put(url, r);
      }
      return JSON.parse(text);
    }
  } catch { /* offline: use what is saved */ }
  return (await cached(job, "report.json")).json();
}

export async function loadModel(job: string): Promise<LiveModel> {
  // A failed script download can leave the runtime promise pending forever: give up after a while, and let the
  // next attempt start over instead of reusing the failed one.
  runtime ??= Promise.race([loadLiteRt("/litert/"), new Promise((_, no) => setTimeout(
    () => no(new Error("Could not download the model runtime. Check the connection and try again.")), 60_000))]);
  try { await runtime; } catch (e) { runtime = null; throw e; }
  const report = await freshReport(job);
  if (!report.portable_model_path) throw new Error("This model was trained before live testing existed: train it again.");
  const bytes = new Uint8Array(await (await cached(job, "model.tflite")).arrayBuffer());
  const size = report.input[0];
  let model: CompiledModel, accelerator: LiveModel["accelerator"] = "webgpu";
  try {
    if (!("gpu" in navigator)) throw new Error("no WebGPU");
    model = await loadAndCompile(bytes, { accelerator: "webgpu" });
  } catch {
    accelerator = "wasm";
    model = await loadAndCompile(bytes, { accelerator: "wasm" });
  }
  return {
    report, size, accelerator,
    run: async (rgb) => {
      const input = new Tensor(rgb, [1, size, size, 3]);
      const outs = await model.run(input);
      input.delete();
      const out = accelerator === "wasm" ? outs[0] : await outs[0].moveTo("wasm");
      const data = new Float32Array(out.toTypedArray() as Float32Array);  // copy: the tensor memory is freed next
      if (out !== outs[0]) out.delete();
      outs.forEach((t) => t.delete());
      return data;
    },
    close: () => model.delete(),
  };
}

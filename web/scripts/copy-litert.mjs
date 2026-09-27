// LiteRT.js loads its WebAssembly runtime at run time from /litert/: copy the two builds it picks from
// (relaxed SIMD or not; the threaded and JSPI ones need browser features we do not ask for) into public/.
import { copyFileSync, mkdirSync } from "node:fs";

const src = "node_modules/@litertjs/core/wasm", dst = "public/litert";
mkdirSync(dst, { recursive: true });
for (const f of ["litert_wasm_internal", "litert_wasm_compat_internal"])
  for (const ext of [".js", ".wasm"]) copyFileSync(`${src}/${f}${ext}`, `${dst}/${f}${ext}`);

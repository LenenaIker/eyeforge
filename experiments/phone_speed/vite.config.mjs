// Phone speed test server: serves the page and the models of export.py, and saves the results the phone posts.
//   web/node_modules/.bin/vite --config experiments/phone_speed/vite.config.mjs
//   adb reverse tcp:5199 tcp:5199, then open http://localhost:5199/ on the phone (localhost is a secure
//   context, which WebGPU needs).
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const out = resolve(root, "runs/bench/phone_speed");

export default {
  root: here,
  publicDir: resolve(out, "public"),
  resolve: { alias: { "@litertjs/core": resolve(root, "web/node_modules/@litertjs/core") } },
  server: { port: 5199, strictPort: true, fs: { allow: [root] } },
  plugins: [{
    name: "save-results",
    configureServer(server) {
      server.middlewares.use("/result", (req, res) => {
        let body = "";
        req.on("data", (c) => (body += c));
        req.on("end", () => {
          mkdirSync(out, { recursive: true });
          const r = JSON.parse(body);
          writeFileSync(resolve(out, `${r.device}-${r.time.replace(/[:.]/g, "-")}.json`), JSON.stringify(r, null, 1));
          for (const row of r.rows) console.log(JSON.stringify(row));
          res.end("ok");
        });
      });
    },
  }],
};

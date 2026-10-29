import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";

// The build goes inside the Python package, which serves it (sim2chip/server.py). In development, `npm run dev`
// serves the app and forwards /api to `sim2chip serve` on port 8000.
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    // Installable app (Android, iOS, desktop) without an app store. The service worker caches only the app
    // itself: /api always goes to the network, so progress and results are never stale.
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["logo.svg", "apple-touch-icon.png"],
      manifest: {
        name: "EyeForge",
        short_name: "EyeForge",
        description: "Describe what your camera should see. Get a tiny model ready to run on your own device.",
        start_url: "/app",
        scope: "/",
        display: "standalone",
        background_color: "#ffffff",
        theme_color: "#ffffff",
        icons: [
          { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,svg,png,woff2}"],
        // Fonts load by unicode range: the browser fetches other alphabets only when a page uses them.
        globIgnores: ["**/inter-{cyrillic,cyrillic-ext,greek,greek-ext,vietnamese}-*", "litert/**", "assets/esptool/**"],
        // The model runtime (9 MB of WebAssembly) is fetched the first time someone tests a model live, then
        // kept for offline use; it is not part of the app's first download.
        // Same for the flasher (esptool-js, 300 KB with every chip's loader): only people who flash a board need it.
        // The board firmware (/firmware/*.bin) is never cached: it always comes from the server.
        runtimeCaching: [{ urlPattern: /\/litert\//, handler: "CacheFirst", options: { cacheName: "litert-runtime" } },
                         { urlPattern: /\/assets\/esptool\//, handler: "CacheFirst", options: { cacheName: "esptool" } }],
        navigateFallback: "/index.html",
        navigateFallbackDenylist: [/^\/api\//, /^\/auth\//],  // the login redirects must reach the server
      },
    }),
  ],
  build: {
    outDir: "../sim2chip/web/dist", emptyOutDir: true,
    // esptool-js chunks go to their own folder so the service worker can leave them out of the first download
    rolldownOptions: { output: { chunkFileNames: (c) =>
      c.moduleIds.some((id) => id.includes("/node_modules/esptool-js/")) ? "assets/esptool/[name]-[hash].js" : "assets/[name]-[hash].js" } },
  },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});

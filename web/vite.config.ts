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
        description: "Describe what your camera should see. Get a tiny model ready to flash on a $10 chip.",
        start_url: "/app",
        scope: "/",
        display: "standalone",
        background_color: "#fafaf9",
        theme_color: "#fafaf9",
        icons: [
          { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,svg,png,woff2}"],
        // Fonts load by unicode range: the browser fetches other alphabets only when a page uses them.
        globIgnores: ["**/inter-{cyrillic,cyrillic-ext,greek,greek-ext,vietnamese}-*"],
        navigateFallback: "/index.html",
        navigateFallbackDenylist: [/^\/api\//],
      },
    }),
  ],
  build: { outDir: "../sim2chip/web/dist", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});

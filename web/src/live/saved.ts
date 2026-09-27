// Models saved on this device for the live test (Cache API), apart from the runtime so the app shell can list
// them without loading LiteRT.js.
export const CACHE = "eyeforge-models";

/** Jobs whose model is saved on this device (usable without a network). */
export async function savedJobs(): Promise<Set<string>> {
  if (!("caches" in window)) return new Set();
  const keys = await (await caches.open(CACHE)).keys();
  return new Set(keys.map((k) => new URL(k.url).pathname.split("/")[3]));
}

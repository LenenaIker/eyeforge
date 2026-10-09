// Frames the user saved during the live test ("this one is wrong"), kept per job in IndexedDB until they are
// sent with "Make it better". They must survive leaving the page and a phone without signal, so not memory.
// Storage can be missing (private windows, blocked site data): every call then quietly does nothing.

// said: what the user said the frame is when they took it (a class, or every object as [x, y, class] on the
// upright frame, 0-1); it wins over the teacher's marks. None: the teacher marks it.
export type Said = { label: string } | { points: [number, number, string][] };
export type Capture = { id: number; job: string; blob: Blob; t: number; said?: Said };

const DB = "eyeforge-captures", STORE = "captures";

function open(): Promise<IDBDatabase> {
  return new Promise((ok, fail) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE, { keyPath: "id", autoIncrement: true }).createIndex("job", "job");
    };
    req.onsuccess = () => ok(req.result);
    req.onerror = () => fail(req.error);
  });
}

async function run<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((ok, fail) => {
      const req = fn(db.transaction(STORE, mode).objectStore(STORE));
      req.onsuccess = () => ok(req.result);
      req.onerror = () => fail(req.error);
    });
  } finally {
    db.close();
  }
}

export async function addCapture(job: string, blob: Blob, said?: Said): Promise<boolean> {
  try { await run("readwrite", (s) => s.add({ job, blob, t: Date.now(), ...(said ? { said } : {}) })); return true; } catch { return false; }
}

export async function listCaptures(job: string): Promise<Capture[]> {
  try { return await run("readonly", (s) => s.index("job").getAll(job)); } catch { return []; }
}

export async function removeCapture(id: number): Promise<void> {
  try { await run("readwrite", (s) => s.delete(id)); } catch { /* storage unavailable */ }
}

export async function clearCaptures(job: string): Promise<void> {
  for (const c of await listCaptures(job)) await removeCapture(c.id);
}

export const captureFile = (c: Capture) => new File([c.blob], `live_${c.t}.jpg`, { type: "image/jpeg" });

/** What the user said of each capture, by the file name captureFile gives it (sent with the photos). */
export const captureLabels = (caps: Capture[]) =>
  Object.fromEntries(caps.filter((c) => c.said).map((c) => [captureFile(c).name, c.said!]));

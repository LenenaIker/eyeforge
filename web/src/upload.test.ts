import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Photos are "shrunk" as they are (no canvas here) and keyed by their name; "bad" ones cannot be read.
const counts = vi.hoisted(() => ({ shrunk: 0 }));
vi.mock("./images", () => ({
  shrink: async (f: File) => { counts.shrunk++; return f.name.startsWith("bad") ? null : f; },
  contentKey: async (f: File) => `k_${f.name}`,
}));

import { addDatasetPhotos, addPhotos, createDataset, storePhotos, UploadError } from "./api";

type Sent = { url: string; names: string[]; fields: Record<string, string> };
let sent: Sent[], known: string[], failures: Record<string, number>, flying: number, most: number;

// A stand-in XMLHttpRequest: records what went up, fails a batch on the network while failures[its first photo]
// lasts, and tells how many were on their way at once.
class FakeXhr {
  upload: { onprogress?: (e: any) => void } = {};
  onload?: () => void; onerror?: () => void; ontimeout?: () => void;
  status = 0; statusText = ""; responseText = "";
  private url = "";
  open(_m: string, url: string) { this.url = url; }
  send(form: FormData) {
    const files = form.getAll("photos") as File[];
    const fields = Object.fromEntries([...form.entries()].filter(([k]) => k !== "photos").map(([k, v]) => [k, String(v)]));
    flying++;
    most = Math.max(most, flying);
    setTimeout(() => {
      flying--;
      const first = files[0]?.name;
      if (failures[first]) { failures[first]--; this.onerror?.(); return; }
      sent.push({ url: this.url, names: files.map((f) => f.name), fields });
      this.status = 200;
      this.responseText = JSON.stringify(this.url === "/api/library/datasets" ? { id: "ds1" } :
        fields.more === "false" ? { rerun_from: "label", photos: Number(fields.total) } : { stored: files.length });
      this.upload.onprogress?.({ lengthComputable: true, loaded: 1, total: 1 });
      this.onload?.();
    }, 5);
  }
}

const photo = (name: string, mb = 3) => new File([new Uint8Array(mb * 1_000_000)], name, { type: "image/jpeg" });
const photos = (n: number, mb = 3) => Array.from({ length: n }, (_, i) => photo(`p${String(i).padStart(2, "0")}.jpg`, mb));

// Runs an upload on the fake clock (the retries wait seconds), its error returned instead of thrown.
async function run(p: Promise<unknown>): Promise<any> {
  let done = false;
  const out = p.catch((e) => e).finally(() => { done = true; });
  while (!done) await vi.advanceTimersByTimeAsync(50);
  return out;
}

beforeEach(() => {
  sent = []; known = []; failures = {}; flying = 0; most = 0; counts.shrunk = 0;
  vi.stubGlobal("XMLHttpRequest", FakeXhr);
  vi.stubGlobal("fetch", async (_url: string, opts: RequestInit) => {
    const { hashes } = JSON.parse(String(opts.body));
    return new Response(JSON.stringify({ known: hashes.filter((h: string) => known.includes(h)) }));
  });
  vi.useFakeTimers({ toFake: ["setTimeout"] });
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("photo upload", () => {
  it("sends small batches side by side, none bigger than the cap", async () => {
    const r = await run(storePhotos("s1", photos(12), "training", null));
    expect(sent.flatMap((s) => s.names).sort()).toEqual(photos(12).map((p) => p.name));
    expect(Math.max(...sent.map((s) => s.names.length))).toBeLessThanOrEqual(3);  // 3 MB photos, 8 MB batches
    expect(most).toBeGreaterThan(1);
    expect(sent.every((s) => s.fields.more === "true")).toBe(true);
    expect(r.unreadable).toEqual([]);
  });

  it("stops shrinking while enough batches wait to go up", async () => {
    let ahead = 0;
    const p = storePhotos("s1", photos(60), "training", null);
    const out = p.finally(() => {});
    for (let i = 0; i < 400 && counts.shrunk < 60; i++) {
      await vi.advanceTimersByTimeAsync(1);
      ahead = Math.max(ahead, counts.shrunk - sent.flatMap((s) => s.names).length);
    }
    await run(out);
    expect(sent.flatMap((s) => s.names)).toHaveLength(60);
    expect(ahead).toBeLessThanOrEqual(3 * (3 * 2 + 3 + 1));  // 6 waiting + 3 flying + the one being built
  });

  it("tries a batch lost on the network again", async () => {
    failures["p00.jpg"] = 2;
    await run(storePhotos("s1", photos(6), "training", null));
    expect(sent.flatMap((s) => s.names)).toContain("p00.jpg");
  });

  it("says how many did not get there, and the retry sends only those", async () => {
    failures["p00.jpg"] = 99;
    const err = await run(storePhotos("s1", photos(6), "training", null));
    expect(err).toBeInstanceOf(UploadError);
    expect(err.message).toMatch(/^3 of 6 photos did not upload/);
    known = sent.flatMap((s) => s.names).map((n) => `k_${n}`);
    sent = []; failures = {};
    await run(storePhotos("s1", photos(6), "training", null));
    expect(sent.flatMap((s) => s.names)).toEqual(["p00.jpg", "p01.jpg", "p02.jpg"]);
  });

  it("names the photos it cannot read and sends the rest", async () => {
    const r = await run(storePhotos("s1", [photo("a.jpg"), photo("bad.heic")], "training", null));
    expect(r.unreadable).toEqual(["bad.heic"]);
    expect(sent.flatMap((s) => s.names)).toEqual(["a.jpg"]);
  });

  it("acts on a model's photos once, after all the others are in, even when all were there", async () => {
    await run(addPhotos("job1", photos(9), "dusk"));
    const finals = sent.filter((s) => s.fields.more === "false");
    expect(finals).toHaveLength(1);
    expect(sent[sent.length - 1]).toBe(finals[0]);
    expect(finals[0].fields.total).toBe("9");
    known = photos(9).map((p) => `k_${p.name}`);
    sent = [];
    await run(addPhotos("job1", photos(9), "dusk"));
    expect(sent).toHaveLength(1);
    expect(sent[0].fields.more).toBe("false");
  });

  it("makes the dataset with the first batch and adds the rest to it; a failed one can be added to", async () => {
    const made = await run(createDataset("Shelf", photos(9)));
    expect(made.id).toBe("ds1");
    expect(sent[0].url).toBe("/api/library/datasets");
    expect(sent.slice(1).every((s) => s.url === "/api/library/datasets/ds1/photos")).toBe(true);
    sent = [];
    failures["p06.jpg"] = 99;
    const err = await run(createDataset("Shelf", photos(9)));
    expect(err.dataset).toBe("ds1");
    expect(err.message).toMatch(/^3 of 9 photos did not upload/);
    sent = []; failures = {}; known = photos(6).map((p) => `k_${p.name}`);
    await run(addDatasetPhotos("ds1", photos(9)));
    expect(sent.flatMap((s) => s.names)).toEqual(["p06.jpg", "p07.jpg", "p08.jpg"]);
  });
});

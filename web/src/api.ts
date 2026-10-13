// Typed client of the EyeForge API (sim2chip/server.py). Every call goes through `api`, which turns an error
// response into an Error carrying the server's message.

import { contentKey, shrink } from "./images";

export type Spec = Record<string, any> & {
  name: string;
  task?: string;
  hardware?: string;
  classes: { id: string; description?: string; assets?: Record<string, any> | null }[];
};

export type Candidate = { file: string; score: number; reason: string };
// One step of an interview turn (thinking, searching, checking the setup); times are the server's epoch seconds.
export type Step = { text: string; state: "running" | "done" | "failed"; started: number; ended?: number;
                     detail?: string; done?: number; total?: number };
export type Turn = {
  kind: "say" | "confirm" | "done" | "failed" | "photos" | "pick";
  text: string;
  intro?: string;  // confirm: the assistant's words without the spec summary, which the app shows as a card
  spec?: Spec;
  request?: { for?: "training" | "3d"; class?: string | null; candidates?: Candidate[] };
  photos?: number;
  activity?: Step[];  // what the turn did, step by step
};

export type Config = {
  backend: "local" | "nebius";
  engine: string;  // the EyeForge this server runs, e.g. 0.9.0+1c20634 (sim2chip/version.py)
  first_question: string;
  public: boolean;
  busy: boolean;
  max_runs: number | null;
  budget: { daily_usd: number; spent_usd: number; can_run: boolean } | null;
  locked: boolean;  // a public server whose API needs the access key, not given yet in this browser
  login: string[] | null;  // a public server with accounts: the providers to sign in with (google, github)
};

export type Me = {
  signed_in: boolean;
  contact: string | null;  // where judges write for free access, and banned accounts for help
  judging: boolean;        // the hackathon's judging is still on: judges use the app for free
  costs: Costs;
  topup: boolean;          // this server takes card payments (Stripe)
  account?: { email: string | null; name: string | null; provider: string;
              tier: "judge" | "normal" | "banned"; balance: number };
};

export type StageState = "done" | "running" | "pending" | "failed" | "skipped";
export type Stage = { name: string; runtime: string; state: StageState; powered_by: string; seconds: number | null;
  reason?: string };  // reason: why a skipped step was left out

export type RealTest = { metric: string; mean: number; std: number; count_mae?: number; exact?: number;
                         test_size?: number; objects?: number; per_seed?: Record<string, number> };
export type Report = {
  job: string; hardware: string; labels: string[]; background?: string | null; input: number[]; model_kb: number; task: string;
  val_score?: number; val_accuracy?: number; real_test?: RealTest; real_accuracy?: RealTest;
  c_array_path: string | null; portable_model_path?: string;
  device_blob_path?: string | null;  // model.bin, flashed on its own next to our firmware (TFLite Micro boards)
  quantization?: string;  // int8, or float16 for a count detector
  model_path?: string;  // the device model: model_int8.tflite, or model.tflite for a count detector
  // The shipped model's own score on the real test photos (the headline); real_test is the mean of the seeds.
  shipped_test?: { score: number; count_mae?: number } | null;
  advice?: string[];  // what most likely held the score back, in plain words
  left_out?: { stage: string; reason: string }[];  // steps that failed and the run went on without
  no_synthetic?: string;  // why this model learned from the user's photos alone
  // The model this one replaced, scored on the same test photos (or why it could not be).
  // then: its score when it was trained, which differs when the test photos or their labels changed since.
  previous?: { version: number; score?: number; now?: number; count_mae?: number; now_count_mae?: number;
               then?: number; skipped?: string } | null;
};

export type Version = { n: number; reason: string; created?: number; started?: number; metric?: string;
                        headline?: number | null; shipped?: number | null; test_size?: number | null;
                        model?: string;    // its device model file (older versions: model_int8.tflite)
                        engine?: string | null };  // the EyeForge that trained it (none before 2026-10-09)

// What a job used of Nebius and NVIDIA (sim2chip/usage.py): model calls and GPU time, in dollars when known.
export type ModelUsage = { model: string; name: string; steps: string[]; calls: number; input_tokens: number;
                           output_tokens: number; usd: number | null; nvidia: boolean; service?: string };
export type Usage = { models: ModelUsage[]; gpu: { where: string; minutes: number; usd: number }[]; usd: number;
                      priced: boolean };

export type Status = {
  job: string;
  title?: string | null;  // the name the user gave the model
  spec: Spec | null;
  running: boolean;
  calibration: { state: "labeling" | "ready" | "failed"; error?: string } | null;  // teacher check before the first run
  // A job that only labels a library dataset (Datasets page): checking the teacher, labeling every photo, done.
  labeling?: { dataset: string; labeling?: string; state: "checking" | "labeling" | "done" | "failed"; error?: string } | null;
  stages: Stage[];
  log: string[];
  cloud_usd: number;
  phase: string | null;
  timing: { started_at: number | null; elapsed_s: number | null; eta_s: number | null } | null;
  report: Report | null;
  no_synthetic?: string | null;  // why the model learns from the user's photos alone (no synthetic images)
  version: Version | null;  // the model export holds or the run in progress will make
  versions: Version[];      // earlier models, oldest first
  usage?: Usage;
};

export type Point = [number, number, string, ...number[]];
export type Review = {
  task: string;
  labels: string[];
  photos: { key: string; image: string | null; points?: Point[]; teacher_points?: Point[]; unsure?: boolean;
            label?: string; teacher?: string; confidence?: number; reason: string; judge: string;
            corrected: boolean }[];
  assets: { uid: string; class: string; role: string; name?: string; author?: string | null; generated: boolean; kept: boolean;
            decision?: string; reason: string; judge?: string; image: string | null }[];
  renders: string[];
  // count (count task): the model's count by class, from its density output; older models have none.
  test: { key: string; file: string; image: string; truth: any; pred: any; count?: Record<string, number> }[];
  teacher_rules?: string[];  // what the teacher learned from the user's corrections
};
export type Fixes = { labels: Record<string, string>; assets: Record<string, string>; points: Record<string, Point[]> };

// status: the HTTP code, so the app can tell "sign in" (401) and "balance too low" (402) from other errors.
export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

// Listeners of account changes (a 401 or 402 anywhere): the shell reloads who is signed in and their balance.
const accountListeners = new Set<() => void>();
export const onAccountChange = (f: () => void) => { accountListeners.add(f); return () => { accountListeners.delete(f); }; };

export async function api<T = any>(path: string, opts: RequestInit = {}): Promise<T> {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw failure(r.status, body, r.statusText);
  return body as T;
}

function failure(status: number, body: any, statusText = ""): ApiError {
  if (status === 401 || status === 402 || status === 403) accountListeners.forEach((f) => f());
  // Over HTTP/2 statusText is empty: an error without a detail (a server crash) still says something.
  return new ApiError(body?.detail || statusText || `the server failed (HTTP ${status}); please try again`, status);
}

// A request that failed on the network (status 0) or at the proxy in front of the server, not by the server's
// answer: worth sending again.
export const networkError = (e: unknown) => !(e instanceof ApiError) || [0, 502, 503, 504].includes(e.status);

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const getConfig = () => api<Config>("/api/config");
// The access key of a public server: right, the server sets a cookie and the API opens for this browser.
export const unlock = (key: string) => api<{ ok: boolean }>("/api/key", json({ key }));
export const getMe = () => api<Me>("/api/me");
export const loginUrl = (provider: string, next: string) =>
  `/auth/${provider}/login?next=${encodeURIComponent(next)}`;
export const logout = () => api<{ ok: boolean }>("/auth/logout", { method: "POST" });
// With a job: an interview that starts from that job's spec, to change something and retrain it. With a kit: a new
// model that starts from that kit of the user's library.
export const startInterview = (job?: string | null, kit?: string | null) => {
  const q = new URLSearchParams();
  if (job) q.set("job", job);
  if (kit) q.set("kit", kit);
  return api<{ session: string; text: string; job?: string }>(`/api/interview${q.size ? `?${q}` : ""}`, { method: "POST" });
};
// act: an id the caller makes up for the turn, to follow its steps with activity() while it runs.
export const reply = (sid: string, text: string, act?: string) =>
  api<Turn>(`/api/interview/${sid}${actQuery(act)}`, json({ text }));
export const pick = (sid: string, files: string[], cls: string, act?: string) =>
  api<Turn>(`/api/interview/${sid}/pick${actQuery(act)}`, json({ files, class: cls }));
// Once the turn is over, its answer too (turn, or error and status): the web's own request may have been cut.
export const activity = (sid: string, act: string) =>
  api<{ steps: Step[]; now: number; arrived?: boolean; turn?: Turn; error?: string; status?: number }>(
    `/api/interview/${sid}/activity?act=${encodeURIComponent(act)}`);
// The chat as the web shows it, kept by the server so it opens again in another tab or device (key: which
// conversation: a new model, a kit, a model being changed).
export const saveChatView = (sid: string, key: string, view: unknown) => api<{ ok: boolean }>(
  `/api/interview/${sid}/view`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key, view }) });
export const openChat = (key: string) =>
  api<{ session?: string; view?: any }>(`/api/interview/open?key=${encodeURIComponent(key)}`);
export const closeChats = (key: string) =>
  api<{ ok: boolean }>(`/api/interview/open?key=${encodeURIComponent(key)}`, { method: "DELETE" });
const actQuery = (act?: string) => (act ? `?act=${encodeURIComponent(act)}` : "");
export const thumbUrl = (sid: string, file: string) => `/api/interview/${sid}/thumb?file=${encodeURIComponent(file)}`;

// Photos go up shrunk (images.ts), in batches of about BATCH_BYTES, PARALLEL at a time, each tried again on a
// network error; photos the destination already has are not sent (a retry after a cut sends only what is
// missing). The public server's proxy refuses requests over 100 MB, and one lost batch of 40 MB lost 29 photos
// of a phone's chat upload (2026-10-13): small batches lose little and go up side by side.
const BATCH_BYTES = 8_000_000;
const PARALLEL = 3;
const RETRIES = [1000, 3000, 9000];  // waits before each new try of a batch that failed on the network

// Where the photos go, as /api/uploads/known names it, to skip the ones already there.
export type Where = { chat: string; purpose: string; cls: string } | { dataset: string } | { job: string };
// prepared: photos shrunk so far; sent: photos uploaded (fractions while a batch is on its way) or found there.
export type Progress = { prepared: number; sent: number; total: number };
export type Sent<T> = { answer?: T; unreadable: string[] };

// What the web says while photos go up: "Uploading 12 of 40 photos".
export const progressText = (p: Progress | null, what = "photos") =>
  p ? `Uploading ${Math.min(p.total, Math.floor(p.sent))} of ${p.total} ${what}` : `Uploading ${what}`;
// The photos the browser could not read, said after an upload that sent the rest.
export const unreadableText = (names: string[]) => names.length ?
  `${names.length} photo${names.length > 1 ? "s" : ""} could not be read in this browser and ${names.length > 1 ? "were" : "was"} left out (${names.slice(0, 3).join(", ")}${names.length > 3 ? ", ..." : ""}). Send ${names.length > 1 ? "them" : "it"} as JPEG.` : "";

export class UploadError extends Error {
  constructor(message: string, public dataset?: string) { super(message); }
}

// POST with upload progress (fetch does not tell how much of the body went), failing like api().
function post<T>(url: string, form: FormData, onSent?: (fraction: number) => void): Promise<T> {
  return new Promise((ok, fail) => {
    const x = new XMLHttpRequest();
    x.open("POST", url);
    x.upload.onprogress = (e) => { if (e.lengthComputable) onSent?.(e.loaded / e.total); };
    x.onload = () => {
      let body: any = {};
      try { body = JSON.parse(x.responseText); } catch { /* not JSON: a proxy's page */ }
      if (x.status >= 200 && x.status < 300) ok(body as T);
      else fail(failure(x.status, body, x.statusText));
    };
    x.onerror = x.ontimeout = () => fail(new ApiError("the connection failed", 0));
    x.send(form);
  });
}

async function knownPhotos(where: Where, keys: string[]): Promise<Set<string>> {
  try {
    return new Set((await api<{ known: string[] }>("/api/uploads/known", json({ ...where, hashes: keys }))).known);
  } catch {
    return new Set();  // only an optimisation: send them all
  }
}

type Batch = { files: File[]; keys: (string | null)[] };
type SendBatch = (files: File[], onSent: (fraction: number) => void) => Promise<any>;

// The upload itself. Photos are shrunk one by one while the batches already made go up. first: sends the first
// batch on its own before any other (it makes the dataset the others join; it returns where the rest go).
// final: sends the last batch once all the others are in (it carries the total and makes the server act on
// them); it always gets at least one photo.
async function sendPhotos<T>(files: File[], o: {
  where: Where | null; store: SendBatch; first?: (files: File[], onSent: (f: number) => void) => Promise<Where>;
  final?: SendBatch; onProgress?: (p: Progress) => void;
}): Promise<Sent<T>> {
  const total = files.length, unreadable: string[] = [];
  let where = o.where, prepared = 0, sent = 0, failed = 0, stop: unknown = null, producing = true;
  const flying = new Map<Batch, number>();
  const report = () => o.onProgress?.({ prepared, sent: sent + [...flying].reduce((a, [b, f]) => a + f * b.files.length, 0), total });
  const queue: Batch[] = [];

  async function go(b: Batch, send: SendBatch): Promise<any> {
    if (where) {  // the ones already there are counted as sent and left out
      const keys = b.keys.filter((k): k is string => !!k);
      const known = keys.length ? await knownPhotos(where, keys) : new Set<string>();
      const keep = b.keys.map((k) => !k || !known.has(k));
      if (send === o.final && !keep.some(Boolean)) keep[keep.length - 1] = true;
      sent += keep.filter((k) => !k).length;
      b = { files: b.files.filter((_, i) => keep[i]), keys: b.keys.filter((_, i) => keep[i]) };
      if (!b.files.length) { report(); return; }
    }
    for (let attempt = 0; ; attempt++) {
      flying.set(b, 0);
      try {
        const out = await send(b.files, (f) => { flying.set(b, f); report(); });
        flying.delete(b);
        sent += b.files.length;
        report();
        return out;
      } catch (e) {
        flying.delete(b);
        report();
        if (!networkError(e) || attempt >= RETRIES.length) throw e;
        await new Promise((ok) => setTimeout(ok, RETRIES[attempt]));
      }
    }
  }
  const worker = async () => {
    while (!stop) {
      const b = queue.shift();
      if (!b) { if (!producing) return; await new Promise((ok) => setTimeout(ok, 50)); continue; }
      try { await go(b, o.store); } catch (e) {
        if (networkError(e)) failed += b.files.length; else stop = e;
      }
    }
  };
  const makeFirst: SendBatch = async (part, onSent) => { where = await o.first!(part, onSent); };
  const workers: Promise<void>[] = [];
  let batch: Batch = { files: [], keys: [] }, size = 0, firstDone = !o.first, last: File | null = null;
  try {
    for (const f of files) {
      if (stop) break;
      const small = await shrink(f);
      prepared++;
      report();
      if (!small) { unreadable.push(f.name); sent++; continue; }
      batch.files.push(small);
      batch.keys.push(await contentKey(small));
      last = small;
      size += small.size;
      if (size < BATCH_BYTES) continue;
      if (!firstDone) {  // the first batch alone: the others need what it makes
        await go(batch, makeFirst);
        firstDone = true;
      } else {
        queue.push(batch);
        if (workers.length < PARALLEL) workers.push(worker());
      }
      batch = { files: [], keys: [] };
      size = 0;
    }
    if (!firstDone) {  // all in one batch
      if (!batch.files.length) throw new UploadError(`none of the photos could be read in this browser (${unreadable.join(", ")})`);
      await go(batch, makeFirst);
      batch = { files: [], keys: [] };
    }
    if (!o.final && batch.files.length) {
      queue.push(batch);
      batch = { files: [], keys: [] };
      if (workers.length < PARALLEL) workers.push(worker());
    }
    producing = false;
    await Promise.all(workers);
    if (stop) throw stop;
    const lost = (n: number) => new UploadError(`${n} of ${total} photos did not upload: the connection failed. ` +
      "Try again: the photos already uploaded are not sent twice.", where && "dataset" in where ? where.dataset : undefined);
    if (failed) throw lost(failed);
    if (!last) throw new UploadError(`none of the photos could be read in this browser (${unreadable.join(", ")})`);
    let answer: T | undefined;
    if (o.final) {
      if (!batch.files.length) batch = { files: [last], keys: [null] };
      answer = await go(batch, o.final).catch((e) => Promise.reject(networkError(e) ? lost(batch.files.length) : e));
    }
    return { answer, unreadable };
  } catch (e) {
    stop = stop ?? e;
    if (e instanceof UploadError || !(where && "dataset" in where)) throw e;
    throw new UploadError((e as Error).message, where.dataset);  // the dataset is made: a retry adds to it
  } finally {
    producing = false;
  }
}

const form = (fields: Record<string, string>, files: File[]) => {
  const f = new FormData();
  Object.entries(fields).forEach(([k, v]) => f.append(k, v));
  files.forEach((p) => f.append("photos", p, p.name));
  return f;
};

// Photos attached to a chat message: stored in the chat (the GPU warms up) without a turn of their own; the message
// sent next (reply, even with no text) tells the interview about them, so it answers once, seeing both.
export function storePhotos(sid: string, files: File[], purpose: string, cls: string | null, onProgress?: (p: Progress) => void) {
  const url = `/api/interview/${sid}/photos`, fields = { purpose, cls: cls || "", more: "true" };
  return sendPhotos<{ stored: number }>(files, { where: { chat: sid, purpose, cls: cls || "" }, onProgress,
    store: (part, onSent) => post(url, form(fields, part), onSent) });
}

// session: the interview that wrote the spec, so the job shows what it used.
export function createJob(spec: Spec, calibrate = false, session?: string | null) {
  const form = new FormData();
  form.append("spec", JSON.stringify(spec));
  if (session) form.append("session", session);
  if (calibrate) form.append("calibrate", "true");
  return api<{ job: string; photos: number; already_running?: boolean; calibrating?: boolean }>("/api/jobs", { method: "POST", body: form });
}

export type RevisePlan = { stages: string[]; changed: string[]; rerun_from: string | null };
export function reviseJob(job: string, spec: Spec, dry = false, session?: string | null) {
  const form = new FormData();
  form.append("spec", JSON.stringify(spec));
  if (session) form.append("session", session);
  form.append("dry", String(dry));
  return api<RevisePlan>(`/api/jobs/${job}/revise`, { method: "POST", body: form });
}

// labels: what the user said of some photos in the live test, by file name (live/captures.ts).
// The photos with a label are always sent: the label is applied to the photo that comes with it.
export function addPhotos(job: string, files: File[], note: string, labels: Record<string, unknown> = {},
                          onProgress?: (p: Progress) => void) {
  const url = `/api/jobs/${job}/photos`, fields = { note, labels: JSON.stringify(labels) };
  const labelled = files.filter((f) => f.name in labels), rest = files.filter((f) => !(f.name in labels));
  return sendPhotos<{ rerun_from: string; photos: number }>([...rest, ...labelled], {
    where: labelled.length ? null : { job }, onProgress,
    store: (part, onSent) => post(url, form({ ...fields, more: "true" }, part), onSent),
    final: async (part, onSent) => {
      try {
        return await post(url, form({ ...fields, more: "false", total: String(files.length) }, part), onSent);
      } catch (e) {  // a retry of a final batch whose first try got there: the run it started is running
        if (e instanceof ApiError && e.status === 409 && /running/.test(e.message)) return { rerun_from: "label", photos: files.length };
        throw e;
      }
    },
  });
}

export const listJobs = () => api<{ jobs: string[] }>("/api/jobs");
export const jobStatus = (job: string) => api<Status>(`/api/jobs/${job}`);
export type JobChat = { t: number; revision: boolean; msgs: { from: "bot" | "me" | "sys"; text: string }[] };
export const jobChat = (job: string) => api<{ chats: JobChat[] }>(`/api/jobs/${job}/chat`);
export const getReview = (job: string) => api<Review>(`/api/jobs/${job}/review`);
export const cancelJob = (job: string) => api(`/api/jobs/${job}/cancel`, { method: "POST" });
// A stopped or failed job again, from where it stopped (what finished is reused).
export const retryJob = (job: string) => api(`/api/jobs/${job}/retry`, { method: "POST" });
// calibrating: fixes of the teacher's check; the server refuses them once training started (another device).
export const applyFixes = (job: string, fixes: Fixes, calibrating = false) =>
  api<{ rerun_from: string | null }>(`/api/jobs/${job}/corrections`, json({ ...fixes, calibrating }));
export const trainCalibrated = (job: string, fixes: Fixes) =>
  api<{ rerun_from: null }>(`/api/jobs/${job}/train`, json(fixes));
export const labelAll = (job: string, fixes: Fixes) =>
  api<{ rerun_from: string }>(`/api/jobs/${job}/label_all`, json(fixes));
// The synthetic images of a job's last run (review.synthetic): what went into training, by photo2scene's phase
// (imitation of the user's place, or extrapolation past what the photo shows), and what each filter left out.
export type SyntheticShot = { image: string; scene?: string; phase?: "imitation" | "extrapolation"; similarity?: number };
export type Synthetic = {
  went_in: number; split: { imitation: number; extrapolation: number }; threshold: number | null;
  left_out: { likeness: number; review: number };
  reviews: { scene: string; score: number | null; keep: boolean; worst: string[] }[];
  samples: { went_in: SyntheticShot[]; likeness: SyntheticShot[]; review: SyntheticShot[] };
  no_synthetic: string | null;
};
export const getSynthetic = (job: string) => api<Synthetic>(`/api/jobs/${job}/synthetic`);
export const mediaUrl = (job: string, path: string, big = false) =>
  `/api/jobs/${job}/media/${path}${big ? "?size=1024" : ""}`;
export const fileUrl = (job: string, file: string) => `/api/jobs/${job}/files/${file}`;
export const versionFileUrl = (job: string, n: number, file: string) => `/api/jobs/${job}/versions/${n}/files/${file}`;

// The library (sim2chip/library.py): models are the visitor's jobs; datasets, objects and scenes are kept for good
// and kits are named lists of references to them.
export type ModelCard = {
  job: string; title?: string | null; created: number; running: boolean; task: string | null; hardware: string | null; classes: string[];
  labeling: { id: string; name: string; reviewed: number; photos: number } | null;
  model: { metric: string | null; score: number | null; kb: number | null; hardware: string | null } | null;
};
// What was seen in a dataset before any goal: objects with the share of photos they show up in, backgrounds.
export type DatasetProfile = { summary: string; photos: number; created: number;
                               objects: { name: string; share: number; count: number }[]; backgrounds: string[] };
// A labeling (docs/plan_etiquetado.md): the classes looked for in one dataset's photos and their marks. A photo
// the user reviewed has their marks as its truth. Marks are points in 0..1 of the upright photo (w, h: the box
// size when the teacher found one); classify marks have no point.
export type LabelClass = { id: number; key: string; description: string; color: string; phrases: string[]; marks: number };
export type Labeling = { id: string; dataset: string; name: string; task: "count" | "classify"; threshold: number | null;
                         copied_from: string | null; created: number; classes: LabelClass[]; photos: number;
                         reviewed: number; marked: number; models: string[] };
export type Mark = { class: string; x: number | null; y: number | null; w?: number | null; h?: number | null;
                     score?: number | null; source?: "teacher" | "user" };
export type LabelingPhoto = { id: number; hash: string; reviewed: boolean; marks: Mark[] };
export type DatasetCard = { id: string; name: string; created: number; photos: number; sample: number[];
                            labelings: Labeling[]; jobs: string[]; profile: DatasetProfile | null; profiling: boolean };
export type DatasetPage = DatasetCard & { photo_ids: number[] };
const send = (method: string, body: unknown): RequestInit =>
  ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
export const getDataset = (id: string) => api<DatasetPage>(`/api/library/datasets/${id}`);
// teach: the teacher starts on it right away (a few photos to check first): the job to follow.
export const createLabeling = (ds: string, name: string, task: "count" | "classify", classes: string[], teach: boolean) =>
  api<{ labeling: string; job?: string }>(`/api/library/datasets/${ds}/labelings`, send("POST", { name, task, classes, teach }));
export const getLabeling = (id: string) => api<Labeling & { photos: LabelingPhoto[] }>(`/api/library/labelings/${id}`);
export const teachLabeling = (id: string) => api<{ job: string }>(`/api/library/labelings/${id}/teach`, { method: "POST" });
export const renameLabeling = (id: string, name: string) => api<{ ok: boolean }>(`/api/library/labelings/${id}`, send("PATCH", { name }));
export const deleteLabeling = (id: string) => api<{ ok: boolean }>(`/api/library/labelings/${id}`, { method: "DELETE" });
export const copyLabeling = (id: string, name?: string) => api<{ labeling: string }>(`/api/library/labelings/${id}/copy`, send("POST", { name }));
// Class changes answer with the labeling that has them: a copy when a model was trained with this one.
export const addClass = (id: string, name: string, description = "") =>
  api<{ labeling: string }>(`/api/library/labelings/${id}/classes`, send("POST", { name, description }));
export const editClass = (id: string, cls: number, change: { key?: string; description?: string; names?: string[] }) =>
  api<{ labeling: string }>(`/api/library/labelings/${id}/classes/${cls}`, send("PATCH", change));
export const deleteClass = (id: string, cls: number) =>
  api<{ labeling: string }>(`/api/library/labelings/${id}/classes/${cls}`, { method: "DELETE" });
export const reviewPhoto = (id: string, photo: number, marks: Mark[]) =>
  api<{ ok: boolean }>(`/api/library/labelings/${id}/photos/${photo}`, send("PUT", { marks }));
export const unreviewPhoto = (id: string, photo: number) =>
  api<{ ok: boolean }>(`/api/library/labelings/${id}/photos/${photo}`, { method: "DELETE" });
export const deletePhoto = (photo: number) => api<{ ok: boolean }>(`/api/library/photos/${photo}`, { method: "DELETE" });
export const reprofileDataset = (id: string) => api<{ profiling: boolean }>(`/api/library/datasets/${id}/profile`, { method: "POST" });
export type ObjectCard = { id: string; name: string; concept: string; source: "photos" | "scene" | "upload";
                           has_file: boolean; thumb: boolean; kept: boolean; license?: string; author?: string;
                           url?: string; created: number; jobs: string[] };
export type SceneCard = { id: string; name: string; route: "support" | "surface" | "street" | "room" | "none";
                          status: "ready" | "failed"; why: string | null;  // why a failed one made no scene
                          photo_id: number | null; dataset: string | null; thumb: boolean; objects: string[];
                          jobs: string[]; created: number };
export type Kit = { id: string; name: string; datasets: string[]; objects: string[]; scenes: string[];
                    created: number; updated: number };
export type LibraryData = { models: ModelCard[]; datasets: DatasetCard[]; objects: ObjectCard[]; scenes: SceneCard[];
                            kits: Kit[] };
export const getLibrary = () => api<LibraryData>("/api/library");
export const libraryPhotoUrl = (id: number, big = false) => `/api/library/photos/${id}${big ? "?size=1024" : ""}`;
export const cardUrl = (kind: "object" | "scene", id: string) => `/api/library/${kind}/${id}/card`;

export type CreditMove = { amount: number; reason: string; detail: string | null; t: number };
// Things are charged by use: what Nebius, Token Factory and Tavily charge us times margin. start: the dollars the
// balance must hold to start each (about its usual price).
export type Costs = { margin: number; start: Record<"run" | "label" | "chat" | "profile", number> };
export const usd = (n: number) => `${n < 0 ? "-" : ""}$${Math.abs(n).toFixed(2)}`;
// Top-ups with Stripe: the saved card (after a first card payment) and the auto top-up that uses it.
export type AutoTopup = { available: boolean; on: boolean; below: number; amount: number; error: string | null };
export type Topup = { enabled: boolean; amounts: number[]; min: number; max: number; tax: boolean;
                      card: string | null; auto: AutoTopup };
export type Billing = { enabled: boolean; costs: Costs; topup: Topup | null;
                        tier?: "judge" | "normal" | "banned"; balance?: number; moves?: CreditMove[] };
export const getBilling = () => api<Billing>("/api/billing");
const sendJson = (method: string, body: unknown): RequestInit =>
  ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
// The Stripe Checkout page to go to; the balance grows when Stripe confirms the payment.
export const startTopup = (usd: number) => api<{ url: string }>("/api/billing/topup", sendJson("POST", { usd }));
export const setAutoTopup = (on: boolean, below: number, amount: number) =>
  api<Topup>("/api/billing/auto", sendJson("PUT", { on, below, amount }));
// Close the account: what is left of the paid balance goes back to the card, everything else is deleted.
export const deleteAccount = () => api<{ ok: boolean; refunded: number }>("/api/account/delete", sendJson("POST", { confirm: "delete" }));

// Editing the library. Photos go in batches like the chat's: the first batch makes the dataset, the rest join it.
export type LibraryKind = "datasets" | "objects" | "scenes" | "kits";
// A failed upload throws an UploadError with the dataset already made, if any, for the retry to add to.
export async function createDataset(name: string, files: File[], onProgress?: (p: Progress) => void) {
  let id = "";
  const r = await sendPhotos(files, { where: null, onProgress,
    first: async (part, onSent) => {
      id = (await post<{ id: string }>("/api/library/datasets", form({ name }, part), onSent)).id;
      return { dataset: id };
    },
    store: (part, onSent) => post(`/api/library/datasets/${id}/photos`, form({}, part), onSent) });
  return { id, unreadable: r.unreadable };
}
export async function addDatasetPhotos(id: string, files: File[], onProgress?: (p: Progress) => void) {
  return sendPhotos(files, { where: { dataset: id }, onProgress,
    store: (part, onSent) => post(`/api/library/datasets/${id}/photos`, form({}, part), onSent) });
}
// card: the picture of the model the app rendered (glbCard.ts), shown on its card.
export function uploadObject(name: string, concept: string, model: File, card: Blob | null) {
  const form = new FormData();
  form.append("name", name);
  form.append("concept", concept);
  form.append("model", model, model.name);
  if (card) form.append("card", card, "card.png");
  return api<{ id: string }>("/api/library/objects", { method: "POST", body: form });
}
export const renameItem = (kind: LibraryKind, id: string, name: string) =>
  api<{ ok: boolean }>(`/api/library/${kind}/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" },
                                                       body: JSON.stringify({ name }) });
// kits: how many kits the item was taken out of.
export const deleteItem = (kind: LibraryKind, id: string) =>
  api<{ kits: number }>(`/api/library/${kind}/${id}`, { method: "DELETE" });
// Models: a name of the user's (the job's folder keeps the first one) and deleting one.
export const renameJob = (job: string, title: string) => api<{ ok: boolean }>(`/api/jobs/${job}`, {
  method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }) });
export const deleteJob = (job: string) => api<{ ok: boolean }>(`/api/jobs/${job}`, { method: "DELETE" });
export type KitRefs = { name: string; datasets: string[]; objects: string[]; scenes: string[] };
export const saveKit = (id: string | null, kit: KitRefs) =>
  api<{ id: string }>(id ? `/api/library/kits/${id}` : "/api/library/kits",
                      { method: id ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(kit) });

// Typed client of the EyeForge API (sim2chip/server.py). Every call goes through `api`, which turns an error
// response into an Error carrying the server's message.

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

export type StageState = "done" | "running" | "pending" | "failed";
export type Stage = { name: string; runtime: string; state: StageState; powered_by: string; seconds: number | null };

export type RealTest = { metric: string; mean: number; std: number; count_mae?: number; exact?: number;
                         test_size?: number; objects?: number; per_seed?: Record<string, number> };
export type Report = {
  job: string; hardware: string; labels: string[]; background?: string | null; input: number[]; model_kb: number; task: string;
  val_score?: number; val_accuracy?: number; real_test?: RealTest; real_accuracy?: RealTest;
  c_array_path: string | null; portable_model_path?: string;
  quantization?: string;  // int8, or float16 for a count detector
  model_path?: string;  // the device model: model_int8.tflite, or model.tflite for a count detector
  // The shipped model's own score on the real test photos (the headline); real_test is the mean of the seeds.
  shipped_test?: { score: number; count_mae?: number } | null;
  advice?: string[];  // what most likely held the score back, in plain words
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
  spec: Spec | null;
  running: boolean;
  calibration: { state: "labeling" | "ready" | "failed"; error?: string } | null;  // teacher check before the first run
  // A job that only labels a library dataset (Datasets page): checking the teacher, labeling every photo, done.
  labeling?: { dataset: string; state: "checking" | "labeling" | "done" | "failed"; error?: string } | null;
  stages: Stage[];
  log: string[];
  cloud_usd: number;
  phase: string | null;
  timing: { started_at: number | null; elapsed_s: number | null; eta_s: number | null } | null;
  report: Report | null;
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
  assets: { uid: string; class: string; role: string; name?: string; generated: boolean; kept: boolean;
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
  if (!r.ok) {
    if (r.status === 401 || r.status === 402 || r.status === 403) accountListeners.forEach((f) => f());
    // Over HTTP/2 statusText is empty: an error without a detail (a server crash) still says something.
    throw new ApiError(body.detail || r.statusText || `the server failed (HTTP ${r.status}); please try again`, r.status);
  }
  return body as T;
}

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
export const activity = (sid: string, act: string) =>
  api<{ steps: Step[]; now: number }>(`/api/interview/${sid}/activity?act=${encodeURIComponent(act)}`);
const actQuery = (act?: string) => (act ? `?act=${encodeURIComponent(act)}` : "");
export const thumbUrl = (sid: string, file: string) => `/api/interview/${sid}/thumb?file=${encodeURIComponent(file)}`;

// The public server sits behind a proxy that refuses requests over 100 MB: photos go in batches under BATCH_BYTES.
// Every batch but the last only stores its photos (more=true); the last one carries the total and does the work.
const BATCH_BYTES = 40_000_000;

function batches(files: File[]): File[][] {
  const out: File[][] = [[]];
  let size = 0;
  for (const f of files) {
    if (size + f.size > BATCH_BYTES && out[out.length - 1].length) { out.push([]); size = 0; }
    out[out.length - 1].push(f);
    size += f.size;
  }
  return out;
}

async function sendInBatches<T>(url: string, files: File[], fields: Record<string, string>): Promise<T> {
  const parts = batches(files);
  let last: T | undefined;
  for (const [i, part] of parts.entries()) {
    const form = new FormData();
    Object.entries(fields).forEach(([k, v]) => form.append(k, v));
    const final = i === parts.length - 1;
    form.append("more", String(!final));
    if (final) form.append("total", String(files.length));
    part.forEach((f) => form.append("photos", f, f.name));
    last = await api<T>(url, { method: "POST", body: form });
  }
  return last as T;
}

export function uploadPhotos(sid: string, files: File[], purpose: string, cls: string | null, act?: string) {
  return sendInBatches<Turn>(`/api/interview/${sid}/photos${actQuery(act)}`, files, { purpose, cls: cls || "" });
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
export function addPhotos(job: string, files: File[], note: string, labels: Record<string, unknown> = {}) {
  return sendInBatches<{ rerun_from: string; photos: number }>(`/api/jobs/${job}/photos`, files, { note, labels: JSON.stringify(labels) });
}

export const listJobs = () => api<{ jobs: string[] }>("/api/jobs");
export const jobStatus = (job: string) => api<Status>(`/api/jobs/${job}`);
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
export const mediaUrl = (job: string, path: string, big = false) =>
  `/api/jobs/${job}/media/${path}${big ? "?size=1024" : ""}`;
export const fileUrl = (job: string, file: string) => `/api/jobs/${job}/files/${file}`;
export const versionFileUrl = (job: string, n: number, file: string) => `/api/jobs/${job}/versions/${n}/files/${file}`;

// The library (sim2chip/library.py): models are the visitor's jobs; datasets, objects and scenes are kept for good
// and kits are named lists of references to them.
export type ModelCard = {
  job: string; created: number; running: boolean; task: string | null; hardware: string | null; classes: string[];
  model: { metric: string | null; score: number | null; kb: number | null; hardware: string | null } | null;
};
// What was seen in a dataset before any goal: objects with the share of photos they show up in, backgrounds.
export type DatasetProfile = { summary: string; photos: number; created: number;
                               objects: { name: string; share: number; count: number }[]; backgrounds: string[] };
export type DatasetCard = { id: string; name: string; created: number; labeled: boolean; photos: number;
                            splits: Record<string, number>; labels: string[]; sample: string[]; jobs: string[];
                            profile: DatasetProfile | null; profiling: boolean };
// What is known about one photo: the teacher's boxes, the user's points, a label, a scene description.
export type PhotoNote = { kind: "scene" | "boxes" | "points" | "class"; ref: string; source: "model" | "user";
                          model: string | null; data: any };
export type DatasetNotes = { keys: string[]; notes: Record<string, PhotoNote[]>; profile: DatasetProfile | null;
                             profiling: boolean };
export const getDatasetNotes = (id: string) => api<DatasetNotes>(`/api/library/datasets/${id}/notes`);
export const labelDataset = (id: string, task: "count" | "classify", classes: string[]) =>
  api<{ job: string }>(`/api/library/datasets/${id}/label`, { method: "POST", headers: { "Content-Type": "application/json" },
                                                            body: JSON.stringify({ task, classes }) });
export const reprofileDataset = (id: string) => api<{ profiling: boolean }>(`/api/library/datasets/${id}/profile`, { method: "POST" });
export type ObjectCard = { id: string; name: string; concept: string; source: "online" | "photos" | "scene" | "upload";
                           has_file: boolean; thumb: boolean; kept: boolean; license?: string; author?: string;
                           url?: string; created: number; jobs: string[] };
export type SceneCard = { id: string; name: string; thumb: boolean; created: number };
export type Kit = { id: string; name: string; datasets: string[]; objects: string[]; scenes: string[];
                    created: number; updated: number };
export type LibraryData = { models: ModelCard[]; datasets: DatasetCard[]; objects: ObjectCard[]; scenes: SceneCard[];
                            kits: Kit[] };
export const getLibrary = () => api<LibraryData>("/api/library");
export const libraryPhotoUrl = (key: string) => `/api/library/photos/${key}`;
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
export async function createDataset(name: string, files: File[]) {
  const [first, ...rest] = batches(files);
  const form = new FormData();
  form.append("name", name);
  first.forEach((f) => form.append("photos", f, f.name));
  const made = await api<{ id: string; photos: number }>("/api/library/datasets", { method: "POST", body: form });
  for (const part of rest) await addDatasetPhotos(made.id, part);
  return made;
}
export function addDatasetPhotos(id: string, files: File[]) {
  const form = new FormData();
  files.forEach((f) => form.append("photos", f, f.name));
  return api<{ photos: number }>(`/api/library/datasets/${id}/photos`, { method: "POST", body: form });
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
export type KitRefs = { name: string; datasets: string[]; objects: string[]; scenes: string[] };
export const saveKit = (id: string | null, kit: KitRefs) =>
  api<{ id: string }>(id ? `/api/library/kits/${id}` : "/api/library/kits",
                      { method: id ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(kit) });

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
  first_question: string;
  public: boolean;
  busy: boolean;
  max_runs: number | null;
  budget: { daily_usd: number; spent_usd: number; can_run: boolean } | null;
  locked: boolean;  // a public server whose API needs the access key, not given yet in this browser
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
                        model?: string };  // its device model file (older versions: model_int8.tflite)

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

export async function api<T = any>(path: string, opts: RequestInit = {}): Promise<T> {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const getConfig = () => api<Config>("/api/config");
// The access key of a public server: right, the server sets a cookie and the API opens for this browser.
export const unlock = (key: string) => api<{ ok: boolean }>("/api/key", json({ key }));
// With a job: an interview that starts from that job's spec, to change something and retrain it.
export const startInterview = (job?: string | null) =>
  api<{ session: string; text: string; job?: string }>(`/api/interview${job ? `?job=${encodeURIComponent(job)}` : ""}`, { method: "POST" });
// act: an id the caller makes up for the turn, to follow its steps with activity() while it runs.
export const reply = (sid: string, text: string, act?: string) =>
  api<Turn>(`/api/interview/${sid}${actQuery(act)}`, json({ text }));
export const pick = (sid: string, files: string[], cls: string, act?: string) =>
  api<Turn>(`/api/interview/${sid}/pick${actQuery(act)}`, json({ files, class: cls }));
export const activity = (sid: string, act: string) =>
  api<{ steps: Step[]; now: number }>(`/api/interview/${sid}/activity?act=${encodeURIComponent(act)}`);
const actQuery = (act?: string) => (act ? `?act=${encodeURIComponent(act)}` : "");
export const thumbUrl = (sid: string, file: string) => `/api/interview/${sid}/thumb?file=${encodeURIComponent(file)}`;

export function uploadPhotos(sid: string, files: File[], purpose: string, cls: string | null, act?: string) {
  const form = new FormData();
  form.append("purpose", purpose);
  form.append("cls", cls || "");
  files.forEach((f) => form.append("photos", f, f.name));
  return api<Turn>(`/api/interview/${sid}/photos${actQuery(act)}`, { method: "POST", body: form });
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

export function addPhotos(job: string, files: File[], note: string) {
  const form = new FormData();
  form.append("note", note);
  files.forEach((f) => form.append("photos", f, f.name));
  return api<{ rerun_from: string; photos: number }>(`/api/jobs/${job}/photos`, { method: "POST", body: form });
}

export const listJobs = () => api<{ jobs: string[] }>("/api/jobs");
export const jobStatus = (job: string) => api<Status>(`/api/jobs/${job}`);
export const getReview = (job: string) => api<Review>(`/api/jobs/${job}/review`);
export const cancelJob = (job: string) => api(`/api/jobs/${job}/cancel`, { method: "POST" });
export const applyFixes = (job: string, fixes: Fixes) =>
  api<{ rerun_from: string | null }>(`/api/jobs/${job}/corrections`, json(fixes));
export const trainCalibrated = (job: string, fixes: Fixes) =>
  api<{ rerun_from: null }>(`/api/jobs/${job}/train`, json(fixes));
export const mediaUrl = (job: string, path: string, big = false) =>
  `/api/jobs/${job}/media/${path}${big ? "?size=1024" : ""}`;
export const fileUrl = (job: string, file: string) => `/api/jobs/${job}/files/${file}`;
export const versionFileUrl = (job: string, n: number, file: string) => `/api/jobs/${job}/versions/${n}/files/${file}`;

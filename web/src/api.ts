// Typed client of the EyeForge API (sim2chip/server.py). Every call goes through `api`, which turns an error
// response into an Error carrying the server's message.

export type Spec = Record<string, any> & {
  name: string;
  task?: string;
  hardware?: string;
  classes: { id: string; description?: string }[];
};

export type Candidate = { file: string; score: number; reason: string };
export type Turn = {
  kind: "say" | "confirm" | "done" | "failed" | "photos" | "pick";
  text: string;
  intro?: string;  // confirm: the assistant's words without the spec summary, which the app shows as a card
  spec?: Spec;
  request?: { for?: "training" | "3d"; class?: string | null; candidates?: Candidate[] };
  photos?: number;
};

export type Config = {
  backend: "local" | "nebius";
  first_question: string;
  public: boolean;
  busy: boolean;
  max_runs: number | null;
  budget: { daily_usd: number; spent_usd: number; can_run: boolean } | null;
};

export type StageState = "done" | "running" | "pending" | "failed";
export type Stage = { name: string; runtime: string; state: StageState; powered_by: string; seconds: number | null };

export type RealTest = { metric: string; mean: number; std: number; count_mae?: number; exact?: number;
                         test_size?: number; objects?: number };
export type Report = {
  job: string; hardware: string; labels: string[]; input: number[]; model_kb: number; task: string;
  val_score?: number; val_accuracy?: number; real_test?: RealTest; real_accuracy?: RealTest;
  c_array_path: string | null; portable_model_path?: string;
  // The model this one replaced, scored on the same test photos (or why it could not be).
  previous?: { version: number; score?: number; now?: number; count_mae?: number; now_count_mae?: number;
               skipped?: string } | null;
};

export type Version = { n: number; reason: string; created?: number; started?: number; metric?: string;
                        headline?: number | null; shipped?: number | null; test_size?: number | null };

export type Status = {
  job: string;
  spec: Spec | null;
  running: boolean;
  stages: Stage[];
  log: string[];
  cloud_usd: number;
  phase: string | null;
  timing: { started_at: number | null; elapsed_s: number | null; eta_s: number | null } | null;
  report: Report | null;
  version: Version | null;  // the model export holds or the run in progress will make
  versions: Version[];      // earlier models, oldest first
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
  test: { key: string; file: string; image: string; truth: any; pred: any }[];
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
// With a job: an interview that starts from that job's spec, to change something and retrain it.
export const startInterview = (job?: string | null) =>
  api<{ session: string; text: string; job?: string }>(`/api/interview${job ? `?job=${encodeURIComponent(job)}` : ""}`, { method: "POST" });
export const reply = (sid: string, text: string) => api<Turn>(`/api/interview/${sid}`, json({ text }));
export const pick = (sid: string, files: string[], cls: string) =>
  api<Turn>(`/api/interview/${sid}/pick`, json({ files, class: cls }));
export const thumbUrl = (sid: string, file: string) => `/api/interview/${sid}/thumb?file=${encodeURIComponent(file)}`;

export function uploadPhotos(sid: string, files: File[], purpose: string, cls: string | null) {
  const form = new FormData();
  form.append("purpose", purpose);
  form.append("cls", cls || "");
  files.forEach((f) => form.append("photos", f, f.name));
  return api<Turn>(`/api/interview/${sid}/photos`, { method: "POST", body: form });
}

export function createJob(spec: Spec) {
  const form = new FormData();
  form.append("spec", JSON.stringify(spec));
  return api<{ job: string; photos: number; already_running?: boolean }>("/api/jobs", { method: "POST", body: form });
}

export type RevisePlan = { stages: string[]; changed: string[]; rerun_from: string | null };
export function reviseJob(job: string, spec: Spec, dry = false) {
  const form = new FormData();
  form.append("spec", JSON.stringify(spec));
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
export const mediaUrl = (job: string, path: string, big = false) =>
  `/api/jobs/${job}/media/${path}${big ? "?size=1024" : ""}`;
export const fileUrl = (job: string, file: string) => `/api/jobs/${job}/files/${file}`;
export const versionFileUrl = (job: string, n: number, file: string) => `/api/jobs/${job}/versions/${n}/files/${file}`;

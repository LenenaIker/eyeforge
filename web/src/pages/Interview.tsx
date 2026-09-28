import { ArrowUp, Check, ImagePlus, Pencil, Sparkles, Upload } from "lucide-react";
import { useEffect, useRef, useState, type DragEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import * as API from "../api";
import type { RevisePlan, Spec, Turn } from "../api";
import { Button, Card, Notice, Spinner, cx } from "../components/ui";
import { prepareImages } from "../images";
import { SpecSummary } from "./SpecSummary";
import { useApp } from "./AppShell";

type Msg = { id: number; from: "bot" | "me" | "sys"; text: string; photos?: string[]; turn?: Turn };
let nextId = 0;

export default function Interview() {
  const { config, refreshJobs } = useApp();
  const navigate = useNavigate();
  const revising = useSearchParams()[0].get("revise");  // a trained job whose setup the user wants to change
  const [plan, setPlan] = useState<RevisePlan | null>(null);  // what a revision would run again
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState<string | null>(null);  // what we are waiting for, shown as a typing bubble
  const [spec, setSpec] = useState<Spec | null>(null);
  const [photoRequest, setPhotoRequest] = useState<Turn["request"] | null>(null);
  const [dragging, setDragging] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const session = useRef<string | null>(null);
  const starting = useRef(false);  // "Looks good" and Train could both launch the same run
  const fileInput = useRef<HTMLInputElement>(null);
  const textInput = useRef<HTMLTextAreaElement>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const push = (m: Omit<Msg, "id">) => setMsgs((all) => [...all, { ...m, id: nextId++ }]);
  useEffect(() => {
    if (!config || msgs.length) return;
    if (!revising) { push({ from: "bot", text: config.first_question }); return; }
    setBusy("Reading your model's setup");
    API.startInterview(revising)
      .then((r) => { session.current = r.session; push({ from: "bot", text: r.text }); })
      .catch((e) => push({ from: "sys", text: `Cannot open that model: ${(e as Error).message}` }))
      .finally(() => setBusy(null));
  }, [config]);
  useEffect(() => {  // before retraining, say which steps run again (and that the rest is reused)
    setPlan(null);
    if (revising && spec) API.reviseJob(revising, spec, true).then(setPlan).catch(() => {});
  }, [revising, spec]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [msgs, busy]);

  const ensureSession = async () => {
    if (!session.current) session.current = (await API.startInterview(revising)).session;
    return session.current;
  };

  async function ask(label: string, call: () => Promise<Turn>) {
    setBusy(label);
    try {
      showTurn(await call());
    } catch (e) {
      push({ from: "sys", text: `Something went wrong: ${(e as Error).message}` });
    }
    setBusy(null);
  }

  function showTurn(t: Turn) {
    setPhotoRequest(t.kind === "photos" ? t.request ?? { for: "training" } : null);
    if (t.kind === "done") {
      session.current = null;
      if (t.spec) { setSpec(t.spec); train(t.spec); }
      return;
    }
    if (t.kind === "failed") { session.current = null; push({ from: "sys", text: t.text }); return; }
    if (t.kind === "confirm" && t.spec) setSpec(t.spec);
    const text = t.kind === "confirm" && t.intro !== undefined ? (t.intro || "Here is the setup I propose.") : t.text;
    push({ from: "bot", text, turn: t });
  }

  async function send(value = text.trim()) {
    if (!value || busy) return;
    push({ from: "me", text: value });
    setText("");
    await ask("Thinking", async () => API.reply(await ensureSession(), value));
  }

  async function addPhotos(list: FileList | File[]) {
    const raw = [...list].filter((f) => f.type.startsWith("image/"));
    if (!raw.length || busy) return;
    const req = photoRequest ?? { for: "training" as const, class: null };
    const previews = raw.slice(0, 8).map((f) => URL.createObjectURL(f));
    push({ from: "me", text: `${raw.length} photo${raw.length > 1 ? "s" : ""}` +
                            (req.for === "3d" ? ` of ${req.class || "the object"} for 3D models` : ""), photos: previews });
    await ask("Uploading and looking at your photos", async () => {
      const files = await prepareImages(raw);
      return API.uploadPhotos(await ensureSession(), files, req.for ?? "training", req.class ?? null);
    });
  }

  async function confirm() {
    if (!session.current) { if (spec) train(spec); return; }
    await ask("Starting", async () => API.reply(session.current!, "yes"));
  }

  async function train(s: Spec) {
    if (starting.current) return;
    starting.current = true;
    setTrainError(null);
    try {
      if (revising) {
        await API.reviseJob(revising, s);
        navigate(`/app/job/${revising}`);
        return;
      }
      const r = await API.createJob(s, true);  // with raw photos, the user first checks the teacher
      refreshJobs();
      navigate(`/app/job/${r.job}`);
    } catch (e) {
      setTrainError((e as Error).message);
    }
    starting.current = false;
  }

  const blocked = config?.busy ? "Every GPU slot is busy right now. Try again in a few minutes." :
    config?.budget && !config.budget.can_run ? "Today's cloud budget is used up. Try again tomorrow." : null;
  const lastConfirm = [...msgs].reverse().find((m) => m.turn?.kind === "confirm")?.id;

  const onDrop = (e: DragEvent) => { e.preventDefault(); setDragging(false); addPhotos(e.dataTransfer.files); };

  return (
    <div className={cx("relative flex flex-1 flex-col", dragging && "after:absolute after:inset-0 after:rounded-2xl after:border-2 after:border-dashed after:border-brand-500 after:bg-brand-50/60")}
         onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
      <div className="flex-1 space-y-4 py-6">
        {revising && <Notice>Changing the setup of <b>{revising.replace(/_/g, " ")}</b>. Only the steps your change affects run again.</Notice>}
        {msgs.map((m) => (
          <Message key={m.id} m={m} session={session.current}
                   onPick={(files, cls) => {
                     push({ from: "me", text: `${files.length} photo${files.length === 1 ? "" : "s"} chosen for 3D models` });
                     ask("Thinking", async () => API.pick(session.current!, files, cls));
                   }}
                   onAddPhotos={() => fileInput.current?.click()}
                   confirm={m.id === lastConfirm && spec ? {
                     spec, blocked, onYes: confirm, plan: revising ? plan : undefined,
                     onChange: () => { textInput.current?.focus(); setText(""); },
                   } : undefined} />
        ))}
        {busy && (
          <div className="flex items-center gap-3 text-sm text-muted">
            <BotAvatar /> <span className="rounded-2xl rounded-tl-sm bg-white px-4 py-3 shadow-sm ring-1 ring-line">
              <Spinner className="mr-2 text-brand-500" />{busy}
            </span>
          </div>
        )}
        {trainError && <Notice tone="bad">Could not start the training: {trainError}</Notice>}
        <div ref={bottom} />
      </div>

      <div className="sticky bottom-0 -mx-4 border-t border-line bg-canvas/95 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur">
        <div className="flex items-end gap-2 rounded-2xl border border-line bg-white p-2 shadow-sm focus-within:border-brand-500">
          <button onClick={() => fileInput.current?.click()} disabled={!!busy}
                  className="grid h-11 w-11 shrink-0 place-items-center rounded-xl text-muted hover:bg-stone-100 hover:text-ink"
                  aria-label="Add photos">
            <ImagePlus className="h-5 w-5" strokeWidth={1.75} />
          </button>
          <textarea ref={textInput} rows={1} value={text} onChange={(e) => setText(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                    placeholder={revising ? "e.g. run it on an ESP32-CAM, or the camera is 5 m high" : "e.g. count the flower pots on balconies, camera on a street lamp"}
                    className="max-h-40 min-h-11 flex-1 resize-none bg-transparent px-1 py-2.5 text-[16px] outline-none placeholder:text-stone-400" />
          <Button onClick={() => send()} disabled={!text.trim() || !!busy} className="h-11 w-11 shrink-0 px-0" aria-label="Send">
            <ArrowUp className="h-5 w-5" />
          </Button>
        </div>
        <input ref={fileInput} type="file" accept="image/*" multiple hidden
               onChange={(e) => { if (e.target.files) addPhotos(e.target.files); e.target.value = ""; }} />
        <AdvancedSpec spec={spec} onTrain={train} />
      </div>
    </div>
  );
}

function BotAvatar() {
  return <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-brand-50 text-brand-600">
    <Sparkles className="h-4 w-4" strokeWidth={1.75} />
  </div>;
}

type Confirm = { spec: Spec; blocked: string | null; onYes: () => void; onChange: () => void; plan?: RevisePlan | null };

function Message({ m, session, onPick, onAddPhotos, confirm }: {
  m: Msg; session: string | null; onPick: (files: string[], cls: string) => void; onAddPhotos: () => void;
  confirm?: Confirm;
}) {
  if (m.from === "sys") return <p className="text-center text-sm text-muted">{m.text}</p>;
  if (m.from === "me") {
    return (
      <div className="flex flex-col items-end gap-2">
        {m.photos && (
          <div className="flex max-w-[85%] flex-wrap justify-end gap-1.5">
            {m.photos.map((src) => <img key={src} src={src} alt="" className="h-16 w-16 rounded-lg object-cover" />)}
          </div>
        )}
        <p className="max-w-[85%] rounded-2xl rounded-tr-sm bg-brand-600 px-4 py-2.5 text-white whitespace-pre-wrap">{m.text}</p>
      </div>
    );
  }
  const t = m.turn;
  return (
    <div className="flex items-start gap-3">
      <BotAvatar />
      <div className="min-w-0 max-w-[85%] flex-1 space-y-3">
        <p className="w-fit rounded-2xl rounded-tl-sm bg-white px-4 py-2.5 whitespace-pre-wrap shadow-sm ring-1 ring-line">{m.text}</p>
        {t?.kind === "photos" && (
          <button onClick={onAddPhotos}
                  className="flex w-full flex-col items-center gap-2 rounded-[var(--radius-card)] border-2 border-dashed border-brand-200 bg-brand-50/50 px-4 py-6 text-brand-700 hover:border-brand-500">
            <Upload className="h-6 w-6" strokeWidth={1.75} />
            <span className="font-medium">
              {t.request?.for === "3d" ? `Add clear photos of ${t.request.class || "the object"}` : "Add photos from your camera"}
            </span>
            <span className="text-sm text-muted">or drop them here, or just answer in the box below</span>
          </button>
        )}
        {t?.kind === "pick" && session && <PickPhotos turn={t} session={session} onPick={onPick} />}
        {confirm && <ConfirmCard {...confirm} />}
      </div>
    </div>
  );
}

function PickPhotos({ turn, session, onPick }: { turn: Turn; session: string; onPick: (f: string[], c: string) => void }) {
  const cands = turn.request?.candidates ?? [];
  const [chosen, setChosen] = useState(() => new Set(cands.map((c) => c.file)));
  const [sent, setSent] = useState(false);
  const toggle = (f: string) => setChosen((s) => { const n = new Set(s); n.has(f) ? n.delete(f) : n.add(f); return n; });
  return (
    <Card className="p-3">
      <p className="mb-2 text-sm text-muted">Photos for the 3D models. Tap one to leave it out.</p>
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
        {cands.map((c) => (
          <button key={c.file} title={c.reason} onClick={() => toggle(c.file)} disabled={sent}
                  className={cx("relative overflow-hidden rounded-lg ring-2 transition", chosen.has(c.file) ? "ring-brand-500" : "opacity-40 ring-transparent")}>
            <img src={API.thumbUrl(session, c.file)} alt="" className="aspect-square w-full object-cover" />
            <span className="absolute right-1 bottom-1 rounded bg-black/60 px-1.5 text-xs text-white">{Math.round(c.score * 100)}%</span>
            {chosen.has(c.file) && <Check className="absolute top-1 right-1 h-5 w-5 rounded-full bg-brand-600 p-0.5 text-white" />}
          </button>
        ))}
      </div>
      <Button className="mt-3 w-full" disabled={sent || chosen.size === 0}
              onClick={() => { setSent(true); onPick([...chosen], turn.request?.class ?? ""); }}>
        Use {chosen.size} photo{chosen.size === 1 ? "" : "s"}
      </Button>
    </Card>
  );
}

function ConfirmCard({ spec, blocked, onYes, onChange, plan }: Confirm) {
  const [sent, setSent] = useState(false);
  const revise = plan !== undefined;
  const nothing = plan && plan.stages.length === 0;
  return (
    <Card className="overflow-hidden">
      <SpecSummary spec={spec} />
      <div className="space-y-3 border-t border-line bg-stone-50 p-4">
        {blocked && <Notice>{blocked}</Notice>}
        {plan && (nothing ? <Notice>That is the setup your model already has: nothing to retrain.</Notice> : (
          <p className="text-sm text-muted">
            {plan.changed.length > 0 && <>Changes: <span className="text-ink">{plan.changed.join(", ")}</span>. </>}
            Runs again: <span className="text-ink">{plan.stages.map((s) => STEP[s] ?? s).filter((s, i, a) => a.indexOf(s) === i).join(", ")}</span>.
            {!plan.stages.includes("render") && " The synthetic images are reused."} The current model is kept as a version.
          </p>
        ))}
        <div className="flex flex-col gap-2 sm:flex-row">
          <Button className="flex-1" disabled={sent || !!blocked || !!nothing || (revise && !plan)} onClick={() => { setSent(true); onYes(); }}>
            <Check className="h-4 w-4" /> {revise ? "Looks good, retrain it" : "Looks good, train it"}
          </Button>
          <Button variant="secondary" disabled={sent} onClick={onChange}><Pencil className="h-4 w-4" /> Change something</Button>
        </div>
      </div>
    </Card>
  );
}

const STEP: Record<string, string> = {
  label: "reading your photos", generate_assets: "3D models from photos", ingest: "sorting photos",
  fetch_assets: "finding 3D models", convert_assets: "preparing models", preview_assets: "checking models",
  qa_assets: "checking models", render: "synthetic images", points: "training data", crops: "training data",
  train: "training", evaluate: "testing", export: "packing",
};

// For people who know the spec format: see, edit or paste the JSON and train it directly.
function AdvancedSpec({ spec, onTrain }: { spec: Spec | null; onTrain: (s: Spec) => void }) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { if (spec) setValue(JSON.stringify(spec, null, 2)); }, [spec]);
  return (
    <details className="mt-2 text-sm" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="cursor-pointer text-muted hover:text-ink">Advanced: the spec as JSON</summary>
      <textarea value={value} onChange={(e) => setValue(e.target.value)} spellCheck={false} rows={10}
                placeholder='{"name": "...", "classes": [...]}'
                className="mt-2 w-full rounded-[var(--radius-control)] border border-line bg-white p-3 font-mono text-xs outline-none focus:border-brand-500" />
      {error && <p className="mt-1 text-red-600">{error}</p>}
      <Button variant="secondary" size="sm" className="mt-2" disabled={!value.trim()} onClick={() => {
        try { setError(null); onTrain(JSON.parse(value)); } catch (e) { setError(`Not valid JSON: ${(e as Error).message}`); }
      }}>Train with this spec</Button>
    </details>
  );
}

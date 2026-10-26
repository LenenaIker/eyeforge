import { ArrowUp, Check, ChevronRight, ImagePlus, LoaderCircle, Pencil, Sparkles, Upload, X } from "lucide-react";
import { useEffect, useRef, useState, type DragEvent, type ReactNode } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import * as API from "../api";
import type { RevisePlan, Spec, Step, Turn } from "../api";
import { Button, Card, Notice, Spinner, cx } from "../components/ui";
import { DEVICES, SpecSummary, hasPhotos, usesSynthetic } from "./SpecSummary";
import { useApp } from "./AppShell";

type Msg = { id: number; from: "bot" | "me" | "sys"; text: string; photos?: string[]; turn?: Turn };
let nextId = 0;

// The chat as it was, kept while the user looks at other pages, in this tab (reloads too) and on the server (other
// tabs and devices): one per conversation (a new model, a kit, a model being changed). "New model" starts a fresh
// one. pending: a turn still running when the user left, whose answer the server keeps (API.activity).
type Saved = { session: string | null; msgs: Msg[]; text: string; spec: Spec | null; proposal: Spec | null;
               photoRequest: Turn["request"] | null; realPhotos: number; pending: { act: string; label: string } | null };
const chats = new Map<string, Saved>();

function loadChat(key: string): Saved | null {
  if (chats.has(key)) return chats.get(key)!;
  try {
    const kept = sessionStorage.getItem(key);
    return kept ? JSON.parse(kept) : null;
  } catch { return null; }
}

function saveChat(key: string, chat: Saved | null) {
  if (chat) chats.set(key, chat); else chats.delete(key);
  try {
    // Photo previews are page-only addresses: a reload drops them, the messages stay.
    if (chat) sessionStorage.setItem(key, JSON.stringify({ ...chat, msgs: chat.msgs.map(({ photos: _, ...m }) => m) }));
    else sessionStorage.removeItem(key);
  } catch { /* private mode or full: kept for this page only */ }
}

export default function Interview() {
  const { config, refreshJobs } = useApp();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const revising = params.get("revise");  // a trained job whose setup the user wants to change
  const kit = revising ? null : params.get("kit");  // a kit of the library the new model starts from
  const key = `eyeforge.chat.${revising ? `revise.${revising}` : kit ? `kit.${kit}` : "new"}`;
  const fresh = (useLocation().state as { fresh?: number } | null)?.fresh;  // "New model": forget the last chat
  const [saved] = useState(() => {
    if (fresh) saveChat(key, null);
    const chat = loadChat(key);
    if (chat) nextId = Math.max(nextId, ...chat.msgs.map((m) => m.id + 1));
    return chat;
  });
  const toServer = useRef<number | undefined>(undefined);  // the chat goes to the server a moment after it changes
  const [kitName, setKitName] = useState<string | null>(null);
  useEffect(() => {
    if (kit) API.getLibrary().then((l) => setKitName(l.kits.find((k) => k.id === kit)?.name ?? null)).catch(() => {});
  }, [kit]);
  const [plan, setPlan] = useState<RevisePlan | null>(null);  // what a revision would run again
  const [msgs, setMsgs] = useState<Msg[]>(saved?.msgs ?? []);
  const [text, setText] = useState(saved?.text ?? params.get("text") ?? "");  // ?text=: a first message to send
  const [busy, setBusy] = useState<string | null>(null);  // what we are waiting for, shown as a typing bubble
  const [live, setLive] = useState<Live | null>(null);    // the steps of the turn in progress, polled
  const [spec, setSpec] = useState<Spec | null>(saved?.spec ?? null);          // the setup to train, with the user's tweaks
  const [proposal, setProposal] = useState<Spec | null>(saved?.proposal ?? null);  // as the interview proposed it
  const [photoRequest, setPhotoRequest] = useState<Turn["request"] | null>(saved?.photoRequest ?? null);
  const [realPhotos, setRealPhotos] = useState(saved?.realPhotos ?? 0);  // training photos uploaded in this interview
  const [pending, setPending] = useState<Saved["pending"]>(saved?.pending ?? null);  // the turn running
  const [dragging, setDragging] = useState(false);
  const [attached, setAttached] = useState<{ file: File; url: string }[]>([]);  // photos waiting in the composer
  const [trainError, setTrainError] = useState<string | null>(null);
  const [unsent, setUnsent] = useState<string | null>(null);  // a message that never reached the server, to send again
  const session = useRef<string | null>(saved?.session ?? null);
  const mounted = useRef(true);
  const starting = useRef(false);  // "Looks good" and Train could both launch the same run
  const fileInput = useRef<HTMLInputElement>(null);
  const textInput = useRef<HTMLTextAreaElement>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const push = (m: Omit<Msg, "id">) => setMsgs((all) => [...all, { ...m, id: nextId++ }]);
  useEffect(() => {
    const chat: Saved = { session: session.current, msgs, text, spec, proposal, photoRequest, realPhotos, pending };
    saveChat(key, chat);
    const sid = session.current;
    if (!sid) return;
    clearTimeout(toServer.current);
    toServer.current = window.setTimeout(() => {
      API.saveChatView(sid, key, { ...chat, msgs: chat.msgs.map(({ photos: _, ...m }) => m) }).catch(() => {});
    }, 800);
  }, [msgs, text, spec, proposal, photoRequest, realPhotos, pending]);
  useEffect(() => {
    mounted.current = true;
    if (fresh) API.closeChats(key).catch(() => {});
    if (saved?.pending && session.current) ask(saved.pending.label, null, saved.pending.act);  // left mid-turn
    else if (!saved && !fresh) {  // nothing in this tab: the chat may be open in another tab or device
      API.openChat(key).then((r) => {
        if (!r.session || !r.view || session.current || !mounted.current) return;
        const v = r.view as Saved;
        session.current = r.session;
        nextId = Math.max(nextId, ...(v.msgs ?? []).map((m) => m.id + 1));
        setMsgs(v.msgs ?? []);
        setText(v.text ?? "");
        setSpec(v.spec ?? null);
        setProposal(v.proposal ?? null);
        setPhotoRequest(v.photoRequest ?? null);
        setRealPhotos(v.realPhotos ?? 0);
        if (v.pending) ask(v.pending.label, null, v.pending.act);
      }).catch(() => {});
    }
    return () => { mounted.current = false; };
  }, []);
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
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [msgs, busy, live?.steps.length]);

  const ensureSession = async () => {
    if (!session.current) session.current = (await API.startInterview(revising, kit)).session;
    return session.current;
  };

  // One turn of the interview. While it runs, its steps (thinking, searching, checking the setup) are polled
  // under an id made up here, so the user sees it working instead of a bare "Thinking". call null: a turn that
  // was running when the user left the page (resumed: its id), whose answer the server kept. onLost: the turn
  // never reached the server (a network error, tried twice), so there is nothing to wait for.
  async function ask(label: string, call: ((act: string) => Promise<Turn>) | null, resumed?: string, onLost?: () => void) {
    const act = resumed ?? Math.random().toString(36).slice(2) + Date.now().toString(36);
    setBusy(label);
    setLive(null);
    setPending({ act, label });
    let polling = true;
    // As the server kept it; arrived: whether the turn reached the server at all.
    const kept: { answer: { turn?: Turn; error?: string; lost?: boolean } | null; arrived?: boolean } = { answer: null };
    const poll = async () => {
      while (polling) {
        if (session.current) {
          try {
            const r = await API.activity(session.current, act);
            kept.arrived = r.arrived;
            if (r.turn || r.error) kept.answer = r;
            else if (polling && mounted.current) setLive({ steps: r.steps, offset: Date.now() / 1000 - r.now });
          } catch { /* a missed poll only delays the next update */ }
        }
        await new Promise((ok) => setTimeout(ok, 700));
      }
    };
    // The answer of a turn whose request was cut; after 15 s the server still has not seen it: it never got
    // there (2026-10-13: a phone's turn lost on the network was waited for 10 minutes).
    const wait = async () => {
      const t0 = Date.now(), until = t0 + 10 * 60_000;
      while (!kept.answer && Date.now() < until) {
        if (kept.arrived === false && Date.now() - t0 > 15_000) return { lost: true };
        await new Promise((ok) => setTimeout(ok, 1000));
      }
      return kept.answer;
    };
    poll();
    let answer: { turn?: Turn; error?: string; lost?: boolean } | null;
    try {
      // Sent again once on a network error: the server answers a turn id it has seen with its first answer.
      answer = call ? { turn: await call(act).catch((e) => API.networkError(e) ? call(act) : Promise.reject(e)) } : await wait();
    } catch (e) {
      // The proxy in front of the server gives up on a long turn (Cloudflare: 100 s, HTTP 524) or the network
      // drops, while the turn may go on there: wait for its answer instead of losing it.
      const status = e instanceof API.ApiError ? e.status : 0;
      answer = [0, 504, 524].includes(status) ? await wait() : null;
      if (!answer) answer = { error: (e as Error).message };
    }
    polling = false;
    if (!mounted.current) return;  // on another page: the chat picks the answer up when the user comes back
    setLive(null);
    setBusy(null);
    setPending(null);
    if (answer?.turn) showTurn(answer.turn);
    else if (answer?.lost && onLost) onLost();
    else push({ from: "sys", text: `Something went wrong: ${answer?.error ?? "no answer from the server; please try again"}` });
  }

  function showTurn(t: Turn) {
    setPhotoRequest(t.kind === "photos" ? t.request ?? { for: "training" } : null);
    if (t.kind === "done") {
      const sid = session.current;
      session.current = null;
      if (t.spec) { setSpec(t.spec); train(t.spec, sid); }
      return;
    }
    if (t.kind === "failed") { session.current = null; push({ from: "sys", text: t.text }); return; }
    if (t.kind === "confirm" && t.spec) { setSpec(t.spec); setProposal(t.spec); }
    const text = t.kind === "confirm" && t.intro !== undefined ? (t.intro || "Here is the setup I propose.") : t.text;
    push({ from: "bot", text, turn: t });
  }

  // The user's message: the text and the photos attached in the composer, sent together (like a messenger app).
  // The photos are stored first (the GPU warms up), then the text goes as the turn, so the assistant answers once.
  // A failed upload gives the message and its photos back to the composer (sent again, only the photos that did
  // not get there go up); a turn that never reached the server gives the text back (its photos are there).
  async function send() {
    const value = text.trim(), photos = attached;
    if ((!value && !photos.length && unsent === null) || busy) return;
    const again = unsent !== null && value === unsent && !photos.length;  // the lost message, already shown
    setUnsent(null);
    const req = photoRequest ?? { for: "training" as const, class: null };
    const purpose = req.for ?? "training";
    const caption = `${photos.length} photo${photos.length > 1 ? "s" : ""}` +
                    (purpose === "3d" ? ` of ${req.class || "the object"} for 3D models` : "");
    const id = nextId;
    if (!again) push({ from: "me", text: value || caption, ...(photos.length ? { photos: photos.slice(0, 8).map((a) => a.url) } : {}) });
    setText("");
    if (photos.length) {
      setAttached([]);
      setBusy(API.progressText(null, "your photos"));
      try {
        const sid = await ensureSession();
        const r = await API.storePhotos(sid, photos.map((a) => a.file), purpose, req.class ?? null,
                                        (p) => setBusy(API.progressText(p, "your photos")));
        if (purpose === "training") setRealPhotos((n) => n + photos.length - r.unreadable.length);
        if (r.unreadable.length) push({ from: "sys", text: API.unreadableText(r.unreadable) });
      } catch (e) {
        setBusy(null);
        setMsgs((all) => all.filter((m) => m.id !== id));
        setText(value);
        setAttached(photos);
        push({ from: "sys", text: `Your photos did not upload: ${(e as Error).message}` });
        return;
      }
    }
    await ask("Thinking", async (act) => API.reply(await ensureSession(), value, act), undefined, () => {
      setText(value);
      setUnsent(value);
      push({ from: "sys", text: "Your message did not reach the server (the connection failed). It is back in the box: send it again." });
    });
  }

  function addPhotos(list: FileList | File[]) {
    const raw = [...list].filter((f) => f.type.startsWith("image/"));
    if (!raw.length) return;
    setAttached((all) => [...all, ...raw.map((file) => ({ file, url: URL.createObjectURL(file) }))]);
    textInput.current?.focus();
  }

  function detach(url: string) {
    URL.revokeObjectURL(url);
    setAttached((all) => all.filter((a) => a.url !== url));
  }

  // Trains the setup as shown, with the user's tweaks in the card (the interview only knows its own proposal).
  function confirm() {
    if (!spec) return;
    const sid = session.current;
    session.current = null;
    train(spec, sid);
  }

  async function train(s: Spec, sid?: string | null) {
    if (starting.current) return;
    starting.current = true;
    setTrainError(null);
    try {
      if (revising) {
        await API.reviseJob(revising, s, false, sid);
        saveChat(key, null);  // over: "Chat" opens a new one
        navigate(`/app/job/${revising}`);
        return;
      }
      const r = await API.createJob(s, true, sid);  // with raw photos, the user first checks the teacher
      saveChat(key, null);
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
        {kitName && <Notice>Starting from your kit <b>{kitName}</b>: its photos and 3D objects are used, nothing is built twice.</Notice>}
        {msgs.map((m) => (
          <Message key={m.id} m={m} session={session.current}
                   onPick={(files, cls) => {
                     push({ from: "me", text: `${files.length} photo${files.length === 1 ? "" : "s"} chosen for 3D models` });
                     ask("Thinking", async (act) => API.pick(session.current!, files, cls, act));
                   }}
                   onAddPhotos={() => fileInput.current?.click()}
                   confirm={m.id === lastConfirm && spec ? {
                     spec, proposal: proposal ?? spec, onEdit: setSpec, blocked, onYes: confirm, realPhotos,
                     plan: revising ? plan : undefined,
                     onChange: () => { textInput.current?.focus(); setText(""); },
                   } : undefined} />
        ))}
        {busy && (
          <div className="flex items-start gap-3 text-sm text-muted">
            <BotAvatar />
            <div className="min-w-0 max-w-[85%] rounded-2xl rounded-tl-sm bg-white px-4 py-3 shadow-sm ring-1 ring-line">
              {live?.steps.length ? <Steps steps={live.steps} offset={live.offset} /> :
                <span><Spinner className="mr-2 text-brand-500" />{busy}</span>}
            </div>
          </div>
        )}
        {trainError && <Notice tone="bad">Could not start the training: {trainError}</Notice>}
        <div ref={bottom} />
      </div>

      <div className="sticky bottom-0 -mx-4 border-t border-line bg-canvas/95 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur">
        <div className="rounded-2xl border border-line bg-white p-2 shadow-sm focus-within:border-brand-500">
          {attached.length > 0 && (
            <div className="flex flex-wrap gap-2 px-1 pt-1 pb-2">
              {attached.map((a) => (
                <div key={a.url} className="relative">
                  <img src={a.url} alt="" className="h-16 w-16 rounded-lg object-cover ring-1 ring-line" />
                  <button onClick={() => detach(a.url)} aria-label="Remove photo"
                          className="absolute -top-1.5 -right-1.5 grid h-5 w-5 place-items-center rounded-full bg-ink text-white shadow hover:bg-gray-700">
                    <X className="h-3 w-3" strokeWidth={2.5} />
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="flex items-end gap-2">
            <button onClick={() => fileInput.current?.click()}
                    className="grid h-11 w-11 shrink-0 place-items-center rounded-xl text-muted hover:bg-gray-100 hover:text-ink"
                    aria-label="Add photos">
              <ImagePlus className="h-5 w-5" strokeWidth={1.75} />
            </button>
            <textarea ref={textInput} rows={1} value={text} onChange={(e) => setText(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                      placeholder={revising ? "What should change?" : "What do you want to forge?"}
                      className="max-h-40 min-h-11 flex-1 resize-none bg-transparent px-1 py-2.5 text-[16px] outline-none placeholder:text-gray-400" />
            <Button onClick={() => send()} disabled={(!text.trim() && !attached.length && unsent === null) || !!busy} className="h-11 w-11 shrink-0 px-0" aria-label="Send">
              <ArrowUp className="h-5 w-5" />
            </Button>
          </div>
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

type Live = { steps: Step[]; offset: number };  // offset: client clock minus server clock, in seconds

const seconds = (s: number) => (s < 60 ? `${Math.max(0, Math.round(s))} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

// The steps of a turn, like an agent's task list: a spinner on the running one with its time so far.
function Steps({ steps, offset = 0 }: { steps: Step[]; offset?: number }) {
  const [, tick] = useState(0);
  const running = steps.some((s) => s.state === "running");
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [running]);
  const now = Date.now() / 1000 - offset;
  return (
    <ol className="space-y-1.5">
      {steps.map((s, i) => (
        <li key={i} className="flex items-start gap-2">
          <span className="mt-0.5 grid h-4 w-4 shrink-0 place-items-center">
            {s.state === "running" ? <LoaderCircle className="h-4 w-4 animate-spin text-brand-500" /> :
             s.state === "done" ? <Check className="h-4 w-4 text-brand-600" /> : <X className="h-4 w-4 text-amber-600" />}
          </span>
          <span className="min-w-0 flex-1">
            <span className={cx(s.state === "running" ? "text-ink" : "text-muted")}>
              {s.text}{s.total ? ` (${s.done ?? 0}/${s.total})` : ""}
            </span>
            {s.detail && <span className="block text-xs text-gray-400">{s.detail}</span>}
          </span>
          <span className="shrink-0 text-xs tabular-nums text-gray-400">{seconds((s.ended ?? now) - s.started)}</span>
        </li>
      ))}
    </ol>
  );
}

// Above an answer: how long the turn took, and its steps on demand.
function WorkedFor({ steps }: { steps: Step[] }) {
  const [open, setOpen] = useState(false);
  const took = Math.max(...steps.map((s) => s.ended ?? s.started)) - steps[0].started;
  return (
    <div className="text-sm text-muted">
      <button onClick={() => setOpen(!open)} className="flex items-center gap-1 hover:text-ink">
        <ChevronRight className={cx("h-4 w-4 transition", open && "rotate-90")} />
        Worked for {seconds(took)}, {steps.length} step{steps.length === 1 ? "" : "s"}
      </button>
      {open && <div className="mt-2 ml-5"><Steps steps={steps} /></div>}
    </div>
  );
}

type Confirm = { spec: Spec; proposal: Spec; onEdit: (s: Spec) => void; blocked: string | null; onYes: () => void;
                 onChange: () => void; plan?: RevisePlan | null; realPhotos: number };

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
        {t?.activity && t.activity.length > 0 && <WorkedFor steps={t.activity} />}
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
            {c.score > 0 && <span className="absolute right-1 bottom-1 rounded bg-black/60 px-1.5 text-xs text-white">{Math.round(c.score * 100)}%</span>}
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

function ConfirmCard({ spec, proposal, onEdit, blocked, onYes, onChange, plan, realPhotos }: Confirm) {
  const [sent, setSent] = useState(false);
  const revise = plan !== undefined;
  const nothing = plan && plan.stages.length === 0;
  return (
    <Card className="overflow-hidden">
      <SpecSummary spec={spec} />
      {!sent && <SpecOptions spec={spec} proposal={proposal} onEdit={onEdit} realPhotos={realPhotos} />}
      <div className="space-y-3 border-t border-line bg-gray-50 p-4">
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

const FRAMES = [500, 1000, 2000, 4000];
// How many synthetic images per real photo we recommend when counting. The detector trains a fixed number of steps,
// so many synthetic images drown the few real ones. On HOPE with 40 real photos (experiments/NOTES.md, 2026-10-06):
// 1-3 per photo F1 0.827-0.833, 7.5 0.817, 15 0.809, 50 0.776 against 0.819 with the photos alone. Published
// studies put the best mix between 0.5 and 4 per real image, depending on the task. Only a recommendation:
// the user picks.
const SYNTH_PER_PHOTO = { good: 4, near: 10 };
type Advice = "good" | "near" | "far";
const ADVICE: Record<Advice, { dot: string; text: string }> = {
  good: { dot: "bg-emerald-500", text: "recommended" },
  near: { dot: "bg-amber-400", text: "may help less" },
  far: { dot: "bg-red-500", text: "likely too many for your photos" },
};
const niceRound = (n: number) => { const step = n < 200 ? 10 : n < 1000 ? 50 : 100; return Math.max(step, Math.round(n / step) * step); };

function frameAdvice(frames: number, photos: number, task: string | undefined): Advice | null {
  if (task !== "count" || photos <= 0) return null;  // measured for counting with real photos only
  const r = frames / photos;
  return r <= SYNTH_PER_PHOTO.good ? "good" : r <= SYNTH_PER_PHOTO.near ? "near" : "far";
}
const MISTAKES: [string, string][] = [["find_more", "Miss fewer objects"], ["balanced", "Balanced"],
                                      ["fewer_false", "Fewer false alarms"]];

// The choices a user may want to make before paying for a run, without going back to the chat.
function SpecOptions({ spec, proposal, onEdit, realPhotos }: {
  spec: Spec; proposal: Spec; onEdit: (s: Spec) => void; realPhotos: number;
}) {
  const synthetic = usesSynthetic(spec);
  const photos = hasPhotos(spec);
  const canSynth = photos;  // synthetic images are rebuilt from the user's photos
  const proposed = proposal.generation?.frames ?? 2000;
  const setSynthetic = (on: boolean) =>
    onEdit({ ...spec, generation: { ...spec.generation, frames: on ? (proposed > 0 ? proposed : 2000) : 0 } });
  const frames = spec.generation?.frames ?? 2000;
  const advised = frameAdvice(1, realPhotos, spec.task) !== null;
  const options = [...new Set([...FRAMES, frames,
                               ...(advised ? [niceRound(realPhotos * 2), niceRound(realPhotos * SYNTH_PER_PHOTO.good)] : [])])]
    .sort((a, b) => a - b);
  const current = frameAdvice(frames, realPhotos, spec.task);
  return (
    <div className="space-y-4 border-t border-line p-4 text-sm">
      <Option label="Training data">
        <Choice options={[["photos", "Your photos only"], ["synthetic", "Photos + synthetic images"]]}
                value={synthetic ? "synthetic" : "photos"} onChange={(v) => setSynthetic(v === "synthetic")}
                disabled={{ photos: !photos, synthetic: !canSynth }} />
        <p className="mt-1.5 text-xs text-muted">
          {!photos ? "Add some of your photos in the chat to train on them alone." :
           synthetic ? "Synthetic images rebuilt from your photos cover cases they miss, but rendering them on a GPU makes the run longer and more expensive." :
           "Faster and cheaper: no rendering. It needs photos of every case the camera will see."}
        </p>
        {synthetic && (
          <div className="mt-2">
            <p className="mb-1.5 text-muted">Synthetic images <span className="text-xs">(fewer is cheaper)</span></p>
            <div className="flex flex-wrap gap-1.5">
              {options.map((n) => {
                const a = frameAdvice(n, realPhotos, spec.task);
                return (
                  <button key={n} onClick={() => onEdit({ ...spec, generation: { ...spec.generation, frames: n } })}
                          disabled={n === frames} title={a ? ADVICE[a].text : undefined}
                          className={cx("flex items-center gap-1.5 rounded-full px-3 py-1.5 ring-1 transition",
                                        n === frames ? "bg-brand-600 text-white ring-brand-600" : "bg-white text-ink ring-line hover:ring-brand-500")}>
                    {a && <span className={cx("h-2 w-2 rounded-full", ADVICE[a].dot)} />}
                    {n.toLocaleString()}
                  </button>
                );
              })}
            </div>
            {current && (
              <p className="mt-1.5 text-xs text-muted">
                {frames.toLocaleString()} for your {realPhotos} photos: {ADVICE[current].text}. Green is what we
                recommend (up to {SYNTH_PER_PHOTO.good} per photo), yellow is close, red is out of proportion; too many
                synthetic images can drown your real photos. Your call.
              </p>
            )}
          </div>
        )}
      </Option>
      <Option label="Device">
        <select value={spec.hardware ?? "generic"} onChange={(e) => onEdit({ ...spec, hardware: e.target.value })}
                className="w-full rounded-[var(--radius-control)] border border-line bg-white px-2 py-1.5">
          {Object.entries(DEVICES).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
      </Option>
      {spec.user_data?.raw && spec.task === "count" && (
        <Option label="Which mistake is worse">
          <Choice options={MISTAKES} value={spec.errors ?? "balanced"}
                  onChange={(v) => onEdit({ ...spec, errors: v === "balanced" ? null : v })} />
        </Option>
      )}
    </div>
  );
}

function Option({ label, children }: { label: string; children: ReactNode }) {
  return <div><p className="mb-1.5 font-medium">{label}</p>{children}</div>;
}

function Choice({ options, value, onChange, disabled = {} }: {
  options: [string, string][]; value: string; onChange: (v: string) => void; disabled?: Record<string, boolean>;
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map(([k, label]) => (
        <button key={k} onClick={() => onChange(k)} disabled={disabled[k] || k === value}
                className={cx("rounded-full px-3 py-1.5 ring-1 transition",
                              k === value ? "bg-brand-600 text-white ring-brand-600" :
                              disabled[k] ? "text-gray-400 ring-line" : "bg-white text-ink ring-line hover:ring-brand-500")}>
          {label}
        </button>
      ))}
    </div>
  );
}

const STEP: Record<string, string> = {
  label: "reading your photos", generate_assets: "3D models from photos", ingest: "sorting photos",
  fetch_assets: "finding 3D models", convert_assets: "preparing 3D models", preview_assets: "photographing 3D models",
  qa_assets: "checking 3D models", scene_render: "objects in your scenes", render: "synthetic images", points: "training data", crops: "training data",
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

import { ArrowLeft, Copy, GraduationCap, Mountain, Pencil, Plus, RefreshCw, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { copyLabeling, createLabeling, deleteLabeling, extractScene, getDataset, getSceneProposals, libraryPhotoUrl,
         renameLabeling, reprofileDataset, type DatasetPage, type DatasetProfile, type Labeling,
         type SceneProposal } from "../api";
import { ConfirmDialog, Dialog, MoreMenu, NameDialog, field, menuItem } from "../components/library";
import { Badge, Button, Card, Notice, Spinner, cx } from "../components/ui";

// One dataset (docs/plan_etiquetado.md): what is in its photos, its labelings (each a card with its classes and
// how many photos you reviewed) and its photos. Classes never show on their own: always inside their labeling.

const PAGE = 60;
type Tab = "labelings" | "photos";

export const classHue = (i: number) => (i * 137) % 360;  // the class's color, by its place in the labeling

export function Dataset() {
  const { id = "" } = useParams();
  const [d, setD] = useState<DatasetPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("labelings");
  const [limit, setLimit] = useState(PAGE);
  const [creating, setCreating] = useState(false);
  const load = useCallback(() => { getDataset(id).then(setD).catch((e) => setError(e.message)); }, [id]);
  useEffect(load, [load]);
  useEffect(() => {  // the profile is being made: look again in a while
    if (!d?.profiling) return;
    const t = setTimeout(load, 5000);
    return () => clearTimeout(t);
  }, [d, load]);

  if (error) return <div className="py-6"><Notice tone="bad">{error}</Notice></div>;
  if (!d) return <div className="flex justify-center py-16"><Spinner /></div>;
  const p = d.profile;
  const tabs: [Tab, string][] = [["labelings", `Labelings (${d.labelings.length})`], ["photos", `Photos (${d.photos})`]];
  return (
    <div className="space-y-6 py-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link to="/app/datasets" className="inline-flex items-center gap-1 text-sm text-muted hover:text-ink"><ArrowLeft className="h-4 w-4" /> Datasets</Link>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight">{d.name}</h1>
          <p className="mt-1 text-sm text-muted">{d.photos} photo{d.photos === 1 ? "" : "s"}</p>
        </div>
        <Button size="sm" className="h-10" onClick={() => setCreating(true)}><Plus className="h-4 w-4" /> New labeling</Button>
      </div>
      {creating && <NewLabeling id={id} profile={p} onClose={() => setCreating(false)} onMade={load} />}

      <Card className="space-y-3 p-5">
        <div className="flex items-center justify-between gap-3">
          <div className="font-semibold">What is in these photos</div>
          <Button size="sm" variant="ghost" disabled={d.profiling}
                  onClick={() => reprofileDataset(id).then(load).catch((e) => setError(e.message))}>
            {d.profiling ? <><Spinner className="h-4 w-4" /> Looking...</> : <><RefreshCw className="h-4 w-4" /> Look again</>}
          </Button>
        </div>
        {!p ? <p className="text-sm text-muted">{d.profiling ? "A vision model is looking at a sample of the photos." : "Not looked at yet."}</p> : <>
          <div className="flex flex-wrap gap-1.5">
            {p.objects.map((o) => <Badge key={o.name} tone="brand">{o.name} · {Math.round(o.share * 100)}% · ~{o.count}</Badge>)}
          </div>
          {p.backgrounds.length > 0 && <p className="text-sm"><span className="text-muted">Backgrounds:</span> {p.backgrounds.join("; ")}</p>}
        </>}
      </Card>

      <SceneProposals id={id} profiling={d.profiling} />

      <div className="flex gap-1 border-b border-line" role="tablist">
        {tabs.map(([k, label]) => (
          <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}
                  className={cx("-mb-px border-b-2 px-3 py-2 text-sm", tab === k ? "border-brand-600 font-semibold text-ink" : "border-transparent text-muted hover:text-ink")}>
            {label}
          </button>
        ))}
      </div>

      {tab === "labelings" && (
        d.labelings.length === 0
          ? <Card className="p-6 text-sm text-muted">No labelings yet. A labeling is a list of things to find (or to sort the photos
              by) with their marks on every photo. Make one with "New labeling", or a model trained on these photos makes it.</Card>
          : <div className="space-y-3">{d.labelings.map((l) => <LabelingCard key={l.id} d={d} l={l} onChanged={load} />)}</div>
      )}

      {tab === "photos" && <>
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 lg:grid-cols-6">
          {d.photo_ids.slice(0, limit).map((pid) => (
            <img key={pid} src={libraryPhotoUrl(pid)} alt="" loading="lazy" className="aspect-square w-full rounded-lg bg-gray-100 object-cover" />
          ))}
        </div>
        {limit < d.photo_ids.length && <div className="flex justify-center"><Button variant="ghost" onClick={() => setLimit(limit + PAGE)}>Show more</Button></div>}
      </>}
    </div>
  );
}

// A labeling as a card: its classes inside it, how many photos you reviewed, and what to do with it.
function LabelingCard({ d, l, onChanged }: { d: DatasetPage; l: Labeling; onChanged: () => void }) {
  const navigate = useNavigate();
  const [dialog, setDialog] = useState<"rename" | "delete" | null>(null);
  const open = `/app/datasets/${d.id}/labelings/${l.id}`;
  const reviewed = l.photos ? l.reviewed / l.photos : 0;
  const marked = l.photos ? Math.max(0, l.marked - l.reviewed) / l.photos : 0;
  const train = () => navigate(`/app?text=${encodeURIComponent(
    `Train a model with the labeling "${l.name}" (id ${l.id}) of my dataset "${d.name}".`)}`, { state: { fresh: Date.now() } });
  return (
    <Card className="space-y-3 p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <Link to={open} className="font-semibold hover:underline">{l.name}</Link>
        <span className="text-sm text-muted">{l.task === "count" ? "Finds and counts things" : "Sorts the photos"}
          {l.models.length > 0 && ` · used by ${l.models.length} model${l.models.length === 1 ? "" : "s"}`}</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {l.classes.map((c, i) => (
          <span key={c.id} className="inline-flex items-center gap-1.5 rounded-md bg-gray-100 px-2 py-0.5 text-sm">
            <i className="h-2.5 w-2.5 rounded-full" style={{ background: `hsl(${classHue(i)} 70% 45%)` }} />{c.key}
          </span>
        ))}
      </div>
      <div className="flex h-2 overflow-hidden rounded-full bg-gray-100" title={`${l.reviewed} reviewed, ${l.marked} with marks`}>
        <i className="block bg-brand-600" style={{ width: `${reviewed * 100}%` }} />
        <i className="block bg-brand-200" style={{ width: `${marked * 100}%` }} />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-muted">{l.reviewed} of {l.photos} reviewed by you · {l.marked} with marks</span>
        <span className="flex-1" />
        <Button size="sm" variant="secondary" onClick={() => navigate(open)}>Open</Button>
        <Button size="sm" variant="secondary" onClick={train}><GraduationCap className="h-4 w-4" /> Train a model</Button>
        <MoreMenu items={(close) => <>
          <button className={menuItem} onClick={() => { close(); setDialog("rename"); }}><Pencil className="h-4 w-4" /> Rename</button>
          <button className={menuItem} onClick={() => { close(); copyLabeling(l.id).then(onChanged); }}><Copy className="h-4 w-4" /> Copy</button>
          <button className={cx(menuItem, "text-red-600")} onClick={() => { close(); setDialog("delete"); }}><Trash2 className="h-4 w-4" /> Delete</button>
        </>} />
      </div>
      {dialog === "rename" && <NameDialog title="Rename labeling" name={l.name} onClose={() => setDialog(null)}
                                          save={(v) => renameLabeling(l.id, v).then(onChanged)} />}
      {dialog === "delete" && (
        <ConfirmDialog title={`Delete "${l.name}"?`} action="Delete" onClose={() => setDialog(null)}
                       run={() => deleteLabeling(l.id).then(onChanged)}>
          Its classes and every mark on its photos are deleted, yours too. The photos stay in the dataset, and models
          already trained keep working.
        </ConfirmDialog>
      )}
    </Card>
  );
}

// A new labeling: find things (a mark on every one) or sort photos (one class each). The things the profile saw
// are offered as a start. The teacher can start on it right away.
function NewLabeling({ id, profile, onClose, onMade }:
  { id: string; profile: DatasetProfile | null; onClose: () => void; onMade: () => void }) {
  const navigate = useNavigate();
  const [task, setTask] = useState<"count" | "classify">("count");
  const [name, setName] = useState("");
  const [names, setNames] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const list = names.split(",").map((n) => n.trim()).filter(Boolean);
  const go = async (teach: boolean) => {
    if (!list.length) { setError("Write the classes first."); return; }
    setBusy(true);
    try {
      const r = await createLabeling(id, name.trim(), task, list, teach);
      if (r.job) navigate(`/app/job/${r.job}`);
      else { onMade(); onClose(); }
    } catch (err: any) {
      setError(err.message);
      setBusy(false);
    }
  };
  const add = (n: string) => setNames(list.includes(n) ? names : [...list, n].join(", "));
  const row = "flex cursor-pointer items-start gap-3 rounded-xl border px-4 py-3 text-sm";
  return (
    <Dialog title="New labeling" onClose={onClose}>
      <form onSubmit={(e) => { e.preventDefault(); go(true); }} className="space-y-4">
        {([["count", "Find things", "A mark on every one, to count or detect them."],
           ["classify", "Sort photos", "Each photo gets one of the classes."]] as const).map(([k, title, text]) => (
          <label key={k} className={cx(row, task === k ? "border-brand-500 bg-brand-50" : "border-line")}>
            <input type="radio" checked={task === k} onChange={() => setTask(k)} className="mt-0.5 accent-brand-600" />
            <span><span className="font-medium">{title}</span><br /><span className="text-muted">{text}</span></span>
          </label>
        ))}
        <input value={names} onChange={(e) => setNames(e.target.value)} className={field} aria-label="Classes"
               placeholder={task === "count" ? "What to find, e.g. blue bin, yellow bin" : "The classes, e.g. full, empty"} />
        {profile && profile.objects.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {profile.objects.map((o) => (
              <button type="button" key={o.name} onClick={() => add(o.name)}
                      className="rounded-full border border-line px-2.5 py-0.5 text-xs hover:border-brand-500">+ {o.name}</button>
            ))}
          </div>
        )}
        <input value={name} onChange={(e) => setName(e.target.value)} className={field} aria-label="Name"
               placeholder="Its name (optional), e.g. Bins by color" />
        <p className="text-xs text-muted">Name each class by what the camera sees (a color, a shape), not by what it means.
          The teacher marks a few photos first so you can check it understood, then it marks them all.</p>
        {error && <Notice tone="bad">{error}</Notice>}
        <div className="flex gap-2">
          <Button type="button" variant="secondary" className="flex-1" disabled={busy} onClick={() => go(false)}>Just create it</Button>
          <Button type="submit" className="flex-1" disabled={busy}>{busy ? "Starting..." : "Create and label"}</Button>
        </div>
      </form>
    </Dialog>
  );
}


// The route's steps (experiments/worldstereo/wsroute/steps.py) in the user's words, by name prefix.
const SCENE_STEPS: [string, string][] = [
  ["things", "Finding big objects to erase"], ["masks", "Finding big objects to erase"], ["split", "Finding big objects to erase"],
  ["erase", "Erasing big objects"], ["publish", "Erasing big objects"], ["caption", "Imagining the place all around"],
  ["pano", "Imagining the place all around"], ["walks", "Planning walks through the place"],
  ["walk_legs", "Walking through the place (WorldStereo)"], ["planned_legs", "Walking where nothing was seen yet"],
  ["depth", "Measuring depth"], ["views", "Building the 3D scene"], ["train", "Building the 3D scene"],
  ["audit", "Looking for gaps"], ["plan", "Looking for gaps"], ["stage", "Looking for gaps"], ["grid", "Checking the scene"],
  ["pseudo", "Cleaning the views (NVIDIA Fixer)"], ["fix", "Cleaning the views (NVIDIA Fixer)"],
  ["usdz", "Packing the scene for Isaac Sim"], ["bundle", "Packing the scene"], ["starting", "Starting"],
];
const stepText = (step?: string) => SCENE_STEPS.find(([k]) => step?.startsWith(k))?.[1] ?? "Working";

// The dataset's photos proposed for a scene (scene_pick.py): extract one, follow its job here, find it in Scenes.
function SceneProposals({ id, profiling }: { id: string; profiling: boolean }) {
  const [list, setList] = useState<SceneProposal[] | null>(null);
  const [computing, setComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => {
    getSceneProposals(id).then((r) => { setList(r.proposals); setComputing(r.computing); }).catch((e) => setError(e.message));
  }, [id]);
  useEffect(load, [load, profiling]);
  const busy = computing || (list ?? []).some((p) => p.scene?.status === "building");
  useEffect(() => {  // proposals being made or a scene being built: look again in a while
    if (!busy) return;
    const t = setTimeout(load, computing ? 5000 : 15000);
    return () => clearTimeout(t);
  }, [busy, computing, load, list]);
  const extract = (pid: number) => extractScene(pid).then(load).catch((e) => setError(e.message));
  return (
    <Card className="space-y-3 p-5">
      <div>
        <div className="font-semibold">Scenes from these photos</div>
        <p className="text-sm text-muted">A scene is the place in a photo rebuilt in 3D, big objects erased, to train with
          synthetic images of any class. It takes about two hours; you can leave the page.</p>
      </div>
      {error && <Notice tone="bad">{error}</Notice>}
      {list === null || (computing && list.length === 0)
        ? <p className="text-sm text-muted">{profiling || computing ? <><Spinner className="h-4 w-4" /> Choosing photos...</> : "Loading..."}</p>
        : list.length === 0
          ? <p className="text-sm text-muted">No photo suits a scene: the place needs to be outdoors, taken by hand at eye
              height looking ahead, by day, with the camera's focal length in the photo (straight from the phone).</p>
          : <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {list.map((p) => {
                const sc = p.scene;
                const pct = sc?.total ? Math.round(100 * (sc.done ?? 0) / sc.total) : 0;
                return (
                  <div key={p.photo_id} className="overflow-hidden rounded-lg border border-line">
                    <img src={libraryPhotoUrl(p.photo_id)} alt="" loading="lazy" className="aspect-[4/3] w-full bg-gray-100 object-cover" />
                    <div className="space-y-2 p-2.5">
                      {!sc || sc.status === "failed" ? <>
                        {sc?.status === "failed" && <p className="text-xs text-red-700">No scene: {sc.why ?? "unknown"}.</p>}
                        <Button size="sm" className="w-full" onClick={() => extract(p.photo_id)}>
                          <Mountain className="h-4 w-4" /> {sc ? "Try again" : "Extract"}
                        </Button>
                      </> : sc.status === "building" ? <>
                        <div className="h-2 overflow-hidden rounded-full bg-gray-100" role="progressbar" aria-valuenow={pct}>
                          <div className="h-full rounded-full bg-brand-600 transition-all" style={{ width: `${Math.max(pct, 3)}%` }} />
                        </div>
                        <p className="truncate text-xs text-muted" title={sc.step}>{stepText(sc.step)} · {pct}%</p>
                      </> : <Link to="/app/assets/scenes" className="block text-center text-sm font-medium text-brand-600 hover:underline">
                        Ready: see it in Scenes</Link>}
                    </div>
                  </div>
                );
              })}
            </div>}
    </Card>
  );
}

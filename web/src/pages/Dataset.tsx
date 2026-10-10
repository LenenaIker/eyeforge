import { ArrowLeft, Pencil, RefreshCw, Tag, Trash2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { forgetLabels, getDatasetNotes, getLibrary, labelDataset, libraryPhotoUrl, renameLabels, reprofileDataset,
         type DatasetNotes, type DatasetProfile, type PhotoNote } from "../api";
import { ConfirmDialog, Dialog, MoreMenu, NameDialog, field, menuItem } from "../components/library";
import { Badge, Button, Card, Notice, Spinner, cx } from "../components/ui";

// One dataset: what was seen in it (its profile) and every photo with what is known about it, the teacher's boxes
// and the user's marks drawn on it. Kept by photo content, so a label made once serves every model.

const PAGE = 60;

// A label's name for the filter: the phrase the teacher looked for, or the labels a class was chosen among.
const refName = (n: PhotoNote) => n.kind === "class" ? JSON.parse(n.ref).join(" / ") : n.ref;

export function Dataset() {
  const { id = "" } = useParams();
  const [d, setD] = useState<DatasetNotes | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [show, setShow] = useState<string>("");
  const [limit, setLimit] = useState(PAGE);
  const [labeling, setLabeling] = useState(false);
  const [editing, setEditing] = useState<"rename" | "delete" | null>(null);  // the labeling shown
  const load = useCallback(() => {
    getDatasetNotes(id).then(setD).catch((e) => setError(e.message));
  }, [id]);
  useEffect(() => {
    load();
    getLibrary().then((l) => setName(l.datasets.find((x) => x.id === id)?.name ?? "")).catch(() => {});
  }, [id, load]);
  useEffect(() => {  // the profile is being made: look again in a while
    if (!d?.profiling) return;
    const t = setTimeout(load, 5000);
    return () => clearTimeout(t);
  }, [d, load]);

  // The labelings this dataset has, by kind and phrase: the filter's choices.
  const kinds = useMemo(() => {
    const out = new Map<string, string>();
    for (const notes of Object.values(d?.notes ?? {}))
      for (const n of notes) if (n.kind !== "scene") out.set(`${n.kind === "points" ? "boxes" : n.kind}:${n.ref}`, refName(n));
    return [...out.entries()];
  }, [d]);
  useEffect(() => { if (!show && kinds.length) setShow(kinds[0][0]); }, [kinds, show]);

  if (error) return <div className="py-6"><Notice tone="bad">{error}</Notice></div>;
  if (!d) return <div className="flex justify-center py-16"><Spinner /></div>;
  const [kind, ref] = show ? [show.slice(0, show.indexOf(":")), show.slice(show.indexOf(":") + 1)] : ["", ""];
  const labeled = d.keys.filter((k) => (d.notes[k] ?? []).some((n) => n.ref === ref)).length;
  const p = d.profile;
  return (
    <div className="space-y-6 py-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
       <div>
        <Link to="/app/datasets" className="inline-flex items-center gap-1 text-sm text-muted hover:text-ink"><ArrowLeft className="h-4 w-4" /> Datasets</Link>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight">{name || "Dataset"}</h1>
        <p className="mt-1 text-sm text-muted">{d.keys.length} photo{d.keys.length === 1 ? "" : "s"}</p>
       </div>
       <Button size="sm" className="h-10" onClick={() => setLabeling(true)}><Tag className="h-4 w-4" /> Label</Button>
      </div>
      {labeling && <LabelDialog id={id} profile={d.profile} onClose={() => setLabeling(false)} />}

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
          {p.summary.includes("Places:") && <p className="text-sm text-muted">{p.summary.slice(p.summary.indexOf("Places:"))}</p>}
        </>}
      </Card>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium">Labels</span>
        {kinds.length === 0 && <span className="text-sm text-muted">None yet: a model trained on these photos labels them.</span>}
        {kinds.map(([k, label]) => (
          <button key={k} onClick={() => setShow(k)}
                  className={cx("rounded-full border px-3 py-1 text-sm", show === k ? "border-brand-500 bg-brand-50 text-brand-700" : "border-line hover:border-brand-500")}>
            {k.startsWith("class:") ? "Class" : "Boxes"}: {label}
          </button>
        ))}
        {show && <span className="text-sm text-muted">{labeled} of {d.keys.length} photos labeled</span>}
        {show && <MoreMenu items={(close) => <>
          {kind === "boxes" && <button className={menuItem} onClick={() => { close(); setEditing("rename"); }}><Pencil className="h-4 w-4" /> Rename</button>}
          <button className={cx(menuItem, "text-red-600")} onClick={() => { close(); setEditing("delete"); }}><Trash2 className="h-4 w-4" /> Delete</button>
        </>} />}
      </div>
      {editing === "rename" && (
        <NameDialog title="Rename these labels" name={ref} onClose={() => setEditing(null)}
                    hint={<>The English name the teacher looked for. A model that looks for the new name reuses these labels;
                      name what the camera sees, like "blue bin". If that name has labels already, a photo labeled for both keeps those.</>}
                    save={(to) => renameLabels(id, ref, to).then(() => { setShow(`boxes:${to.toLowerCase()}`); load(); })} />
      )}
      {editing === "delete" && (
        <ConfirmDialog title={`Delete the labels "${kinds.find(([k]) => k === show)?.[1] ?? ref}"?`} action="Delete"
                       onClose={() => setEditing(null)} run={() => forgetLabels(id, kind, ref).then(() => { setShow(""); load(); })}>
          The teacher's marks and your corrections for these labels go from every photo of this dataset (and from the same
          photos in your other datasets). Models already trained keep working; a new model labels them again.
        </ConfirmDialog>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {d.keys.slice(0, limit).map((key) => <Photo key={key} k={key} notes={(d.notes[key] ?? []).filter((n) => n.ref === ref && (kind === "class" ? n.kind === "class" : n.kind !== "class"))} />)}
      </div>
      {limit < d.keys.length && <div className="flex justify-center"><Button variant="ghost" onClick={() => setLimit(limit + PAGE)}>Show more</Button></div>}
    </div>
  );
}

// A photo with its labels drawn on it: the teacher's boxes, the user's points (which win), or its class.
function Photo({ k, notes }: { k: string; notes: PhotoNote[] }) {
  const user = notes.find((n) => n.source === "user");
  const model = notes.find((n) => n.source === "model");
  const cls = notes.find((n) => n.kind === "class");
  return (
    <div className="relative overflow-hidden rounded-xl bg-gray-100">
      <img src={libraryPhotoUrl(k)} alt="" loading="lazy" className="block h-auto w-full" />
      <svg viewBox="0 0 1 1" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
        {!user && model?.kind === "boxes" && (model.data.boxes as { box: number[] }[]).map((b, i) => (
          <rect key={i} x={b.box[0]} y={b.box[1]} width={b.box[2] - b.box[0]} height={b.box[3] - b.box[1]}
                fill="none" stroke="#7a6bfb" strokeWidth={2} vectorEffect="non-scaling-stroke" />
        ))}
        {user?.kind === "points" && (user.data.points as number[][]).map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={0.018} fill="#22c55e" stroke="white" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        ))}
      </svg>
      {(user || cls) && (
        <div className="absolute bottom-1.5 left-1.5 flex gap-1">
          {cls && <Badge tone={cls.source === "user" || user ? "good" : "brand"}>{(user ?? cls).data.label ?? ""}</Badge>}
          {user && <Badge tone="good">yours</Badge>}
        </div>
      )}
    </div>
  );
}

// Ask the teacher to label the dataset: count (a box on every thing named) or classify (one name per photo).
// The names the profile saw are offered as a start.
function LabelDialog({ id, profile, onClose }: { id: string; profile: DatasetProfile | null; onClose: () => void }) {
  const navigate = useNavigate();
  const [task, setTask] = useState<"count" | "classify">("count");
  const [names, setNames] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const list = names.split(",").map((n) => n.trim()).filter(Boolean);
  const go = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!list.length) { setError("Write what to label first."); return; }
    setBusy(true);
    try {
      const r = await labelDataset(id, task, list);
      navigate(`/app/job/${r.job}`);
    } catch (err: any) {
      setError(err.message);
      setBusy(false);
    }
  };
  const add = (n: string) => setNames(list.includes(n) ? names : [...list, n].join(", "));
  const row = "flex cursor-pointer items-start gap-3 rounded-xl border px-4 py-3 text-sm";
  return (
    <Dialog title="Label these photos" onClose={onClose}>
      <form onSubmit={go} className="space-y-4">
        {([["count", "Find things", "A box on every one, to count or detect them."],
           ["classify", "Sort photos", "Each photo gets one of the names."]] as const).map(([k, title, text]) => (
          <label key={k} className={cx(row, task === k ? "border-brand-500 bg-brand-50" : "border-line")}>
            <input type="radio" checked={task === k} onChange={() => setTask(k)} className="mt-0.5 accent-brand-600" />
            <span><span className="font-medium">{title}</span><br /><span className="text-muted">{text}</span></span>
          </label>
        ))}
        <input value={names} onChange={(e) => setNames(e.target.value)} className={field} aria-label="Names"
               placeholder={task === "count" ? "What to find, e.g. bottle, cup" : "The names, e.g. full, empty"} />
        {profile && profile.objects.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {profile.objects.map((o) => (
              <button type="button" key={o.name} onClick={() => add(o.name)}
                      className="rounded-full border border-line px-2.5 py-0.5 text-xs hover:border-brand-500">+ {o.name}</button>
            ))}
          </div>
        )}
        <p className="text-xs text-muted">The teacher marks a few photos first so you can check it understood. Then it labels them all.</p>
        {error && <Notice tone="bad">{error}</Notice>}
        <Button type="submit" className="w-full" disabled={busy}>{busy ? "Starting..." : "Start"}</Button>
      </form>
    </Dialog>
  );
}

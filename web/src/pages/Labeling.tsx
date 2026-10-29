import { ArrowLeft, Pencil, Plus, Sparkles, Trash2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { addClass, deleteClass, editClass, getLabeling, libraryPhotoUrl, reviewPhoto, teachLabeling, unreviewPhoto,
         type LabelClass, type Labeling, type LabelingPhoto, type Point } from "../api";
import { ConfirmDialog, Dialog, MoreMenu, NameDialog, menuItem } from "../components/library";
import { Badge, Button, Card, Notice, Spinner, cx } from "../components/ui";
import { classHue } from "./Dataset";
import { Editor } from "./Review";

// One labeling: its classes on the side (as a label book: color, description, how many marks), its photos with
// their marks (the teacher's dashed, yours solid) filtered by state, and the editor. Saving a photo marks it as
// reviewed and opens the next one to review, so going through them is one tap each.

type Filter = "all" | "todo" | "reviewed" | "empty";
const PAGE = 60;

export function LabelingPage() {
  const { id = "", lid = "" } = useParams();
  const navigate = useNavigate();
  const [l, setL] = useState<(Labeling & { photos: LabelingPhoto[] }) | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [limit, setLimit] = useState(PAGE);
  const [open, setOpen] = useState<number | null>(null);  // the photo being reviewed
  const [adding, setAdding] = useState(false);
  const load = useCallback(() => getLabeling(lid).then(setL).catch((e) => setError(e.message)), [lid]);
  useEffect(() => { load(); }, [load]);

  const keys = useMemo(() => l?.classes.map((c) => c.key) ?? [], [l]);
  const hue = (c: string) => classHue(Math.max(0, keys.indexOf(c)));
  if (error) return <div className="py-6"><Notice tone="bad">{error}</Notice></div>;
  if (!l) return <div className="flex justify-center py-16"><Spinner /></div>;

  const state = (p: LabelingPhoto) => p.reviewed ? "reviewed" : p.marks.length ? "teacher" : "empty";
  const counts = { all: l.photos.length, todo: l.photos.filter((p) => !p.reviewed).length,
                   reviewed: l.photos.filter((p) => p.reviewed).length, empty: l.photos.filter((p) => !p.marks.length).length };
  const shown = l.photos.filter((p) => filter === "all" || (filter === "todo" ? !p.reviewed : filter === "reviewed" ? p.reviewed : !p.marks.length));
  const photo = open === null ? null : l.photos.find((p) => p.id === open) ?? null;
  // A class change answers with the labeling that has it: a copy when a model was trained with this one.
  const changed = (r: { labeling: string }) => {
    if (r.labeling !== lid) {
      setNote("A model was trained with this labeling, so your change made a copy of it: you are in the copy now.");
      navigate(`/app/datasets/${id}/labelings/${r.labeling}`);
    } else load();
  };
  const next = (after: number) => {  // the next photo to review, in the order shown
    const i = shown.findIndex((p) => p.id === after);
    const rest = [...shown.slice(i + 1), ...shown.slice(0, Math.max(0, i))];
    setOpen(rest.find((p) => !p.reviewed && p.id !== after)?.id ?? null);
  };
  const save = async (pid: number, marks: Parameters<typeof reviewPhoto>[2]) => {
    try {
      await reviewPhoto(lid, pid, marks);
      await load();
      next(pid);
    } catch (e) { setError((e as Error).message); }
  };
  const teach = () => teachLabeling(lid).then((r) => navigate(`/app/job/${r.job}`)).catch((e) => setError(e.message));

  return (
    <div className="space-y-5 py-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link to={`/app/datasets/${id}`} className="inline-flex items-center gap-1 text-sm text-muted hover:text-ink"><ArrowLeft className="h-4 w-4" /> Dataset</Link>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight">{l.name}</h1>
          <p className="mt-1 text-sm text-muted">{l.task === "count" ? "Finds and counts things" : "Sorts the photos"} ·
            {" "}{l.reviewed} of {l.photos.length} reviewed by you{l.models.length ? ` · used by ${l.models.length} model${l.models.length === 1 ? "" : "s"}` : ""}</p>
        </div>
        <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row">
          <Button size="sm" variant="secondary" onClick={teach}><Sparkles className="h-4 w-4" /> Let the teacher mark them</Button>
          {counts.todo > 0 && <Button size="sm" onClick={() => setOpen(shown.find((p) => !p.reviewed)?.id ?? l.photos.find((p) => !p.reviewed)!.id)}>
            Review the photos left</Button>}
        </div>
      </div>
      {note && <Notice>{note}</Notice>}

      <div className="grid gap-5 lg:grid-cols-[240px_1fr]">
        <aside className="space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted">Classes</div>
          {l.classes.map((c, i) => <ClassItem key={c.id} c={c} hue={classHue(i)} lid={lid} last={l.classes.length === 1} onChanged={changed} />)}
          <Button size="sm" variant="ghost" onClick={() => setAdding(true)}><Plus className="h-4 w-4" /> Add a class</Button>
          {adding && <NameDialog title="Add a class" name="" onClose={() => setAdding(false)}
                                 hint="Name it by what the camera sees, like a color or a shape. Every photo you reviewed will need a look again for it."
                                 save={(v) => addClass(lid, v).then(changed)} />}
        </aside>

        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap gap-1.5">
            {([["all", "All"], ["todo", "To review"], ["reviewed", "Reviewed"], ["empty", "No marks"]] as const).map(([k, label]) => (
              <button key={k} onClick={() => { setFilter(k); setLimit(PAGE); }}
                      className={cx("rounded-full border px-3 py-1 text-sm", filter === k ? "border-brand-500 bg-brand-50 font-medium text-brand-700" : "border-line text-muted hover:border-brand-500")}>
                {label} {counts[k]}
              </button>
            ))}
          </div>
          <div className="grid grid-cols-2 items-start gap-3 sm:grid-cols-3 xl:grid-cols-4">
            {shown.slice(0, limit).map((p) => (
              <button key={p.id} onClick={() => setOpen(p.id)} className="relative overflow-hidden rounded-xl bg-gray-100 text-left">
                <img src={libraryPhotoUrl(p.id)} alt="" loading="lazy" className="block h-auto w-full" />
                {l.task === "count" && <Marks marks={p.marks} hue={hue} />}
                <div className="absolute bottom-1.5 left-1.5 flex gap-1">
                  {l.task === "classify" && p.marks[0] && <Badge tone={p.reviewed ? "good" : "brand"}>{p.marks[0].class}</Badge>}
                  <Badge tone={state(p) === "reviewed" ? "good" : "neutral"}>
                    {state(p) === "reviewed" ? "reviewed" : state(p) === "teacher" ? "teacher" : "no marks"}</Badge>
                </div>
              </button>
            ))}
          </div>
          {limit < shown.length && <div className="flex justify-center"><Button variant="ghost" onClick={() => setLimit(limit + PAGE)}>Show more</Button></div>}
        </div>
      </div>

      {photo && l.task === "count" && (
        <Editor key={photo.id} labels={keys} hue={hue} media={(p, big) => libraryPhotoUrl(Number(p), big)}
                editing={{ key: String(photo.id), image: String(photo.id), title: photo.reviewed ? "Reviewed: fix if needed" : "Review this photo",
                           points: photo.marks.filter((m) => m.x !== null).map((m) => [m.x!, m.y!, m.class] as Point) }}
                teacher={photo.marks.filter((m) => m.source === "teacher" && m.x !== null).map((m) => [m.x!, m.y!, m.class] as Point)}
                onClose={() => setOpen(null)}
                onDone={(points) => save(photo.id, points.map(([x, y, c]) => ({ class: c, x, y })))} />
      )}
      {photo && l.task === "classify" && (
        <Dialog title={photo.reviewed ? "Reviewed: change it if needed" : "Which class is this photo?"} onClose={() => setOpen(null)}>
          <div className="space-y-4">
            <img src={libraryPhotoUrl(photo.id, true)} alt="" className="max-h-[50dvh] w-full rounded-lg object-contain" />
            <div className="flex flex-wrap gap-2">
              {keys.map((k) => (
                <Button key={k} size="sm" variant={photo.marks[0]?.class === k ? "primary" : "secondary"}
                        onClick={() => save(photo.id, [{ class: k, x: null, y: null }])}>{k}</Button>
              ))}
            </div>
            {photo.reviewed && <Button size="sm" variant="ghost" onClick={() => unreviewPhoto(lid, photo.id).then(load).then(() => setOpen(null))}>
              Mark as not reviewed</Button>}
          </div>
        </Dialog>
      )}
    </div>
  );
}

// A count photo's marks: yours solid, the teacher's dashed (until you review the photo).
function Marks({ marks, hue }: { marks: LabelingPhoto["marks"]; hue: (c: string) => number }) {
  return (
    <svg viewBox="0 0 1 1" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
      {marks.filter((m) => m.x !== null).map((m, i) => m.w ? (
        <rect key={i} x={m.x! - m.w / 2} y={m.y! - (m.h ?? 0) / 2} width={m.w} height={m.h ?? 0} fill="none"
              stroke={`hsl(${hue(m.class)} 80% 50%)`} strokeWidth={2} vectorEffect="non-scaling-stroke"
              strokeDasharray={m.source === "teacher" ? "4 3" : undefined} />
      ) : (
        <circle key={i} cx={m.x!} cy={m.y!} r={0.02} fill={m.source === "user" ? `hsl(${hue(m.class)} 80% 50%)` : "none"}
                stroke={`hsl(${hue(m.class)} 80% 50%)`} strokeWidth={2} vectorEffect="non-scaling-stroke"
                strokeDasharray={m.source === "teacher" ? "3 2" : undefined} />
      ))}
    </svg>
  );
}

// A class of the label book: rename it (its marks follow), describe it, delete it.
function ClassItem({ c, hue, lid, last, onChanged }:
  { c: LabelClass; hue: number; lid: string; last: boolean; onChanged: (r: { labeling: string }) => void }) {
  const [dialog, setDialog] = useState<"rename" | "describe" | "delete" | null>(null);
  return (
    <Card className="flex items-start gap-2 p-3">
      <i className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: `hsl(${hue} 70% 45%)` }} />
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium">{c.key}</div>
        <div className="text-xs text-muted">{c.description || "No description"} · {c.marks} mark{c.marks === 1 ? "" : "s"}</div>
      </div>
      <MoreMenu items={(close) => <>
        <button className={menuItem} onClick={() => { close(); setDialog("rename"); }}><Pencil className="h-4 w-4" /> Rename</button>
        <button className={menuItem} onClick={() => { close(); setDialog("describe"); }}><Pencil className="h-4 w-4" /> Describe</button>
        {!last && <button className={cx(menuItem, "text-red-600")} onClick={() => { close(); setDialog("delete"); }}><Trash2 className="h-4 w-4" /> Delete</button>}
      </>} />
      {dialog === "rename" && <NameDialog title="Rename class" name={c.key} onClose={() => setDialog(null)}
                                          hint="In English, by what the camera sees. Its marks keep it."
                                          save={(v) => editClass(lid, c.id, { key: v }).then(onChanged)} />}
      {dialog === "describe" && <NameDialog title="Describe the class" name={c.description} onClose={() => setDialog(null)}
                                            hint="What counts as one, in your words: it helps the teacher and whoever reviews."
                                            save={(v) => editClass(lid, c.id, { description: v }).then(onChanged)} />}
      {dialog === "delete" && (
        <ConfirmDialog title={`Delete the class "${c.key}"?`} action="Delete" onClose={() => setDialog(null)}
                       run={() => deleteClass(lid, c.id).then(onChanged)}>
          Its {c.marks} mark{c.marks === 1 ? "" : "s"} go with it. The other classes keep theirs.
        </ConfirmDialog>
      )}
    </Card>
  );
}

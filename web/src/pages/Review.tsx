import { RotateCcw, X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import * as API from "../api";
import type { Fixes, Point, Review } from "../api";
import { Badge, Button, Card, cx } from "../components/ui";

// Human review of what the models decided: fix the teacher's marks, keep or drop 3D models, look at the
// synthetic images and the test predictions, then retrain from the first step the fixes affect.

type Tab = "photos" | "assets" | "renders" | "test";
type Editing = { key: string; image: string; title: string; points: Point[] };

// calibrating: before the first run, only the teacher's marks on a few photos; fixes teach it, then train.
// labeling: a job that only labels a dataset, so its last step labels every photo instead of training.
export default function ReviewPanel({ job, onRerun, calibrating = false, labeling = false }:
  { job: string; onRerun: (news?: string) => void; calibrating?: boolean; labeling?: boolean }) {
  const [review, setReview] = useState<Review | null>(null);
  const [tab, setTab] = useState<Tab>("test");
  const [fixes, setFixes] = useState<Fixes>({ labels: {}, assets: {}, points: {} });
  const [editing, setEditing] = useState<Editing | null>(null);
  const [applying, setApplying] = useState(false);

  useEffect(() => { API.getReview(job).then((r) => { setReview(r); if (!r.test.length) setTab("photos"); }).catch(() => {}); }, [job]);
  if (!review) return null;
  const any = review.photos.length || review.assets.length || review.renders.length || review.test.length;
  if (!any) return null;

  const counting = review.task === "count";
  const media = (p: string, big = false) => API.mediaUrl(job, p, big);
  const hue = (c: string) => (review.labels.indexOf(c) * 137) % 360;
  const marksOf = (key: string) => fixes.points[key] ?? review.photos.find((x) => x.key === key)?.points ?? [];
  // The check of a count may ask about some classes only (the others were checked before): only those are shown
  // and edited, the marks of the others go back unchanged.
  const only = calibrating && review.check?.length ? new Set(review.check) : null;
  const shown = (points: Point[]) => only ? points.filter((p) => only.has(p[2])) : points;
  const n = Object.keys(fixes.labels).length + Object.keys(fixes.assets).length + Object.keys(fixes.points).length;
  const set = (kind: keyof Fixes, key: string, value: any) => setFixes((f) => {
    const next = { ...f, [kind]: { ...f[kind] } } as Fixes;
    if (value === undefined) delete (next[kind] as any)[key]; else (next[kind] as any)[key] = value;
    return next;
  });

  const apply = async (train = false) => {
    setApplying(true);
    try {
      const r = train ? await (labeling ? API.labelAll : API.trainCalibrated)(job, fixes) : await API.applyFixes(job, fixes, calibrating);
      if (train) { onRerun(); return; }
      setFixes({ labels: {}, assets: {}, points: {} });
      if (r.rerun_from) onRerun();
    } catch (e) {
      // 409: the job moved on elsewhere (the same job open on another device). Show where it is now.
      if (e instanceof API.ApiError && e.status === 409) { onRerun(e.message); return; }
      alert((e as Error).message);
    }
    setApplying(false);
  };

  const tabs: [Tab, string, number][] = [["test", "Test results", review.test.length], ["photos", "Your photos", review.photos.length],
                                         ["assets", "3D models", review.assets.length], ["renders", "Synthetic images", review.renders.length]];
  const choices = [...review.labels, "unsure"];

  return (
    <section className="space-y-4">
      {calibrating ? (
        <div>
          <h2 className="text-xl font-bold tracking-tight">Check the teacher first</h2>
          <p className="text-sm text-muted">Before training, a large model marked a few of your photos. Fix what it got wrong and
            press "Teach it": it learns from your fixes and marks a few more photos. When they look right, train.</p>
        </div>
      ) : <div>
        <h2 className="text-xl font-bold tracking-tight">Check what the models decided</h2>
        <p className="text-sm text-muted">Fix anything that looks wrong, then retrain. You never label from scratch.
          The teacher learns from your fixes and relabels every photo with them, so a few corrected photos are enough.</p>
      </div>}
      {review.teacher_rules?.length ? (
        <div className="rounded-lg border p-3 text-sm">
          <p className="font-medium">What the teacher learned from your fixes</p>
          <ul className="mt-1 list-disc pl-5 text-muted">{review.teacher_rules.map((r) => <li key={r}>{r}</li>)}</ul>
        </div>
      ) : null}
      <div className="-mx-4 flex gap-1 overflow-x-auto px-4 pb-1">
        {tabs.filter(([, , c]) => c > 0).map(([k, label, c]) => (
          <button key={k} onClick={() => setTab(k)}
                  className={cx("shrink-0 rounded-full px-4 py-2 text-sm font-medium transition-colors",
                                tab === k ? "bg-ink text-white" : "bg-white text-muted ring-1 ring-line hover:text-ink")}>
            {label} <span className="opacity-60">{c}</span>
          </button>
        ))}
      </div>

      {tab === "test" && <Hint>{counting ? "Orange dots: what the model found. White rings: the objects really there. Tap a photo to fix the rings."
                                         : "What the model answered on your real test photos."}</Hint>}
      {tab === "photos" && <Hint>{counting ? "The marks the teacher model put on your photos. Tap a photo to add or remove marks."
                                           : "The teacher's label for each photo. Tap to change it."}
        {only && ` Only ${[...only].join(", ")} this time: you checked the other classes before.`}</Hint>}
      {tab === "assets" && <Hint>3D models found for your objects. Tap to keep or drop one.</Hint>}

      <div className="columns-2 gap-3 sm:columns-3 lg:columns-4 [&>*]:mb-3 [&>*]:break-inside-avoid">
        {tab === "photos" && counting && review.photos.map((p) => (
          <Tile key={p.key} src={p.image && media(p.image)} changed={p.key in fixes.points}
                overlay={<Dots points={shown(marksOf(p.key))} hue={hue} />}
                onClick={p.image ? () => setEditing({ key: p.key, image: p.image!, title: p.key, points: shown(marksOf(p.key)).map((x) => [...x] as Point) }) : undefined}>
            {p.unsure && !(p.key in fixes.points) ? <Badge tone="bad">teacher unsure</Badge> : <Badge>{shown(marksOf(p.key)).length} marked</Badge>}
            <span className="text-xs text-muted">{p.key in fixes.points || p.judge === "user" ? "you" : "teacher"}</span>
          </Tile>
        ))}
        {tab === "photos" && !counting && review.photos.map((p) => {
          const now = fixes.labels[p.key] ?? p.label!;
          return (
            <Tile key={p.key} src={p.image && media(p.image)} changed={p.key in fixes.labels} title={p.reason}
                  onClick={() => {
                    const next = choices[(choices.indexOf(now) + 1) % choices.length];
                    set("labels", p.key, next === p.label && p.judge !== "user" ? undefined : next);
                  }}>
              <Badge tone="brand">{now}</Badge>
              <span className="text-xs text-muted">{p.key in fixes.labels || p.judge === "user" ? "you" : `teacher ${Math.round((p.confidence ?? 0) * 100)}%`}</span>
            </Tile>
          );
        })}
        {tab === "assets" && review.assets.map((a) => {
          const keep = a.uid in fixes.assets ? fixes.assets[a.uid] === "keep" : a.kept;
          return (
            <Tile key={a.uid} src={a.image && media(a.image)} changed={a.uid in fixes.assets} title={`${a.name ?? a.uid}: ${a.reason}`}
                  dim={!keep} onClick={() => { const next = keep ? "discard" : "keep"; set("assets", a.uid, (next === "keep") === a.kept ? undefined : next); }}>
              <Badge tone={keep ? "good" : "bad"}>{keep ? "kept" : "dropped"}</Badge>
              <span className="truncate text-xs text-muted">{a.class}{a.role === "prop" ? " (prop)" : ""}</span>
            </Tile>
          );
        })}
        {tab === "renders" && review.renders.map((r) => <Tile key={r} src={media(r)} />)}
        {tab === "test" && counting && review.test.map((t) => {
          const truth: Point[] = fixes.points[t.key] ?? t.truth;
          const found = t.count ? Object.values(t.count).reduce((a, b) => a + b, 0) : t.pred.length;
          const ok = found === truth.length;
          return (
            <Tile key={t.key} src={media(t.image)} changed={t.key in fixes.points}
                  overlay={<><Dots points={truth} hue={hue} ring /><Dots points={t.pred} hue={hue} predicted /></>}
                  onClick={() => setEditing({ key: t.key, image: t.image, title: "Fix the marks the model is scored against", points: truth.map((x) => [...x] as Point) })}>
              <Badge tone={ok ? "good" : "bad"}>counted {found}</Badge>
              <span className="text-xs text-muted">real {truth.length}</span>
            </Tile>
          );
        })}
        {tab === "test" && !counting && review.test.map((t) => (
          <Tile key={t.key ?? t.file} src={media(t.image)} title={t.file}>
            <Badge tone={t.truth === t.pred ? "good" : "bad"}>{t.pred}</Badge>
            <span className="text-xs text-muted">real: {t.truth}</span>
          </Tile>
        ))}
      </div>

      {(n > 0 || calibrating) && (
        <div className="sticky bottom-0 -mx-4 border-t border-line bg-white/95 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur">
          <div className="flex items-center justify-between gap-3">
            <span className="text-sm text-muted">{n} correction{n === 1 ? "" : "s"}</span>
            <div className="flex gap-2">
              {n > 0 && <Button variant="ghost" size="sm" onClick={() => setFixes({ labels: {}, assets: {}, points: {} })}>Undo all</Button>}
              {calibrating ? <>
                <Button variant="ghost" size="sm" onClick={() => apply()} disabled={applying || n === 0}>Teach it</Button>
                <Button size="sm" onClick={() => apply(true)} disabled={applying}>
                  {labeling ? (n ? "Keep fixes, label all" : "Looks right, label all") : n ? "Keep fixes and train" : "Looks right, train"}
                </Button>
              </> : <Button size="sm" onClick={() => apply()} disabled={applying}>Apply and retrain</Button>}
            </div>
          </div>
        </div>
      )}

      {editing && (
        <Editor editing={editing} labels={only ? [...only] : review.labels.slice(1)} media={media} hue={hue}
                teacher={shown(review.photos.find((x) => x.key === editing.key)?.teacher_points ?? [])}
                onClose={() => setEditing(null)}
                onDone={(points) => {
                  const p = review.photos.find((x) => x.key === editing.key);
                  const t = review.test.find((x) => x.key === editing.key);
                  const original = p?.points ?? t?.truth;
                  const all = only && p ? [...marksOf(p.key).filter((x) => !only.has(x[2])), ...points] : points;
                  const same = (a?: Point[]) => JSON.stringify(a?.map(String).sort()) === JSON.stringify(all.map(String).sort());
                  set("points", editing.key, same(original) ? undefined : all);
                  setEditing(null);
                }} />
      )}
    </section>
  );
}

function Hint({ children }: { children: ReactNode }) {
  return <p className="text-sm text-muted">{children}</p>;
}

function Tile({ src, overlay, children, onClick, changed, dim, title }: {
  src?: string | null; overlay?: ReactNode; children?: ReactNode; onClick?: () => void; changed?: boolean; dim?: boolean;
  title?: string;
}) {
  const Tag = onClick ? "button" : "div";
  return (
    <Card className={cx("overflow-hidden text-left", changed && "ring-2 ring-ember-500", onClick && "transition hover:shadow-md")}>
      <Tag onClick={onClick} title={title} className="block w-full">
        <div className={cx("relative bg-gray-100", dim && "opacity-40")}>
          {src ? <img src={src} alt="" loading="lazy" className="block w-full" /> : <div className="aspect-square" />}
          {overlay}
        </div>
        {children && <div className="flex items-center justify-between gap-2 px-3 py-2">{children}</div>}
      </Tag>
    </Card>
  );
}

// Count task marks: [x, y, class] in 0..1 of the upright image.
// Marks are coloured by class; a model's predictions are orange so they stand apart from the white truth rings.
function Dots({ points, hue, ring, predicted, onRemove }: { points: Point[]; hue: (c: string) => number; ring?: boolean;
                                                          predicted?: boolean; onRemove?: (i: number) => void }) {
  return <>{points.map((p, i) => (
    <span key={i} title={p[2]} onClick={onRemove ? (e) => { e.stopPropagation(); onRemove(i); } : undefined}
          className={cx("absolute -translate-x-1/2 -translate-y-1/2 rounded-full",
                        ring ? "h-4 w-4 border-2 border-white shadow-[0_0_0_1px_rgb(0_0_0/0.5)]"
                             : predicted ? "h-2.5 w-2.5 border border-white bg-ember-500 shadow"
                             : onRemove ? "h-7 w-7 cursor-pointer border-2 border-white shadow"
                             : "h-2.5 w-2.5 border border-white shadow")}
          style={{ left: `${p[0] * 100}%`, top: `${p[1] * 100}%`,
                   ...(!ring && !predicted && { background: `hsl(${hue(p[2])} 85% 50% / ${onRemove ? 0.45 : 1})` }) }} />
  ))}</>;
}

// Marks of two classes on one spot put one object under both (review.stacked on the server; same distance).
const SAME_SPOT = 0.01;
const stacked = (points: Point[]) => points.filter((p, i) => points.some((q, j) => j !== i && q[2] !== p[2]
  && Math.abs(q[0] - p[0]) < SAME_SPOT && Math.abs(q[1] - p[1]) < SAME_SPOT)).length;

function Editor({ editing, labels, media, hue, teacher, onClose, onDone }: {
  editing: Editing; labels: string[]; media: (p: string, big?: boolean) => string; hue: (c: string) => number;
  teacher: Point[]; onClose: () => void; onDone: (p: Point[]) => void;
}) {
  const [points, setPoints] = useState<Point[]>(editing.points);
  const [cls, setCls] = useState(labels[0]);
  const piled = stacked(points);
  return (
    <div className="fixed inset-0 z-30 flex items-end justify-center bg-black/50 sm:items-center sm:p-6" onClick={onClose}>
      <div className="flex max-h-dvh w-full max-w-3xl flex-col overflow-hidden bg-white sm:rounded-[var(--radius-card)]" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            <p className="truncate font-medium">{editing.title}</p>
            <p className="text-xs text-muted">Tap where an object is missing; tap a mark to remove it. {points.length} marked.</p>
            {piled > 0 && <p className="text-xs text-red-600">{piled} marks of different classes sit on the same object.
              Tap it to remove the wrong ones until one class is left.</p>}
          </div>
          <button onClick={onClose} className="rounded-lg p-2 text-muted hover:bg-gray-100" aria-label="Close"><X className="h-5 w-5" /></button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto bg-gray-900">
          <div className="relative mx-auto w-fit">
            <img src={media(editing.image, true)} alt="" className="block max-h-[70dvh] w-auto cursor-crosshair select-none"
                 onClick={(e) => {
                   const r = e.currentTarget.getBoundingClientRect();
                   setPoints([...points, [+((e.clientX - r.left) / r.width).toFixed(4), +((e.clientY - r.top) / r.height).toFixed(4), cls]]);
                 }} />
            <Dots points={points} hue={hue} onRemove={(i) => setPoints(points.filter((_, j) => j !== i))} />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 border-t border-line px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
          {labels.length > 1 && (
            <select value={cls} onChange={(e) => setCls(e.target.value)} className="h-9 rounded-lg border border-line px-2 text-sm">
              {labels.map((l) => <option key={l}>{l}</option>)}
            </select>
          )}
          {teacher.length > 0 && (
            <Button variant="ghost" size="sm" onClick={() => setPoints(teacher.map((x) => [...x] as Point))}>
              <RotateCcw className="h-4 w-4" /> Teacher's marks
            </Button>
          )}
          <div className="flex-1" />
          <Button variant="secondary" size="sm" onClick={onClose}>Cancel</Button>
          <Button size="sm" onClick={() => onDone(points)} disabled={piled > 0}>Done</Button>
        </div>
      </div>
    </div>
  );
}

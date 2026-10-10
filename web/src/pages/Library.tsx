import { Box, Cpu, Images, Mountain, Package, Pencil, Plus, Upload } from "lucide-react";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { cardUrl, createDataset, addDatasetPhotos, getLibrary, libraryPhotoUrl, saveKit, uploadObject,
         type DatasetCard, type Kit, type LibraryData, type ModelCard } from "../api";
import { Dialog, ItemMenu, ModelMenu, SelectBar, Tick, field, useSelection } from "../components/library";
import { Badge, Button, ButtonLink, Card, Notice, Spinner, cx } from "../components/ui";
import { glbCard } from "../glbCard";
import { useApp } from "./AppShell";

// The library pages: models (the visitor's jobs), datasets of real photos, 3D objects and scenes, all kept for good,
// and kits (named lists of them that a new model starts from).

const TASKS: Record<string, string> = { count: "Count", roi_classification: "Recognize", classify: "Recognize" };
const pretty = (name: string) => name.replace(/_/g, " ");
const day = (t: number) => new Date(t * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });

function useLibrary() {
  const { stamp } = useApp();
  const [lib, setLib] = useState<LibraryData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reload = useCallback(() => {
    getLibrary().then((r) => setLib({ ...r, models: [...r.models].sort((a, b) => b.created - a.created) }))
      .catch((e) => setError(e.message));
  }, []);
  useEffect(reload, [stamp, reload]);
  return { lib, error, reload };
}

function Page({ title, intro, action, children }: { title: string; intro: string; action?: ReactNode; children: ReactNode }) {
  return (
    <div className="space-y-6 py-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted">{intro}</p>
        </div>
        {action}
      </div>
      {children}
    </div>
  );
}

function Loading({ error }: { error: string | null }) {
  if (error) return <Notice tone="bad">{error}</Notice>;
  return <div className="grid animate-pulse gap-4 sm:grid-cols-2 lg:grid-cols-3">
    {[0, 1, 2].map((i) => <div key={i} className="h-44 rounded-[var(--radius-card)] bg-gray-200" />)}
  </div>;
}

function Empty({ icon, title, children, action }: { icon: ReactNode; title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <Card className="flex flex-col items-center gap-3 px-6 py-12 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-50 text-brand-600">{icon}</div>
      <div className="text-lg font-semibold">{title}</div>
      <p className="max-w-md text-sm text-muted">{children}</p>
      {action}
    </Card>
  );
}

const newModel = <NewModelLink />;

// A new chat, not the one left open (Interview keeps it while the user looks at other pages).
function NewModelLink() {
  return <ButtonLink to="/app" state={{ fresh: Date.now() }} className="h-10 px-4 text-sm"><Plus className="h-4 w-4" /> New model</ButtonLink>;
}

function Score({ m }: { m: NonNullable<ModelCard["model"]> }) {
  if (m.score == null) return null;
  return <span className="font-semibold">
    {m.metric === "f1" ? `F1 ${m.score.toFixed(2)}` : `${Math.round(m.score * 100)}% right`}
  </span>;
}

export function Models() {
  const { lib, error, reload } = useLibrary();
  const jobs = lib?.models;
  return (
    <Page title="Models" intro="Every model you built, with how well it did on your real photos." action={newModel}>
      {!jobs ? <Loading error={error} /> : jobs.length === 0 ? (
        <Empty icon={<Cpu className="h-6 w-6" />} title="No models yet" action={newModel}>
          Describe what your camera should see in the chat and we build a model sized for your device.
        </Empty>
      ) : (
        <Card className="divide-y divide-line">
          {jobs.map((j) => (
            <div key={j.job} className="flex items-center gap-2 pr-3 hover:bg-gray-50">
            <Link to={`/app/job/${j.job}`} className="flex min-w-0 flex-1 items-center gap-4 py-3.5 pl-4">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gray-100 text-muted">
                <Cpu className="h-5 w-5" strokeWidth={1.75} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="truncate font-medium">{j.title ?? pretty(j.job)}</span>
                  {j.task && <Badge>{TASKS[j.task] ?? j.task}</Badge>}
                  {j.running && <Badge tone="ember">Training</Badge>}
                </div>
                <div className="mt-0.5 truncate text-sm text-muted">
                  {j.classes.map(pretty).join(", ")}{j.hardware && ` · ${j.hardware}`} · {day(j.created)}
                </div>
              </div>
              <div className="hidden shrink-0 text-right text-sm sm:block">
                {j.model ? <><Score m={j.model} />{j.model.kb != null && <div className="text-muted">{Math.round(j.model.kb)} KB</div>}</>
                  : !j.running && <span className="text-muted">Not finished</span>}
              </div>
            </Link>
            <ModelMenu job={j.job} name={j.title ?? pretty(j.job)} onChanged={reload} onDeleted={reload} />
            </div>
          ))}
        </Card>
      )}
    </Page>
  );
}

export function Datasets() {
  const { lib, error, reload } = useLibrary();
  const { sel, toggle, clear } = useSelection();
  const [upload, setUpload] = useState<{ id: string; name: string } | "new" | null>(null);
  const sets = lib?.datasets;
  const action = <Button size="sm" className="h-10" onClick={() => setUpload("new")}><Upload className="h-4 w-4" /> Upload photos</Button>;
  return (
    <Page title="Datasets" intro="Every real photo you gave us, kept as datasets. Photos you attach in a chat become a dataset right away." action={action}>
      {!sets ? <Loading error={error} /> : sets.length === 0 ? (
        <Empty icon={<Images className="h-6 w-6" />} title="No photos yet" action={action}>
          Upload photos here or attach them in the chat. Each set becomes a dataset you can use again in other models.
        </Empty>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {sets.map((d) => (
            <Card key={d.id} className={cx("relative overflow-hidden", sel.has(d.id) && "ring-2 ring-brand-500")}>
              <Tick on={sel.has(d.id)} onToggle={() => toggle(d.id)} />
              <div className="grid aspect-[2/1] grid-cols-4 gap-px bg-line">
                {[0, 1, 2, 3].map((i) => d.sample[i]
                  ? <img key={i} src={libraryPhotoUrl(d.sample[i])} alt="" loading="lazy" className="h-full w-full object-cover" />
                  : <div key={i} className="bg-gray-100" />)}
              </div>
              <div className="p-4">
                <div className="flex items-start justify-between gap-2">
                  <Link to={`/app/datasets/${d.id}`} className="truncate font-medium hover:text-brand-600">{d.name}</Link>
                  <ItemMenu kind="datasets" id={d.id} name={d.name} onChanged={reload}
                            extra={<button className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-gray-50"
                                           onClick={() => setUpload({ id: d.id, name: d.name })}><Plus className="h-4 w-4" /> Add photos</button>} />
                </div>
                <div className="text-sm text-muted">
                  {d.photos} photo{d.photos === 1 ? "" : "s"}
                  {d.labelings.length > 0 && ` · ${d.labelings.map((l) => l.name).join(", ")}`}
                </div>
                <Seen d={d} />
                <Used jobs={d.jobs} />
              </div>
            </Card>
          ))}
        </div>
      )}
      <SelectBar kind="datasets" picked={(sets ?? []).filter((d) => sel.has(d.id))} kits={lib?.kits ?? []} onClear={clear} onDone={reload} />
      {upload && <PhotoUpload into={upload === "new" ? null : upload} onClose={() => setUpload(null)} onDone={reload} />}
    </Page>
  );
}

function PhotoUpload({ into, onClose, onDone }: { into: { id: string; name: string } | null; onClose: () => void; onDone: () => void }) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const go = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      if (into) await addDatasetPhotos(into.id, files);
      else await createDataset(name.trim(), files);
      onClose();
      onDone();
    } catch (err: any) {
      setError(err.message);
      setBusy(false);
    }
  };
  return (
    <Dialog title={into ? `Add photos to ${into.name}` : "New dataset"} onClose={onClose}>
      <form onSubmit={go} className="space-y-4">
        {!into && <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name, e.g. Shop shelf" autoFocus className={field} aria-label="Name" />}
        <label className="flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed border-line px-4 py-8 text-center text-sm text-muted hover:border-brand-500">
          <Images className="h-6 w-6" />
          {files.length ? `${files.length} photos chosen` : "Choose photos"}
          <input type="file" accept="image/*" multiple className="hidden" onChange={(e) => setFiles([...(e.target.files ?? [])])} />
        </label>
        {error && <Notice tone="bad">{error}</Notice>}
        <Button type="submit" className="w-full" disabled={busy || !files.length || (!into && !name.trim())}>
          {busy ? "Uploading..." : "Upload"}
        </Button>
      </form>
    </Dialog>
  );
}

// The objects seen in a dataset's photos, as chips, or that we are still looking.
function Seen({ d }: { d: DatasetCard }) {
  if (!d.profile) return d.profiling
    ? <div className="mt-2 flex items-center gap-2 text-xs text-muted"><Spinner className="h-3 w-3" /> Looking at the photos...</div>
    : null;
  return <div className="mt-2 flex flex-wrap gap-1">
    {d.profile.objects.slice(0, 4).map((o) => <Badge key={o.name}>{o.name}</Badge>)}
  </div>;
}

function Used({ jobs }: { jobs: string[] }) {
  if (jobs.length === 0) return <div className="mt-2 text-xs text-muted">Not used by a model yet</div>;
  return <div className="mt-2 truncate text-xs text-muted">
    Used by {jobs.length === 1 ? <Link to={`/app/job/${jobs[0]}`} className="text-brand-600 hover:underline">{pretty(jobs[0])}</Link>
      : `${jobs.length} models`}
  </div>;
}

const SOURCES: Record<string, string> = { photos: "From your photos", scene: "From a scene", upload: "Uploaded" };
const licenseName = (l: string) => `${l.toLowerCase().startsWith("cc") ? "" : "CC "}${l.toUpperCase()}`;

export function Objects() {
  const { lib, error, reload } = useLibrary();
  const { sel, toggle, clear } = useSelection();
  const [upload, setUpload] = useState(false);
  // Only the user's own: built from their photos, taken out of their scenes or uploaded. Models found on the
  // internet are our catalog, not theirs (2026-10-11).
  const all = lib?.objects ?? [];
  const [rejected, setRejected] = useState(false);
  const shown = all.filter((o) => rejected || o.kept);
  const action = <Button size="sm" className="h-10" onClick={() => setUpload(true)}><Upload className="h-4 w-4" /> Upload 3D model</Button>;
  return (
    <Page title="Objects" intro="Your 3D models: built from your photos, taken out of your scenes or uploaded. The approved ones go into synthetic images." action={action}>
      {!lib ? <Loading error={error} /> : all.length === 0 ? (
        <Empty icon={<Box className="h-6 w-6" />} title="No 3D objects yet" action={action}>
          3D models built from your photos or taken out of your scenes are kept here. You can also upload your own (.glb).
        </Empty>
      ) : (
        <>
          <div className="flex justify-end">
            <label className="flex items-center gap-2 text-sm text-muted">
              <input type="checkbox" checked={rejected} onChange={(e) => setRejected(e.target.checked)} className="accent-brand-600" />
              Show rejected
            </label>
          </div>
          {shown.length === 0 && (
            <p className="py-8 text-center text-sm text-muted">
              All your 3D models were rejected by the check. Tick "Show rejected" to see them.
            </p>
          )}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {shown.map((o) => (
              <Card key={o.id} className={cx("relative overflow-hidden", !o.kept && "opacity-60", sel.has(o.id) && "ring-2 ring-brand-500")}>
                <Tick on={sel.has(o.id)} onToggle={() => toggle(o.id)} />
                <div className="flex aspect-square items-center justify-center bg-gray-100 text-gray-300">
                  {o.thumb ? <img src={cardUrl("object", o.id)} alt="" loading="lazy" className="h-full w-full object-contain" />
                    : <Box className="h-10 w-10" strokeWidth={1.25} />}
                </div>
                <div className="space-y-1 p-3">
                  <div className="flex items-start justify-between gap-1">
                    <div className="truncate text-sm font-medium" title={o.name}>{o.name}</div>
                    <ItemMenu kind="objects" id={o.id} name={o.name} onChanged={reload} />
                  </div>
                  <div className="flex flex-wrap gap-1">
                    <Badge>{pretty(o.concept)}</Badge>
                    {SOURCES[o.source] && <Badge tone="ember">{SOURCES[o.source]}</Badge>}
                    {!o.kept && <Badge tone="bad">Rejected</Badge>}
                  </div>
                  {o.author && (
                    <div className="truncate text-xs text-muted">
                      by {o.url ? <a href={o.url} target="_blank" rel="noreferrer" className="hover:underline">{o.author}</a> : o.author}
                      {o.license && ` · ${licenseName(o.license)}`}
                    </div>
                  )}
                  <Used jobs={o.jobs} />
                </div>
              </Card>
            ))}
          </div>
        </>
      )}
      <SelectBar kind="objects" picked={all.filter((o) => sel.has(o.id))} kits={lib?.kits ?? []} onClear={clear} onDone={reload} />
      {upload && <ObjectUpload onClose={() => setUpload(false)} onDone={reload} />}
    </Page>
  );
}

function ObjectUpload({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [card, setCard] = useState<{ blob: Blob; url: string } | null>(null);
  const [name, setName] = useState("");
  const [concept, setConcept] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pick = async (f: File | undefined) => {
    setError(null);
    setCard(null);
    setFile(f ?? null);
    if (!f) return;
    if (!name) setName(f.name.replace(/\.glb$/i, "").replace(/[_-]+/g, " "));
    try {
      const blob = await glbCard(f);
      if (blob) setCard({ blob, url: URL.createObjectURL(blob) });
    } catch {
      setError("This file does not open as a 3D model (.glb).");
      setFile(null);
    }
  };
  useEffect(() => () => { if (card) URL.revokeObjectURL(card.url); }, [card]);
  const go = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      await uploadObject(name.trim(), concept.trim(), file!, card?.blob ?? null);
      onClose();
      onDone();
    } catch (err: any) {
      setError(err.message);
      setBusy(false);
    }
  };
  return (
    <Dialog title="Upload a 3D model" onClose={onClose}>
      <form onSubmit={go} className="space-y-4">
        <label className="flex aspect-[4/3] cursor-pointer flex-col items-center justify-center gap-2 overflow-hidden rounded-xl border-2 border-dashed border-line text-center text-sm text-muted hover:border-brand-500">
          {card ? <img src={card.url} alt="" className="h-full w-full object-contain" />
            : <><Box className="h-6 w-6" />{file ? "Opening the model..." : "Choose a .glb file"}</>}
          <input type="file" accept=".glb,model/gltf-binary" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
        </label>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name, e.g. Red mug" className={field} aria-label="Name" />
        <input value={concept} onChange={(e) => setConcept(e.target.value)} placeholder="What is it? e.g. mug" className={field} aria-label="What is it" />
        {error && <Notice tone="bad">{error}</Notice>}
        <Button type="submit" className="w-full" disabled={busy || !file || !name.trim()}>{busy ? "Uploading..." : "Upload"}</Button>
      </form>
    </Dialog>
  );
}

const ROUTES: Record<string, string> = { support: "Things on one surface", street: "A street", room: "A room" };

export function Scenes() {
  const { lib, error, reload } = useLibrary();
  const { sel, toggle, clear } = useSelection();
  const objects = new Map((lib?.objects ?? []).map((o) => [o.id, o]));
  const datasets = new Map((lib?.datasets ?? []).map((d) => [d.id, d]));
  return (
    <Page title="Scenes" intro="Places rebuilt in 3D from one of your photos. Your objects are placed in them again at random, so the synthetic images look like your real place.">
      {!lib ? <Loading error={error} /> : lib.scenes.length === 0 ? (
        <Empty icon={<Mountain className="h-6 w-6" />} title="No scenes yet">
          When a model counts things on a surface seen from above (a table, a lawn), it rebuilds one of your photos in
          3D, takes out what moves and keeps it here, ready for your next models of that dataset.
        </Empty>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {lib.scenes.map((sc) => {
            const ds = sc.dataset ? datasets.get(sc.dataset) : undefined;
            return (
              <Card key={sc.id} className={cx("relative overflow-hidden", sel.has(sc.id) && "ring-2 ring-brand-500")}>
                <Tick on={sel.has(sc.id)} onToggle={() => toggle(sc.id)} />
                <div className="aspect-[4/3] bg-gray-100">
                  {sc.thumb && <img src={cardUrl("scene", sc.id)} alt="" loading="lazy" className="h-full w-full object-cover" />}
                </div>
                <div className="p-4">
                  <div className="flex items-start justify-between gap-2">
                    <div className="truncate font-medium" title={sc.name}>{sc.name}</div>
                    <ItemMenu kind="scenes" id={sc.id} name={sc.name} onChanged={reload} />
                  </div>
                  <div className="text-sm text-muted">
                    {ROUTES[sc.route] ?? sc.route}
                    {ds && <> · from <Link to={`/app/datasets/${ds.id}`} className="hover:text-brand-600">{ds.name}</Link></>}
                  </div>
                  {sc.objects.length > 0 && (
                    <div className="mt-3 flex gap-1.5 overflow-hidden" title="Taken out of the photo, placed again in every view">
                      {sc.objects.slice(0, 6).map((id) => {
                        const o = objects.get(id);
                        return <div key={id} className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-md bg-gray-100 text-gray-300">
                          {o?.thumb ? <img src={cardUrl("object", id)} alt={o.name} title={o.name} className="h-full w-full object-contain" />
                            : <Box className="h-5 w-5" strokeWidth={1.25} />}
                        </div>;
                      })}
                      {sc.objects.length > 6 && <span className="self-center text-xs text-muted">+{sc.objects.length - 6}</span>}
                    </div>
                  )}
                  <Used jobs={sc.jobs} />
                </div>
              </Card>
            );
          })}
        </div>
      )}
      <SelectBar kind="scenes" picked={lib?.scenes.filter((sc) => sel.has(sc.id)) ?? []} kits={lib?.kits ?? []} onClear={clear} onDone={reload} />
    </Page>
  );
}

export function Kits() {
  const { lib, error, reload } = useLibrary();
  const [edit, setEdit] = useState<Kit | "new" | null>(null);
  const action = <Button size="sm" className="h-10" onClick={() => setEdit("new")}><Plus className="h-4 w-4" /> New kit</Button>;
  const datasets = new Map((lib?.datasets ?? []).map((d) => [d.id, d]));
  const objects = new Map((lib?.objects ?? []).map((o) => [o.id, o]));
  return (
    <Page title="Kits" intro="A kit is a list of datasets, objects and scenes you already have. Name it in the chat to start a new model from it, nothing is built twice." action={action}>
      {!lib ? <Loading error={error} /> : lib.kits.length === 0 ? (
        <Empty icon={<Package className="h-6 w-6" />} title="No kits yet" action={action}>
          Pick the photos, 3D objects and scenes you want to reuse and save them as a kit. Then tell the chat
          "using kit Shop, count the bottles".
        </Empty>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {lib.kits.map((k) => {
            const pics = [...k.objects.filter((id) => objects.get(id)?.thumb).map((id) => cardUrl("object", id)),
                          ...k.datasets.flatMap((id) => datasets.get(id)?.sample.slice(0, 1).map((p) => libraryPhotoUrl(p)) ?? [])].slice(0, 4);
            return (
              <Card key={k.id} className="overflow-hidden">
                <div className="grid aspect-[2/1] grid-cols-4 gap-px bg-line">
                  {[0, 1, 2, 3].map((i) => pics[i]
                    ? <img key={i} src={pics[i]} alt="" loading="lazy" className="h-full w-full bg-white object-cover" />
                    : <div key={i} className="bg-gray-100" />)}
                </div>
                <div className="p-4">
                  <div className="flex items-start justify-between gap-2">
                    <div className="truncate font-medium">{k.name}</div>
                    <ItemMenu kind="kits" id={k.id} name={k.name} onChanged={reload}
                              extra={<button className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-gray-50"
                                             onClick={() => setEdit(k)}><Pencil className="h-4 w-4" /> Edit contents</button>} />
                  </div>
                  <div className="text-sm text-muted">
                    {[[k.datasets.length, "dataset"], [k.objects.length, "object"], [k.scenes.length, "scene"]]
                      .map(([n, w]) => `${n} ${w}${n === 1 ? "" : "s"}`).join(" · ")}
                  </div>
                  <ButtonLink to={`/app?kit=${k.id}`} state={{ fresh: Date.now() }} variant="secondary" className="mt-3 h-10 w-full text-sm">
                    <Plus className="h-4 w-4" /> New model with this kit
                  </ButtonLink>
                </div>
              </Card>
            );
          })}
        </div>
      )}
      {edit && lib && <KitEditor kit={edit === "new" ? null : edit} lib={lib} onClose={() => setEdit(null)} onDone={reload} />}
    </Page>
  );
}

function KitEditor({ kit, lib, onClose, onDone }: { kit: Kit | null; lib: LibraryData; onClose: () => void; onDone: () => void }) {
  const [name, setName] = useState(kit?.name ?? "");
  const [refs, setRefs] = useState({ datasets: new Set(kit?.datasets), objects: new Set(kit?.objects), scenes: new Set(kit?.scenes) });
  const [tab, setTab] = useState<"datasets" | "objects" | "scenes">("datasets");
  const [error, setError] = useState<string | null>(null);
  const flip = (k: typeof tab, id: string) => setRefs((r) => {
    const n = new Set(r[k]);
    if (n.has(id)) n.delete(id); else n.add(id);
    return { ...r, [k]: n };
  });
  type Item = { id: string; name: string; pic: string | null; note: string };
  const objects = lib.objects.filter((o) => o.kept || refs.objects.has(o.id));
  const items: Record<typeof tab, Item[]> = {
    datasets: lib.datasets.map((d) => ({ id: d.id, name: d.name, pic: d.sample[0] ? libraryPhotoUrl(d.sample[0]) : null, note: `${d.photos} photos` })),
    objects: objects.map((o) => ({ id: o.id, name: o.name, pic: o.thumb ? cardUrl("object", o.id) : null, note: pretty(o.concept) })),
    scenes: lib.scenes.map((s) => ({ id: s.id, name: s.name, pic: s.thumb ? cardUrl("scene", s.id) : null, note: "" })),
  };
  const go = async () => {
    if (!name.trim()) { setError("Give the kit a name first."); return; }
    try {
      await saveKit(kit?.id ?? null, { name: name.trim(), datasets: [...refs.datasets], objects: [...refs.objects], scenes: [...refs.scenes] });
      onClose();
      onDone();
    } catch (err: any) {
      setError(err.message);
    }
  };
  return (
    <Dialog title={kit ? `Edit ${kit.name}` : "New kit"} onClose={onClose} wide>
      <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Kit name, e.g. Shop shelf" autoFocus className={field} aria-label="Kit name" />
      <div className="mt-4 flex gap-1 border-b border-line">
        {(["datasets", "objects", "scenes"] as const).map((k) => (
          <button key={k} onClick={() => setTab(k)}
                  className={cx("-mb-px border-b-2 px-3 py-2 text-sm font-medium capitalize",
                                tab === k ? "border-brand-600 text-brand-700" : "border-transparent text-muted hover:text-ink")}>
            {k} <span className="opacity-60">{refs[k].size}</span>
          </button>
        ))}
      </div>
      <div className="mt-3 grid max-h-[45dvh] grid-cols-2 gap-2 overflow-y-auto sm:grid-cols-3">
        {items[tab].length === 0 && <p className="col-span-full py-6 text-center text-sm text-muted">Nothing here yet.</p>}
        {items[tab].map((it) => (
          <div key={it.id} onClick={() => flip(tab, it.id)} role="checkbox" aria-checked={refs[tab].has(it.id)}
               className={cx("relative cursor-pointer overflow-hidden rounded-xl border text-left", refs[tab].has(it.id) ? "border-brand-500 ring-2 ring-brand-500" : "border-line")}>
            <Tick on={refs[tab].has(it.id)} onToggle={() => flip(tab, it.id)} />
            <div className="aspect-[4/3] bg-gray-100">{it.pic && <img src={it.pic} alt="" loading="lazy" className="h-full w-full object-cover" />}</div>
            <div className="p-2">
              <div className="truncate text-sm font-medium">{it.name}</div>
              <div className="truncate text-xs text-muted">{it.note}</div>
            </div>
          </div>
        ))}
      </div>
      {error && <div className="mt-4"><Notice tone="bad">{error}</Notice></div>}
      <div className="mt-5 flex justify-end gap-2">
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button onClick={go}>Save kit</Button>
      </div>
    </Dialog>
  );
}

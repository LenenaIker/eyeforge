import { Check, MoreHorizontal, Package, Pencil, Plus, Trash2, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { deleteItem, renameItem, saveKit, type Kit, type LibraryKind } from "../api";
import { Button, Notice, cx } from "./ui";

// Pieces the library pages share: a dialog, a card's menu (rename, delete), picking cards and putting them in a kit.

export function Dialog({ title, onClose, children, wide }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", esc);
    return () => document.removeEventListener("keydown", esc);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/40 sm:items-center sm:p-6" onClick={onClose}>
      <div className={cx("flex max-h-[90dvh] w-full flex-col rounded-t-2xl bg-white shadow-xl sm:rounded-[var(--radius-card)]",
                         wide ? "sm:max-w-2xl" : "sm:max-w-md")}
           onClick={(e) => e.stopPropagation()} role="dialog" aria-label={title}>
        <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
          <div className="font-semibold">{title}</div>
          <button onClick={onClose} aria-label="Close" className="rounded-lg p-1.5 text-muted hover:bg-gray-100"><X className="h-5 w-5" /></button>
        </div>
        <div className="overflow-y-auto p-5">{children}</div>
      </div>
    </div>
  );
}

export const field = "h-11 w-full rounded-[var(--radius-control)] border border-line px-3 outline-none focus:border-brand-500";

const SINGULAR: Record<LibraryKind, string> = { datasets: "dataset", objects: "object", scenes: "scene", kits: "kit" };

// The "..." of a card: rename it or delete it (asking first).
export function ItemMenu({ kind, id, name, onChanged, extra }:
  { kind: LibraryKind; id: string; name: string; onChanged: () => void; extra?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [dialog, setDialog] = useState<"rename" | "delete" | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);
  const item = "flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-gray-50";
  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen(!open)} aria-label="More"
              className="rounded-lg p-1.5 text-muted hover:bg-gray-100 hover:text-ink"><MoreHorizontal className="h-4 w-4" /></button>
      {open && (
        <div className="absolute right-0 z-10 mt-1 w-44 overflow-hidden rounded-xl border border-line bg-white py-1 shadow-lg">
          {extra}
          <button className={item} onClick={() => { setOpen(false); setDialog("rename"); }}><Pencil className="h-4 w-4" /> Rename</button>
          <button className={cx(item, "text-red-600")} onClick={() => { setOpen(false); setDialog("delete"); }}><Trash2 className="h-4 w-4" /> Delete</button>
        </div>
      )}
      {dialog === "rename" && <RenameDialog kind={kind} id={id} name={name} onClose={() => setDialog(null)} onDone={onChanged} />}
      {dialog === "delete" && <DeleteDialog kind={kind} items={[{ id, name }]} onClose={() => setDialog(null)} onDone={onChanged} />}
    </div>
  );
}

function RenameDialog({ kind, id, name, onClose, onDone }:
  { kind: LibraryKind; id: string; name: string; onClose: () => void; onDone: () => void }) {
  const [value, setValue] = useState(name);
  const [error, setError] = useState<string | null>(null);
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    renameItem(kind, id, value.trim()).then(() => { onClose(); onDone(); }).catch((err) => setError(err.message));
  };
  return (
    <Dialog title={`Rename ${SINGULAR[kind]}`} onClose={onClose}>
      <form onSubmit={submit} className="space-y-4">
        <input value={value} onChange={(e) => setValue(e.target.value)} autoFocus className={field} aria-label="Name" />
        {error && <Notice tone="bad">{error}</Notice>}
        <Button type="submit" className="w-full" disabled={!value.trim()}>Save</Button>
      </form>
    </Dialog>
  );
}

export function DeleteDialog({ kind, items, onClose, onDone }:
  { kind: LibraryKind; items: { id: string; name: string }[]; onClose: () => void; onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const what = items.length === 1 ? `"${items[0].name}"` : `${items.length} ${kind}`;
  const go = async () => {
    setBusy(true);
    try {
      for (const it of items) await deleteItem(kind, it.id);
      onClose();
      onDone();
    } catch (err: any) {
      setError(err.message);
      setBusy(false);
    }
  };
  return (
    <Dialog title={`Delete ${what}?`} onClose={onClose}>
      <p className="text-sm text-muted">
        {kind === "kits" ? "The kit goes, the datasets, objects and scenes in it stay."
          : "It is deleted for good and taken out of your kits. Models already trained with it keep working."}
      </p>
      {error && <div className="mt-4"><Notice tone="bad">{error}</Notice></div>}
      <div className="mt-5 flex justify-end gap-2">
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button variant="danger" onClick={go} disabled={busy}>{busy ? "Deleting..." : "Delete"}</Button>
      </div>
    </Dialog>
  );
}

// A pickable card's corner: the tick that selects it.
export function Tick({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button onClick={(e) => { e.preventDefault(); e.stopPropagation(); onToggle(); }} aria-label={on ? "Unselect" : "Select"}
            className={cx("absolute top-2 left-2 z-[1] flex h-6 w-6 items-center justify-center rounded-md border-2 shadow-sm transition-colors",
                          on ? "border-brand-600 bg-brand-600 text-white" : "border-white bg-white/80 text-transparent hover:border-brand-500")}>
      <Check className="h-4 w-4" strokeWidth={3} />
    </button>
  );
}

export function useSelection() {
  const [sel, setSel] = useState<Set<string>>(new Set());
  const toggle = (id: string) => setSel((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  return { sel, toggle, clear: () => setSel(new Set()) };
}

// What to do with the picked cards, pinned to the bottom while any is picked.
export function SelectBar({ kind, picked, kits, onClear, onDone }:
  { kind: "datasets" | "objects" | "scenes"; picked: { id: string; name: string }[]; kits: Kit[]; onClear: () => void; onDone: () => void }) {
  const [dialog, setDialog] = useState<"kit" | "delete" | null>(null);
  if (picked.length === 0) return null;
  return (
    <div className="sticky bottom-0 z-10 -mx-4 border-t border-line bg-white/95 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium">{picked.length} selected</span>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={onClear}>Clear</Button>
          <Button variant="danger" size="sm" onClick={() => setDialog("delete")}><Trash2 className="h-4 w-4" /> Delete</Button>
          <Button size="sm" onClick={() => setDialog("kit")}><Package className="h-4 w-4" /> Add to kit</Button>
        </div>
      </div>
      {dialog === "kit" && <AddToKit kind={kind} ids={picked.map((p) => p.id)} kits={kits} onClose={() => setDialog(null)}
                                     onDone={() => { onClear(); onDone(); }} />}
      {dialog === "delete" && <DeleteDialog kind={kind} items={picked} onClose={() => setDialog(null)}
                                            onDone={() => { onClear(); onDone(); }} />}
    </div>
  );
}

function AddToKit({ kind, ids, kits, onClose, onDone }:
  { kind: "datasets" | "objects" | "scenes"; ids: string[]; kits: Kit[]; onClose: () => void; onDone: () => void }) {
  const [target, setTarget] = useState<string>(kits[0]?.id ?? "new");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const go = async () => {
    try {
      const kit = kits.find((k) => k.id === target);
      const base = kit ? { name: kit.name, datasets: kit.datasets, objects: kit.objects, scenes: kit.scenes }
        : { name: name.trim(), datasets: [], objects: [], scenes: [] };
      await saveKit(kit ? kit.id : null, { ...base, [kind]: [...new Set([...base[kind], ...ids])] });
      onClose();
      onDone();
    } catch (err: any) {
      setError(err.message);
    }
  };
  const row = "flex cursor-pointer items-center gap-3 rounded-xl border px-4 py-3 text-sm";
  return (
    <Dialog title={`Add ${ids.length} to a kit`} onClose={onClose}>
      <div className="space-y-2">
        {kits.map((k) => (
          <label key={k.id} className={cx(row, target === k.id ? "border-brand-500 bg-brand-50" : "border-line")}>
            <input type="radio" checked={target === k.id} onChange={() => setTarget(k.id)} className="accent-brand-600" />
            <span className="flex-1 font-medium">{k.name}</span>
            <span className="text-muted">{k.datasets.length + k.objects.length + k.scenes.length} items</span>
          </label>
        ))}
        <label className={cx(row, target === "new" ? "border-brand-500 bg-brand-50" : "border-line")}>
          <input type="radio" checked={target === "new"} onChange={() => setTarget("new")} className="accent-brand-600" />
          <Plus className="h-4 w-4" />
          <input value={name} onChange={(e) => { setName(e.target.value); setTarget("new"); }} placeholder="New kit name"
                 className="flex-1 bg-transparent outline-none" aria-label="New kit name" />
        </label>
      </div>
      {error && <div className="mt-4"><Notice tone="bad">{error}</Notice></div>}
      <Button className="mt-5 w-full" onClick={go} disabled={target === "new" && !name.trim()}>Add</Button>
    </Dialog>
  );
}

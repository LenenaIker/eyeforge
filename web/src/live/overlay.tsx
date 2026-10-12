// What the live tests draw over a picture, shared by the phone's camera (pages/Live.tsx) and the ESP32-CAM's
// frames over USB (pages/LiveBoard.tsx): the dots of a count model and the check of a photo just taken.
import { Check } from "lucide-react";
import { useState, type ReactNode } from "react";
import { cx } from "../components/ui";
import type { Said } from "./captures";
import { fit } from "./orient";

/** What the model said of one picture: a count model's objects (0-1 of the picture), or a class with its score. */
export type Guess = { points: [number, number, string][]; top: { label: string; p: number } | null };
/** The report fields the checks need (api.ts Report has them). */
export type LabelsOf = { task: string; labels: string[]; background?: string | null };

/** The objects a count model found, as dots on the picture they sit in (a positioned parent). */
export function Dots({ points, still }: { points: [number, number, ...unknown[]][]; still?: boolean }) {
  return (
    <>
      {points.map((p, i) => (
        <span key={i} className={cx("absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-white bg-ember-500/80 shadow-lg",
                                    still && "pointer-events-none")}
              style={{ left: `${p[0] * 100}%`, top: `${p[1] * 100}%` }} />
      ))}
    </>
  );
}

// The photo just taken, frozen, with what the model said: the user fixes it (a class, or the objects' points)
// before it is kept. Keeping it as it is takes one tap when the model was right.
export function CheckPhoto({ c, report, view, onKeep, onDiscard }: {
  c: { url: string; w: number; h: number; guess: Guess }; report: LabelsOf; view: { w: number; h: number };
  onKeep: (said: Said | undefined) => void; onDiscard: () => void;
}) {
  const counting = report.task === "count";
  const bg = "background" in report ? report.background : report.labels[0];
  const classes = counting ? report.labels.filter((l) => l !== bg) : report.labels;
  const [label, setLabel] = useState(c.guess.top?.label ?? classes[0]);
  const [points, setPoints] = useState<[number, number, string][]>(c.guess.points.map(([x, y, l]) => [x, y, l]));
  const [cls, setCls] = useState(classes[0]);
  const panel = counting ? (classes.length > 1 ? 190 : 140) : 120 + 48 * Math.ceil(classes.length / 2);
  const box = fit(c.w, c.h, view.w, Math.max(160, view.h - panel), "contain");
  const nice = (l: string) => l.replace(/_/g, " ");

  const tap = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!counting) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    const near = points.findIndex(([px, py]) => Math.hypot((px - x) * r.width, (py - y) * r.height) < 22);
    setPoints(near >= 0 ? points.filter((_, i) => i !== near)
                        : [...points, [Math.round(x * 1e4) / 1e4, Math.round(y * 1e4) / 1e4, cls]]);
  };

  return (
    <div className="absolute inset-0 z-50 flex flex-col bg-black touch-auto">
      <div className="relative" style={{ height: Math.max(160, view.h - panel) }}>
        <div className="absolute" style={box} onClick={tap}>
          <img src={c.url} alt="The photo just taken" className="block h-full w-full" />
          {counting && <Dots points={points} still />}
        </div>
      </div>
      <div className="flex-1 space-y-3 px-4 pt-3 pb-[max(1rem,env(safe-area-inset-bottom))]">
        {counting ? (
          <>
            <p className="text-sm text-white/80">
              <b className="text-white">{points.length}</b> marked. Tap to add one, tap a mark to remove it.
            </p>
            {classes.length > 1 && (
              <div className="flex flex-wrap gap-2">
                {classes.map((l) => <Chip key={l} on={cls === l} onClick={() => setCls(l)}>{nice(l)}</Chip>)}
              </div>
            )}
          </>
        ) : (
          <>
            <p className="text-sm text-white/80">
              The model said <b className="text-white">{nice(c.guess.top?.label ?? "?")}</b>. What is it?
            </p>
            <div className="grid grid-cols-2 gap-2">
              {classes.map((l) => <Chip key={l} on={label === l} onClick={() => setLabel(l)}>{nice(l)}</Chip>)}
            </div>
          </>
        )}
        <div className="flex gap-3">
          <button onClick={onDiscard} className="h-11 flex-1 rounded-full bg-white/15 font-medium">Discard</button>
          <button onClick={() => onKeep(counting ? { points } : { label })}
                  className="flex h-11 flex-1 items-center justify-center gap-2 rounded-full bg-white font-medium text-ink">
            <Check className="h-4 w-4" /> Keep
          </button>
        </div>
      </div>
    </div>
  );
}

export function Chip({ on, onClick, children }: { on: boolean; onClick: () => void; children: ReactNode }) {
  return <button onClick={onClick} className={cx("h-10 truncate rounded-full px-4 text-sm font-medium",
                                                 on ? "bg-ember-500 text-white" : "bg-white/15 text-white")}>{children}</button>;
}

export function Centered({ children }: { children: ReactNode }) {
  return <div className="absolute inset-0 grid place-items-center">{children}</div>;
}

export function RoundButton({ label, on, onClick, style, children }: {
  label: string; on?: boolean; onClick: () => void; style?: React.CSSProperties; children: ReactNode;
}) {
  return <button onClick={onClick} aria-label={label} style={style}
                 className={cx("grid h-12 w-12 place-items-center rounded-full backdrop-blur", on ? "bg-ember-500" : "bg-black/35")}>{children}</button>;
}

// "Try it live": the model on this phone's camera, or on the ESP32-CAM plugged in over USB. A split button: the
// arrow picks the camera, the button runs the picked one; the pick is remembered on this device. Only the
// cameras this browser can use are offered (an iPhone has no USB access: the phone only).
import { Camera, Check, ChevronDown, Cpu } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FlashError, handPort, pickPort } from "../flash/flasher";
import { enterCameraMode } from "../live/device";
import { cx } from "./ui";

export type TestCamera = "phone" | "board";
const KEY = "eyeforge.testCamera";
const NAMES: Record<TestCamera, string> = { phone: "This phone", board: "ESP32-CAM (USB)" };

function remembered(): TestCamera | null {
  try { const v = localStorage.getItem(KEY); return v === "phone" || v === "board" ? v : null; } catch { return null; }
}
function remember(v: TestCamera) {
  try { localStorage.setItem(KEY, v); } catch { /* storage blocked: the pick lasts this visit */ }
}

export function TestButton({ job, phone, board }: { job: string; phone: boolean; board: boolean }) {
  const navigate = useNavigate();
  const options = ([phone && "phone", board && "board"].filter(Boolean)) as TestCamera[];
  const [pick, setPick] = useState<TestCamera | null>(() => remembered());
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const menu = useRef<HTMLDivElement>(null);
  const chosen = pick && options.includes(pick) ? pick : options[0];

  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => { if (!menu.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [open]);

  if (!chosen) return null;

  // The board's device picker only opens straight from a tap, so it opens here and the port goes to the page.
  const run = async () => {
    setError(null);
    if (chosen === "phone") {
      enterCameraMode();
      navigate(`/app/job/${job}/live`);
      return;
    }
    try {
      handPort(await pickPort());
    } catch (e) {
      if (e instanceof FlashError && /No board was chosen/.test(e.message)) return;
      setError((e as Error).message);
      return;
    }
    navigate(`/app/job/${job}/board`);
  };

  const choose = (v: TestCamera) => { setPick(v); remember(v); setOpen(false); };
  const Icon = chosen === "phone" ? Camera : Cpu;
  const base = "flex h-14 items-center justify-center gap-2 bg-ink text-base font-semibold text-white shadow-sm";

  return (
    <div className="mt-4">
      <div ref={menu} className="relative flex">
        <button onClick={run} className={cx(base, "flex-1 rounded-[var(--radius-control)]", options.length > 1 && "rounded-r-none")}>
          <Icon className="h-5 w-5" /> {chosen === "phone" ? "Try it live with your camera" : "Try it live on the ESP32-CAM"}
        </button>
        {options.length > 1 && (
          <button onClick={() => setOpen(!open)} aria-label="Pick the camera" aria-expanded={open}
                  className={cx(base, "w-12 rounded-l-none rounded-r-[var(--radius-control)] border-l border-white/20")}>
            <ChevronDown className="h-5 w-5" />
          </button>
        )}
        {open && (
          <div role="menu" className="absolute top-full right-0 z-30 mt-1 w-56 overflow-hidden rounded-xl border border-line bg-white py-1 text-sm shadow-lg">
            {options.map((v) => (
              <button key={v} role="menuitemradio" aria-checked={v === chosen} onClick={() => choose(v)}
                      className="flex w-full items-center gap-2 px-3 py-2.5 text-left hover:bg-gray-50">
                <Check className={cx("h-4 w-4", v === chosen ? "text-brand-600" : "invisible")} /> {NAMES[v]}
              </button>
            ))}
          </div>
        )}
      </div>
      {error && <p className="mt-2 text-sm text-red-700">{error}</p>}
    </div>
  );
}

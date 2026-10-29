import { Camera, Cpu, Image, Map, Target } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import * as API from "../api";
import type { Spec } from "../api";

// Plain-language names for what the spec says; the JSON stays under "Advanced".
export const DEVICES: Record<string, string> = {
  esp32cam: "ESP32-CAM (microcontroller)",
  esp32s3: "ESP32-S3 camera board",
  jetson_nano: "NVIDIA Jetson Nano",
  generic: "Phone or computer (no specific chip yet)",
};
const TASKS: Record<string, string> = { count: "Count", roi_classification: "Recognize", classify: "Recognize" };

// The library's scene names (the user never sees ids), read once per page.
let sceneNames: Promise<Record<string, string>> | null = null;
function useSceneNames(): Record<string, string> {
  const [names, setNames] = useState<Record<string, string>>({});
  useEffect(() => {
    sceneNames ??= API.getLibrary().then((l) => Object.fromEntries(l.scenes.map((s) => [s.id, s.name])), () => ({}));
    sceneNames.then(setNames);
  }, []);
  return names;
}

export function SpecSummary({ spec }: { spec: Spec }) {
  const scenes: string[] = spec.generation?.scenes ?? [];
  const names = useSceneNames();
  const ud = spec.user_data ?? {};
  const cam = spec.camera ?? {};
  return (
    <div className="p-4">
      <p className="text-xs font-semibold tracking-wide text-brand-600 uppercase">Your model</p>
      <h3 className="mt-1 text-lg font-semibold tracking-tight">{spec.name.replace(/_/g, " ")}</h3>
      <dl className="mt-4 space-y-3 text-sm">
        <Row icon={<Target />} label={TASKS[spec.task ?? ""] ?? "Detect"}>
          {spec.classes.map((c) => (
            <span key={c.id} className="block"><b className="font-medium">{c.id.replace(/_/g, " ")}</b>
              {c.description && <span className="text-muted">: {c.description}</span>}</span>
          ))}
        </Row>
        <Row icon={<Map />} label="Scenes">
          {scenes.length ? scenes.map((id) => names[id] ?? "a scene of your library").join(", ") : "none"}
        </Row>
        <Row icon={<Camera />} label="Camera">
          {cam.distance_m ? `${cam.distance_m[0]} to ${cam.distance_m[1]} m away` : "as in your photos"}
          {cam.height_m && <span className="text-muted">, {cam.height_m[0]} to {cam.height_m[1]} m high</span>}
        </Row>
        <Row icon={<Image />} label="Training images">
          {usesSynthetic(spec) ? <>
            {(spec.generation?.frames ?? 2000).toLocaleString()} made to measure
            {hasPhotos(spec) && <span className="text-muted"> + your photos{ud.raw ? " (labelled for you)" : ""}</span>}
          </> : <>Your photos only{ud.raw && <span className="text-muted"> (labelled for you)</span>}</>}
        </Row>
        <Row icon={<Cpu />} label="Device">{DEVICES[spec.hardware ?? "generic"] ?? spec.hardware}</Row>
      </dl>
    </div>
  );
}

// Synthetic images are made in the walk scenes the spec names, when frames are asked for; frames 0 or no scene
// trains on the user's photos alone and skips the rendering (spec.synthetic).
export const usesSynthetic = (spec: Spec) =>
  (spec.generation?.frames ?? 2000) > 0 && (spec.generation?.scenes ?? []).length > 0;
export const hasPhotos = (spec: Spec) => {
  const ud = spec.user_data ?? {};
  return Boolean(ud.raw || ud.test || (ud.labeled && Object.keys(ud.labeled).length));
};

function Row({ icon, label, children }: { icon: ReactNode; label: string; children: ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="mt-0.5 h-4 w-4 shrink-0 text-muted [&>svg]:h-4 [&>svg]:w-4">{icon}</span>
      <div className="flex min-w-0 flex-1 flex-col sm:flex-row sm:gap-3">
        <dt className="shrink-0 text-muted sm:w-32">{label}</dt>
        <dd className="min-w-0 flex-1">{children}</dd>
      </div>
    </div>
  );
}

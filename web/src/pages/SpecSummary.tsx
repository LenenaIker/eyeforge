import { Camera, Cpu, Image, Map, Target } from "lucide-react";
import type { ReactNode } from "react";
import type { Spec } from "../api";

// Plain-language names for what the spec says; the JSON stays under "Advanced".
const DEVICES: Record<string, string> = {
  esp32cam: "ESP32-CAM (about $10 microcontroller)",
  esp32s3: "ESP32-S3 camera board",
  jetson_nano: "NVIDIA Jetson Nano",
  generic: "Phone or computer (no specific chip yet)",
};
const TASKS: Record<string, string> = { count: "Count", roi_classification: "Recognize", classify: "Recognize" };
const PLACES: Record<string, string> = {
  facade: "building fronts", balcony: "balconies", indoor: "indoors", road: "roads", ground: "open ground",
};

export function SpecSummary({ spec }: { spec: Spec }) {
  const scene = spec.scene ?? {};
  const places = (scene.contexts ?? []).map((c: any) => `${PLACES[c.kind] ?? c.kind} ${Math.round(c.share * 100)}%`);
  const where = places.length ? places.join(", ") : scene.template === "road" ? "a road" : "outdoors";
  const materials = [...(scene.wall_tags ?? []), ...(scene.ground_tags ?? [])];
  const props = (spec.props ?? []).map((p: any) => p.id.replace(/_/g, " "));
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
          {where}{materials.length > 0 && <span className="text-muted">; {materials.slice(0, 5).join(", ")}</span>}
          {props.length > 0 && <span className="block text-muted">with {props.join(", ")} around</span>}
        </Row>
        <Row icon={<Camera />} label="Camera">
          {cam.distance_m ? `${cam.distance_m[0]} to ${cam.distance_m[1]} m away` : "as in your photos"}
          {cam.height_m && <span className="text-muted">, {cam.height_m[0]} to {cam.height_m[1]} m high</span>}
        </Row>
        <Row icon={<Image />} label="Training images">
          {(spec.generation?.frames ?? 0).toLocaleString()} made to measure
          {(ud.raw || ud.labeled) && <span className="text-muted"> + your photos{ud.raw ? " (labelled for you)" : ""}</span>}
        </Row>
        <Row icon={<Cpu />} label="Device">{DEVICES[spec.hardware ?? "generic"] ?? spec.hardware}</Row>
      </dl>
    </div>
  );
}

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

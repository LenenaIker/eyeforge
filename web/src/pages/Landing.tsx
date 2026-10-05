import { ArrowRight, Box, Camera, Code, Cpu, MessageSquare, Sparkles } from "lucide-react";
import type { ReactNode } from "react";
import { Badge, ButtonLink, Card, Logo } from "../components/ui";

// The public repository, set at build time (VITE_REPO_URL); the link is hidden until there is one.
const REPO = import.meta.env.VITE_REPO_URL as string | undefined;

export default function Landing() {
  return (
    <div className="min-h-dvh">
      <header className="sticky top-0 z-10 border-b border-line/60 bg-canvas/85 backdrop-blur">
        <nav className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
          <Logo />
          <div className="flex items-center gap-1 sm:gap-2">
            <a href="#how" className="hidden rounded-lg px-3 py-2 text-sm text-muted hover:text-ink sm:block">How it works</a>
            {REPO && (
              <a href={REPO} className="rounded-lg p-2 text-muted hover:text-ink" aria-label="Source code">
                <Code className="h-5 w-5" strokeWidth={1.75} />
              </a>
            )}
            <ButtonLink to="/app" className="h-10 px-4 text-sm">Try it</ButtonLink>
          </div>
        </nav>
      </header>

      <main>
        <Hero />
        <How />
        <Why />
        <BuiltWith />
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-8 text-sm text-muted sm:flex-row sm:justify-between sm:px-6">
          <span>EyeForge, open source under the AGPL-3.0 license.</span>
          <span>Built for the Nebius x NVIDIA Global AI Hackathon.</span>
        </div>
      </footer>
    </div>
  );
}

function Hero() {
  return (
    <section className="mx-auto grid max-w-6xl items-start gap-12 px-4 pt-6 pb-20 sm:px-6 lg:grid-cols-2 lg:pt-8">
      <div>
        <Badge tone="brand" className="mb-5"><Sparkles className="h-3.5 w-3.5" /> On-device vision, no cloud needed</Badge>
        <h1 className="text-4xl font-bold tracking-tight text-balance sm:text-5xl lg:text-6xl">
          Describe what your camera should see.
          <span className="block text-brand-600">Get a tiny model for a $10 chip.</span>
        </h1>
        <p className="mt-6 max-w-xl text-lg text-muted text-pretty">
          Tell EyeForge what to spot or count and add a few phone photos. It builds thousands of made-to-measure
          training images, trains a specialist and hands you a model small enough to run on the device itself.
        </p>
        <div className="mt-8 flex flex-col gap-3 sm:flex-row">
          <ButtonLink to="/app">Build a model <ArrowRight className="h-4 w-4" /></ButtonLink>
          <a href="#how" className="inline-flex h-12 items-center justify-center rounded-[var(--radius-control)] px-6 font-medium text-muted hover:text-ink">
            See how it works
          </a>
        </div>
      </div>
      <Proof />
    </section>
  );
}

// Measured on Nebius with the account the web uses (2026-10-04, worker kits): kits_photos_4 (69 lawn photos,
// no synthetic data, 5.0 min, $0.13) and kits_synth (smoke_farm_animals for the ESP32-CAM, 1,500 renders,
// 11.0 min, $0.28). Keep these real: measure again when the images or the pipeline change.
function Proof() {
  const stats: [string, ReactNode][] = [["~5 min", "photos to model, with your photos only"],
                 ["~11 min", "adding 1,500 synthetic images from Isaac Sim"],
                 ["from $0.13", "of cloud GPU per run on Nebius AI Cloud"],
                 ["612 KB", <>int8 model for an <Esp32Cam />, no network</>]];
  return (
    <div className="relative mx-auto w-full max-w-md">
      <div className="absolute -inset-6 -z-10 rounded-[32px] bg-gradient-to-br from-brand-100 via-white to-ember-100 blur-2xl" />
      <p className="mb-3 text-sm font-semibold text-brand-600">Measured, not promised: two runs, start to finish</p>
      <div className="grid grid-cols-2 gap-3">
        {stats.map(([v, l]) => (
          <Card key={v} className="p-5 shadow-lg">
            <p className="text-2xl font-bold tracking-tight sm:text-3xl">{v}</p>
            <p className="mt-1 text-sm text-muted">{l}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}

// The board's name: its price on hover (stated here, not in the text, where a price reads like a promise), and
// a click opens a shop that sells it.
const ESP32_CAM_SHOP = "https://www.aliexpress.com/w/wholesale-esp32-cam.html";
function Esp32Cam() {
  return (
    <a href={ESP32_CAM_SHOP} target="_blank" rel="noopener noreferrer"
       className="group relative whitespace-nowrap underline decoration-dotted underline-offset-2 hover:text-ink">
      ESP32-CAM
      <span className="pointer-events-none absolute top-full left-1/2 z-10 mt-1.5 -translate-x-1/2 whitespace-nowrap rounded-md bg-ink px-2 py-1 text-xs text-white opacity-0 shadow transition-opacity group-hover:opacity-100">
        About $10, camera included
      </span>
    </a>
  );
}

function Section({ id, eyebrow, title, children }: { id?: string; eyebrow: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="mx-auto max-w-6xl scroll-mt-20 px-4 py-16 sm:px-6">
      <p className="text-sm font-semibold text-brand-600">{eyebrow}</p>
      <h2 className="mt-2 max-w-2xl text-3xl font-bold tracking-tight text-balance">{title}</h2>
      <div className="mt-10">{children}</div>
    </section>
  );
}

function How() {
  const steps = [
    { icon: MessageSquare, title: "Describe", text: "Chat with the assistant (NVIDIA Nemotron) about what to detect or count, and add a few photos from your phone. No vision expertise needed: it looks at your photos and works out sizes, scenes and look-alikes itself." },
    { icon: Box, title: "Generate", text: "EyeForge finds 3D models, builds scenes like yours in NVIDIA Isaac Sim and renders thousands of labelled images, all on a GPU in Nebius AI Cloud that shuts down when done." },
    { icon: Cpu, title: "Deploy", text: "A small int8 model trained on those images and your photos, scored on real pictures, ready for an ESP32-CAM, a Raspberry Pi, a Jetson or your own computer." },
  ];
  return (
    <Section id="how" eyebrow="How it works" title="From a sentence to a model on your device">
      <div className="grid gap-4 md:grid-cols-3">
        {steps.map(({ icon: Icon, title, text }, i) => (
          <Card key={title} className="p-6">
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-brand-50 text-brand-600">
                <Icon className="h-5 w-5" strokeWidth={1.75} />
              </div>
              <span className="text-sm font-medium text-muted">Step {i + 1}</span>
            </div>
            <h3 className="mt-4 text-lg font-semibold">{title}</h3>
            <p className="mt-2 text-muted">{text}</p>
          </Card>
        ))}
      </div>
    </Section>
  );
}

function Why() {
  const devices = [
    ["ESP32-CAM", "96 px, a few coarse classes", "about $10"],
    ["Raspberry Pi", "detect and count", "about $60"],
    ["NVIDIA Jetson", "fine distinctions", "from $250"],
    ["Your GPU", "the biggest student", "what you have"],
  ];
  return (
    <Section eyebrow="Why a specialist" title="Big models know a bit of everything. Your camera needs one thing, done well.">
      <div className="grid gap-8 lg:grid-cols-2">
        <p className="text-lg text-muted text-pretty">
          Foundation models can read almost any photo, but they need a data center and still miss fine details.
          EyeForge uses them as teachers: they label your photos and check the 3D models, and a student sized for
          your hardware learns from them. The answer stays on the device: no images leave it, no subscription, no
          network needed.
        </p>
        <div className="grid grid-cols-2 gap-3">
          {devices.map(([name, what, price]) => (
            <Card key={name} className="p-4">
              <p className="font-semibold">{name}</p>
              <p className="mt-1 text-sm text-muted">{what}</p>
              <p className="mt-3 text-xs font-medium text-ember-700">{price}</p>
            </Card>
          ))}
        </div>
      </div>
    </Section>
  );
}

function BuiltWith() {
  const parts = [["NVIDIA Nemotron", "interview and reasoning"], ["NVIDIA Isaac Sim", "synthetic images"],
                 ["Nebius AI Cloud", "L40S GPU, pay per run"], ["Nebius Token Factory", "open models as teachers"]];
  return (
    <section className="border-y border-line bg-white">
      <div className="mx-auto grid max-w-6xl grid-cols-2 gap-6 px-4 py-10 sm:px-6 lg:grid-cols-4">
        {parts.map(([name, what]) => (
          <div key={name}>
            <p className="font-semibold">{name}</p>
            <p className="text-sm text-muted">{what}</p>
          </div>
        ))}
      </div>
      <div className="mx-auto max-w-6xl px-4 pb-10 sm:px-6">
        <ButtonLink to="/app" className="w-full sm:w-auto"><Camera className="h-4 w-4" /> Build your first model</ButtonLink>
      </div>
    </section>
  );
}

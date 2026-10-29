import type { CSSProperties, ReactNode } from "react";
import { Link } from "react-router-dom";
import "./landing.css";

// The landing, proposal D (user, 2026-10-08): the light structure of nebius.com with EyeForge's own colors. The
// wording is the user's: keep its order and lines. Styles in landing.css, scoped to .lp.

// The public repository, set at build time (VITE_REPO_URL); its links are hidden until there is one.
const REPO = import.meta.env.VITE_REPO_URL as string | undefined;

export default function Landing() {
  return (
    <div className="lp">
      <header><div className="wrap"><nav>
        <Link className="logo" to="/"><LogoMark /> EyeForge</Link>
        <div className="links">
          <a href="#how">How it works</a><a href="#numbers">Real numbers</a><a href="#devices">Devices</a>
          {REPO && <a href={REPO}>GitHub</a>}
        </div>
        <Link className="btn btn-p btn-sm" to="/app">Try it</Link>
      </nav></div></header>

      <div className="wrap hero">
        <h1>Describe it. <span>Forge it.</span> Spot it.</h1>
        <p className="lead">Add a few photos and EyeForge trains a model that runs on your device, ready to flash.</p>
        <div className="ctas"><Link className="btn btn-p" to="/app">Build a model</Link><a className="btn btn-s" href="#how">See how it works</a></div>
        <Shelf />
      </div>

      <section>
        <div className="wrap">
          <span className="eyebrow">Why EyeForge</span>
          <h2>Vision on your own device, without being an expert</h2>
          <div className="four">
            <Promise icon={<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />} title="Plain words">
              Say what you want to spot or count. No labels, no code, no jargon.
            </Promise>
            <Promise icon={<><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3z" /><circle cx="12" cy="13" r="3" /></>} title="A few photos">
              Raw phone photos are enough. Big models label them for you.
            </Promise>
            <Promise icon={<><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" /><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" /></>} title="Sized for your device">
              From a tiny chip to your own GPU. The model fits the hardware.
            </Promise>
            <Promise icon={<><rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" /></>} title="Stays private">
              The answer is computed on the camera. No images leave it.
            </Promise>
          </div>
        </div>
      </section>

      <section className="band" id="numbers">
        <div className="wrap">
          <span className="eyebrow">Measured, not promised</span>
          <h2>Two real runs, start to finish</h2>
          <div className="stats">
            <div className="st"><b>~5 min</b><span>from photos to model, with your photos only</span></div>
            <div className="st"><b>~11 min</b><span>adding 1,500 synthetic images</span></div>
            <div className="st amber"><b>$0.13</b><span>of cloud GPU per run, from</span></div>
            <div className="st"><b>612 KB</b><span>int8 model for an ESP32-CAM, no network</span></div>
          </div>
          <p className="note">Measured on 2026-10-04 on Nebius AI Cloud.</p>
        </div>
      </section>

      <section id="how">
        <div className="wrap">
          <span className="eyebrow">How it works</span>
          <h2>From a sentence to a model on your device</h2>
          <div className="rows">
            <div className="row">
              <div>
                <span className="step"><b>1</b> Describe</span>
                <h3>Talk to it like you would to a friend who knows cameras</h3>
                <p>The assistant asks what matters: what to look for, where the camera hangs, the light, the look-alikes. It looks at your photos and works out sizes and scenes itself.</p>
                <p className="pw">Interview by <b>NVIDIA Nemotron</b></p>
              </div>
              <div className="shot">
                <div className="bub bot">What should the camera count, and where will it hang?</div>
                <div className="bub me">Bottles on the shelf of my shop, so I know when to restock. The camera faces the shelf.</div>
                <div className="bub bot">Nice. Add 10 to 20 photos of that shelf, some full and some half empty.</div>
                <span className="typing"><i /><i /><i /></span>
              </div>
            </div>
            <div className="row rev">
              <div>
                <span className="step"><b>2</b> Generate</span>
                <h3>Thousands of training images, made to measure</h3>
                <p>EyeForge finds 3D models of your object, builds scenes like yours and renders them with labels. Your own photos join them. A cloud GPU does it and switches off when done.</p>
                <p className="pw">Rendered with <b>NVIDIA Isaac Sim</b>, runs on Nebius AI Cloud</p>
              </div>
              <div className="shot">
                <div className="gallery">{Array.from({ length: 8 }, (_, i) => <i key={i} />)}</div>
                <div className="glabel"><span>1,120 of 1,500 images</span><span>about 4 min left</span></div>
              </div>
            </div>
            <div className="row">
              <div>
                <span className="step"><b>3</b> Deploy</span>
                <h3>A tiny model, tested on your real photos</h3>
                <p>You get a score in plain words, the photos it got right and wrong, and a file ready for your board. Spot a mistake? Fix it and retrain.</p>
                <p className="pw">int8, for <b>ESP32-CAM, Raspberry Pi, Jetson</b> or your computer</p>
              </div>
              <div className="shot">
                <div className="chipcard">
                  <div className="dl"><span>Model for ESP32-CAM<br /><small>int8, 612 KB</small></span><span className="btn btn-p btn-sm">Download</span></div>
                  <div className="dl"><span>C array for Arduino<br /><small>ready to paste</small></span><span className="btn btn-s btn-sm">Download</span></div>
                  <div className="dl"><span>Report<br /><small>score, examples, settings</small></span><span className="btn btn-s btn-sm">Open</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="band" id="devices">
        <div className="wrap">
          <span className="eyebrow">Why a specialist</span>
          <h2>Big models know a bit of everything. Your camera needs one thing, done well.</h2>
          <p className="sub">Foundation models are the teachers. A student sized for your hardware learns from them.</p>
          <div className="devs">
            <div className="d"><h3>ESP32-CAM</h3><p>96 px, a few coarse classes</p></div>
            <div className="d"><h3>Raspberry Pi</h3><p>Detect and count</p></div>
            <div className="d"><h3>NVIDIA Jetson</h3><p>Fine distinctions</p></div>
            <div className="d"><h3>Your GPU</h3><p>The biggest student</p></div>
          </div>
          <div className="built">
            <div><b>NVIDIA Nemotron</b><span>interview and reasoning</span></div>
            <div><b>NVIDIA Isaac Sim</b><span>synthetic images</span></div>
            <div><b>Nebius AI Cloud</b><span>L40S GPU, pay per run</span></div>
            <div><b>Nebius Token Factory</b><span>open models as teachers</span></div>
          </div>
        </div>
      </section>

      <div className="wrap cta">
        <h2>Build your first model</h2>
        <p className="sub" style={{ marginLeft: "auto", marginRight: "auto" }}>One sentence and a few photos. About the price of a coffee.</p>
        <div className="ctas">
          <Link className="btn btn-p" to="/app">Start now</Link>
          {REPO && <a className="btn btn-s" href={REPO}>Read the code</a>}
        </div>
      </div>

      <footer><div className="wrap">
        <span className="logo">EyeForge</span>
        <span>Open source under the Apache-2.0 license. Built for the Nebius x NVIDIA Global AI Hackathon.</span>
        <span className="legal"><Link to="/privacy">Privacy</Link> <Link to="/terms">Terms</Link> <Link to="/support">Support</Link></span>
      </div></footer>
    </div>
  );
}

function LogoMark() {
  return (
    <svg viewBox="0 0 64 64" fill="none" aria-hidden="true">
      <path d="M4 32C11 19 21 12 32 12s21 7 28 20c-7 13-17 20-28 20S11 45 4 32Z" fill="#efedff" stroke="#614efa" strokeWidth="4" strokeLinejoin="round" />
      <g stroke="#b45309" strokeWidth="2.5" strokeLinecap="round"><path d="M27 20v4M32 20v4M37 20v4M27 40v4M32 40v4M37 40v4M20 27h4M20 32h4M20 37h4M40 27h4M40 32h4M40 37h4" /></g>
      <rect x="23" y="23" width="18" height="18" rx="3.5" fill="#f59e0b" /><circle cx="32" cy="32" r="4" fill="#1c1917" />
    </svg>
  );
}

function Promise({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <div className="f">
      <div className="ic"><svg viewBox="0 0 24 24" fill="none" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">{icon}</svg></div>
      <h3>{title}</h3><p>{children}</p>
    </div>
  );
}

// A shop shelf drawn in CSS: products on three boards, the bottles boxed by the model. [kind, left %, color, label]
type Item = [string, number, string, string?];
const BOARDS: [number, Item[]][] = [
  [70, [["pack", 4, "#c75b4a"], ["pack", 13, "#e2b04a"], ["bottle", 26, "#4f8a5b", "bottle 0.94"], ["bottle", 34, "#b86a2c", "bottle 0.91"],
        ["pack", 46, "#5b7fb5"], ["can", 58, "#c94a4a"], ["can", 63, "#c94a4a"], ["pack", 73, "#7a9a5a"], ["pack", 84, "#e2b04a"]]],
  [38, [["bottle", 6, "#7fa7c4", "bottle 0.90"], ["bottle", 14, "#4f8a5b", "bottle 0.88"], ["can", 26, "#9aa3ad"], ["can", 31, "#9aa3ad"],
        ["pack", 42, "#c75b4a"], ["bottle", 64, "#b86a2c", "bottle 0.86"], ["pack", 76, "#5b7fb5"], ["can", 88, "#c94a4a"]]],
  [6, [["pack", 4, "#7a9a5a"], ["pack", 13, "#5b7fb5"], ["pack", 22, "#e2b04a"], ["can", 36, "#c94a4a"], ["can", 41, "#c94a4a"],
       ["can", 46, "#9aa3ad"], ["pack", 58, "#c75b4a"], ["pack", 67, "#c75b4a"], ["pack", 80, "#7a9a5a"]]],
];

function Shelf() {
  return (
    <div className="visual" role="img" aria-label="A camera counting bottles on a shop shelf">
      <div className="shelf">
        {[30, 62, 94].map((top) => <div key={top} className="board" style={{ top: `${top}%` }} />)}
        {BOARDS.flatMap(([bottom, items]) => items.map(([kind, left, color, label]) => {
          const style: CSSProperties = { left: `${left}%`, bottom: `${bottom}%`, background: color };
          return <div key={`${bottom}-${left}`} className={`item ${kind}${label ? " hit" : ""}`} style={style}>{label && <b>{label}</b>}</div>;
        }))}
      </div>
      <div className="hud">
        <div className="cnt"><small>Counted on the chip</small><strong>5 bottles</strong></div>
        <div className="dev"><i /> ESP32-CAM · offline</div>
      </div>
    </div>
  );
}

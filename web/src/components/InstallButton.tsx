import { Share, Smartphone, X } from "lucide-react";
import { useEffect, useState } from "react";

// "Install app": Chrome and Edge (Android, desktop) fire beforeinstallprompt and install with one tap; iOS
// Safari has no such event, so it gets the two steps to do by hand. Hidden once installed.
type PromptEvent = Event & { prompt: () => Promise<void>; userChoice: Promise<{ outcome: string }> };

const standalone = () => window.matchMedia("(display-mode: standalone)").matches || (navigator as any).standalone;
const ios = () => /iphone|ipad|ipod/i.test(navigator.userAgent);

export default function InstallButton() {
  const [prompt, setPrompt] = useState<PromptEvent | null>(null);
  const [help, setHelp] = useState(false);
  useEffect(() => {
    const onPrompt = (e: Event) => { e.preventDefault(); setPrompt(e as PromptEvent); };
    const onInstalled = () => setPrompt(null);
    window.addEventListener("beforeinstallprompt", onPrompt);
    window.addEventListener("appinstalled", onInstalled);
    return () => {
      window.removeEventListener("beforeinstallprompt", onPrompt);
      window.removeEventListener("appinstalled", onInstalled);
    };
  }, []);

  if (standalone() || (!prompt && !ios())) return null;
  const install = async () => {
    if (!prompt) { setHelp(!help); return; }
    await prompt.prompt();
    if ((await prompt.userChoice).outcome === "accepted") setPrompt(null);
  };
  return (
    <div className="relative">
      <button onClick={install} aria-label="Install the app"
              className="flex h-9 items-center gap-1.5 rounded-lg px-2.5 text-sm font-medium text-brand-600 hover:bg-brand-50">
        <Smartphone className="h-4 w-4" strokeWidth={1.75} /> <span className="hidden sm:inline">Install</span>
      </button>
      {help && (
        <div className="absolute right-0 mt-2 w-64 rounded-[var(--radius-card)] border border-line bg-white p-4 text-sm shadow-lg">
          <button onClick={() => setHelp(false)} className="absolute top-2 right-2 p-1 text-muted" aria-label="Close"><X className="h-4 w-4" /></button>
          <p className="font-medium">Add EyeForge to your home screen</p>
          <ol className="mt-2 list-decimal space-y-1 pl-4 text-muted">
            <li>Tap <Share className="inline h-4 w-4 align-text-bottom" /> Share in Safari.</li>
            <li>Choose "Add to Home Screen".</li>
          </ol>
        </div>
      )}
    </div>
  );
}

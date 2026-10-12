import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes, useLocation, useSearchParams } from "react-router-dom";
import "./index.css";
import Landing from "./pages/Landing";
import AppShell from "./pages/AppShell";
import Interview from "./pages/Interview";

// A new chat when "?revise=<job>" or "?kit=<id>" comes or goes (different conversations) or "New model" is pressed.
function FreshInterview() {
  const [params] = useSearchParams();
  const fresh = (useLocation().state as { fresh?: number } | null)?.fresh ?? "";
  return <Interview key={`${params.get("revise") ?? ""}|${params.get("kit") ?? ""}|${params.get("text") ?? ""}|${fresh}`} />;
}
import JobPage from "./pages/Job";
import { Datasets, Kits, Models, Objects, Scenes } from "./pages/Library";
import { Dataset } from "./pages/Dataset";
import { LabelingPage } from "./pages/Labeling";
import Billing from "./pages/Billing";
import { Privacy, Support, Terms } from "./pages/Legal";
// The live test pulls in the model runtime: loaded only when someone opens it.
const Live = lazy(() => import("./pages/Live"));
const LiveBoard = lazy(() => import("./pages/LiveBoard"));

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<AppShell />}>
          <Route index element={<FreshInterview />} />
          <Route path="job/:name" element={<JobPage />} />
          <Route path="models" element={<Models />} />
          <Route path="datasets" element={<Datasets />} />
          <Route path="datasets/:id" element={<Dataset />} />
          <Route path="datasets/:id/labelings/:lid" element={<LabelingPage />} />
          <Route path="assets/objects" element={<Objects />} />
          <Route path="assets/scenes" element={<Scenes />} />
          <Route path="kits" element={<Kits />} />
          <Route path="billing" element={<Billing />} />
        </Route>
        <Route path="/app/job/:name/live" element={<Suspense><Live /></Suspense>} />
        <Route path="/app/job/:name/board" element={<Suspense><LiveBoard /></Suspense>} />
        <Route path="/privacy" element={<Privacy />} />
        <Route path="/terms" element={<Terms />} />
        <Route path="/support" element={<Support />} />
        <Route path="*" element={<Landing />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);

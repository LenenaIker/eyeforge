import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes, useSearchParams } from "react-router-dom";
import "./index.css";
import Landing from "./pages/Landing";
import AppShell from "./pages/AppShell";
import Interview from "./pages/Interview";

// A new chat when "?revise=<job>" comes or goes: a revision and a new model are different conversations.
function FreshInterview() {
  return <Interview key={useSearchParams()[0].get("revise") ?? ""} />;
}
import JobPage from "./pages/Job";
import { Datasets, Kits, Models, Objects, Scenes } from "./pages/Library";
import Billing from "./pages/Billing";
// The live test pulls in the model runtime: loaded only when someone opens it.
const Live = lazy(() => import("./pages/Live"));

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
          <Route path="assets/objects" element={<Objects />} />
          <Route path="assets/scenes" element={<Scenes />} />
          <Route path="kits" element={<Kits />} />
          <Route path="billing" element={<Billing />} />
        </Route>
        <Route path="/app/job/:name/live" element={<Suspense><Live /></Suspense>} />
        <Route path="*" element={<Landing />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);

import { useCallback, useEffect, useState } from "react";
import { api, type ModelInfo } from "./api";
import { cx } from "./components/ui";
import { ApiDocs } from "./pages/ApiDocs";
import { Architecture } from "./pages/Architecture";
import { Gallery } from "./pages/Gallery";
import { Lab } from "./pages/Lab";
import { Overview } from "./pages/Overview";
import { Results } from "./pages/Results";

const ROUTES = [
  ["overview", "Overview"],
  ["lab", "Colorization lab"],
  ["architecture", "Architecture"],
  ["results", "Results"],
  ["gallery", "Gallery"],
  ["api", "API"],
] as const;
type Route = (typeof ROUTES)[number][0];

const current = (): Route => {
  const h = window.location.hash.replace("#/", "") as Route;
  return ROUTES.some(([r]) => r === h) ? h : "overview";
};

export default function App() {
  const [route, setRoute] = useState<Route>(current);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    const onHash = () => { setRoute(current()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", onHash);
    api.model().then(setModel).catch(() => setOffline(true));
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const label = ROUTES.find(([r]) => r === route)?.[1];
    document.title = `${label} · SAR Colorization Workbench`;
  }, [route]);

  const go = useCallback((r: string) => { window.location.hash = `#/${r}`; }, []);

  return (
    <div className="min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-cyan focus:px-3 focus:py-1.5 focus:text-night">Skip to content</a>
      <header className="sticky top-0 z-40 border-b border-edge bg-night/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1240px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
          <a href="#/overview" className="flex items-center gap-2.5">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span className="text-sm font-semibold tracking-tight">SAR Colorization <span className="font-normal text-dim">Workbench</span></span>
          </a>
          <nav aria-label="Primary" className="-mx-1 flex flex-1 gap-1 overflow-x-auto">
            {ROUTES.map(([r, label]) => (
              <a key={r} href={`#/${r}`} aria-current={route === r ? "page" : undefined}
                className={cx("whitespace-nowrap rounded-md px-2.5 py-1.5 text-sm", route === r ? "bg-panel-2 text-ink" : "text-dim hover:text-ink")}>{label}</a>
            ))}
          </nav>
          <a href="https://github.com/parasar0810/SAR-Imagery-Colorization" target="_blank" rel="noreferrer" className="text-sm text-dim hover:text-ink">GitHub ↗</a>
        </div>
      </header>

      <main id="main" className="mx-auto max-w-[1240px] px-4 py-8 sm:px-6">
        {offline && <div role="alert" className="mb-5 rounded-md border border-rose/40 bg-rose/10 px-4 py-2.5 text-sm text-rose">The inference server is unreachable. Pages that need it will show errors.</div>}
        {route === "overview" && <Overview model={model} go={go} />}
        {route === "lab" && <Lab model={model} />}
        {route === "architecture" && <Architecture />}
        {route === "results" && <Results />}
        {route === "gallery" && <Gallery />}
        {route === "api" && <ApiDocs model={model} />}
      </main>

      <footer className="border-t border-edge">
        <div className="mx-auto flex max-w-[1240px] flex-wrap justify-between gap-2 px-4 py-5 text-xs text-dim sm:px-6">
          <p>Model &amp; notebook: Fahad Ur Rehman (MIT). Service &amp; workbench: Priyanshu Parasar.</p>
          <p>Generated colours are model predictions, not measurements.</p>
        </div>
      </footer>
    </div>
  );
}

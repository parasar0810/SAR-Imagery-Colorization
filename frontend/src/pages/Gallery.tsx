import { useEffect, useState } from "react";
import { api, type Sample } from "../api";
import { Badge, Panel, ZoomImage } from "../components/ui";

export function Gallery() {
  const [samples, setSamples] = useState<Sample[] | null>(null);
  useEffect(() => {
    api.samples().then(setSamples).catch(() => setSamples([]));
  }, []);
  return (
    <div className="space-y-5">
      <header>
        <p className="font-mono text-xs uppercase tracking-[0.14em] text-cyan">Sample gallery</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">Input, prediction and reference, side by side</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">Every image here is from the original author’s published validation figure (MIT). Generated panels are precomputed with their checkpoint, not produced live. Only scene types that actually exist are shown.</p>
      </header>
      {samples === null && <p className="text-sm text-dim" role="status">Loading samples…</p>}
      {samples?.length === 0 && <p className="text-sm text-dim">No samples are installed.</p>}
      {samples?.map((s) => (
        <Panel key={s.id} title={s.title} kicker={s.category} actions={<Badge tone="precomputed">precomputed</Badge>}>
          <div className="grid gap-3 sm:grid-cols-3">
            {(["sar", "generated", "reference"] as const).filter((k) => s.assets.includes(k)).map((k) => (
              <ZoomImage key={k} src={api.sampleUrl(s.id, k)} alt={`${s.title}: ${k}`} label={{ sar: "SAR input", generated: "Generated (upstream checkpoint)", reference: "Optical reference" }[k]} />
            ))}
          </div>
          <p className="mt-3 text-sm text-ink-2">{s.note}</p>
          <p className="mt-1 text-xs text-dim">{s.resolution_note}</p>
        </Panel>
      ))}
    </div>
  );
}

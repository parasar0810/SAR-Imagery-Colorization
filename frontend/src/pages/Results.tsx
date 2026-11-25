import { useEffect, useState } from "react";
import { api, type Evaluation } from "../api";
import { Badge, Notice, Panel, Stat } from "../components/ui";

export function Results() {
  const [ev, setEv] = useState<Evaluation | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    api.evaluation().then(setEv).catch(() => setErr(true));
  }, []);

  return (
    <div className="space-y-5">
      <header>
        <p className="font-mono text-xs uppercase tracking-[0.14em] text-cyan">Results &amp; evaluation</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">What has, and hasn’t, been measured</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">Only numbers produced by <code className="font-mono text-cyan-soft">scripts/evaluate.py</code> in this repository are shown as measured. Anything else is labelled with its source.</p>
      </header>

      {err && <Notice title="Unavailable">Could not load evaluation data from the server.</Notice>}

      {ev && !ev.available && (
        <>
          <Panel title="Reproduced in this repository" actions={<Badge tone="warn">not yet run</Badge>}>
            <p className="text-sm text-ink-2">{ev.message}</p>
            <div className="mt-4 grid gap-2 sm:grid-cols-4">
              <Stat label="PSNR" value="—" note="unavailable" />
              <Stat label="SSIM" value="—" note="unavailable" />
              <Stat label="MAE" value="—" note="unavailable" />
              <Stat label="Latency" value="—" note="needs checkpoint" />
            </div>
          </Panel>
          <Panel title="Reported by the original author" actions={<Badge tone="precomputed">upstream · not reproduced</Badge>}>
            <div className="grid gap-2 sm:grid-cols-3">
              <Stat label="PSNR" value={`${ev.upstream_reported.psnr_db} dB`} />
              <Stat label="SSIM" value={ev.upstream_reported.ssim} />
              <Stat label="Test images" value={ev.upstream_reported.n_images} note="2% seeded hold-out" />
            </div>
            <p className="mt-3 text-xs text-dim">Source: {ev.upstream_reported.source}</p>
          </Panel>
        </>
      )}

      {ev && ev.available && (
        <Panel title="Measured in this repository" actions={<Badge tone="live">measured</Badge>}>
          <pre className="overflow-x-auto rounded bg-night p-3 font-mono text-xs text-ink-2">{JSON.stringify(ev, null, 2)}</pre>
        </Panel>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="How to read PSNR and SSIM here">
          <ul className="space-y-2 text-sm text-ink-2">
            <li><strong className="text-ink">Reference-based.</strong> Both compare the prediction with the <em>pixel-aligned</em> optical tile of the same scene from the paired dataset. They cannot be computed for an arbitrary upload, which has no ground truth.</li>
            <li><strong className="text-ink">Low absolute values are expected.</strong> SAR and optical sensors measure different physics; many optical details (paint colour, crop type, snow) are not recoverable from backscatter, so pixel metrics stay modest even for plausible outputs.</li>
            <li><strong className="text-ink">Not a perceptual score.</strong> PSNR rewards blur; a sharper, plausible output can score lower than a smooth average.</li>
          </ul>
        </Panel>
        <Panel title="Methodology caveats">
          <ul className="space-y-2 text-sm text-ink-2">
            <li><strong className="text-ink">Possible spatial leakage.</strong> The split is a seeded shuffle of tile files. If neighbouring tiles from one scene land in both train and test, test scores can be optimistic. A scene- or region-level split would be stricter.</li>
            <li><strong className="text-ink">Single dataset.</strong> Generalisation to other sensors, polarisations, incidence angles or regions has not been evaluated.</li>
            <li><strong className="text-ink">No training curves here.</strong> The notebook saves <code className="font-mono">history.json</code> during training, but no training run’s history is in the repository.</li>
          </ul>
        </Panel>
      </div>

      <Panel title="Qualitative result (original author’s validation figure)" actions={<Badge tone="precomputed">precomputed</Badge>}>
        <div className="grid gap-3 sm:grid-cols-3">
          {(["sar", "generated", "reference"] as const).map((k) => (
            <figure key={k}>
              <img src={api.sampleUrl("upstream-02", k)} alt={`Mountain terrain: ${k}`} className="aspect-square w-full rounded-md border border-edge object-cover" />
              <figcaption className="mt-1.5 font-mono text-[0.7rem] text-dim">{{ sar: "SAR input", generated: "Generated", reference: "Optical reference" }[k]}</figcaption>
            </figure>
          ))}
        </div>
        <p className="mt-3 text-xs text-dim">Failure mode visible here: relief shading is learned, but snow fields and the road network in the reference are absent from the prediction — information SAR backscatter alone doesn’t carry well at this resolution.</p>
      </Panel>
    </div>
  );
}

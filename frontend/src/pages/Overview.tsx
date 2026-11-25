import type { ModelInfo } from "../api";
import { api } from "../api";
import { Badge, CompareSlider, Panel, Stat } from "../components/ui";

export function Overview({ model, go }: { model: ModelInfo | null; go: (r: string) => void }) {
  return (
    <div className="space-y-6">
      <section className="grid items-center gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)]">
        <div>
          <div className="flex flex-wrap gap-2">
            <Badge tone={model?.ready ? "live" : "warn"}>{model?.ready ? "● Live inference available" : "○ Live inference offline — no checkpoint installed"}</Badge>
            <Badge>PyTorch · FastAPI · React</Badge>
          </div>
          <h1 className="mt-5 text-3xl font-semibold leading-tight tracking-tight sm:text-[2.6rem]">Seeing through clouds, then colouring what radar sees</h1>
          <p className="mt-4 max-w-2xl text-[1.02rem] leading-relaxed text-ink-2">
            A conditional GAN that translates grayscale <strong className="text-ink">Synthetic Aperture Radar</strong> tiles into
            optical-style RGB, wrapped in an inference service and this workbench. The model predicts what an optical camera
            <em> might</em> have seen. It does not recover true colour.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <button type="button" onClick={() => go("lab")} className="rounded-md bg-cyan px-4 py-2 text-sm font-semibold text-night">Open the colorization lab</button>
            <button type="button" onClick={() => go("architecture")} className="rounded-md border border-edge-strong px-4 py-2 text-sm hover:border-cyan hover:text-cyan">See the architecture</button>
          </div>
          <div className="mt-6 grid max-w-xl grid-cols-3 gap-2">
            <Stat label="Generator" value="42.5 M" note="parameters" />
            <Stat label="Input → output" value="1×256² → 3×256²" />
            <Stat label="CPU latency" value="≈139 ms" note="median warm, 256² tile, laptop CPU" />
          </div>
        </div>
        <Panel kicker="Original author’s result" title="Agricultural fields · drag to compare" actions={<Badge tone="precomputed">precomputed</Badge>}>
          <CompareSlider before={api.sampleUrl("upstream-01", "sar")} after={api.sampleUrl("upstream-01", "generated")} beforeLabel="SAR" afterLabel="Generated" alt="agricultural fields" />
        </Panel>
      </section>

      <div className="grid gap-5 md:grid-cols-3">
        <Panel title="What is SAR?">
          <p className="text-sm leading-relaxed text-ink-2">A satellite radar sends microwave pulses and records the echo. Because it brings its own illumination at wavelengths that pass through cloud, it images the ground day or night, in any weather.</p>
        </Panel>
        <Panel title="Why it looks so different">
          <p className="text-sm leading-relaxed text-ink-2">Pixel brightness measures <em>backscatter</em> — surface roughness, geometry and moisture — not reflected sunlight. The result is grainy (speckle), has no colour, and distorts steep terrain (layover and shadow), which makes it hard for non-specialists to read.</p>
        </Panel>
        <Panel title="What this model does">
          <p className="text-sm leading-relaxed text-ink-2">Trained on ~16k paired SAR/optical tiles, the generator learns a mapping from backscatter texture to plausible optical appearance. It helps visual interpretation; it should not be used to measure land cover, vegetation health or anything else colour-dependent.</p>
        </Panel>
      </div>

      <Panel title="Credits">
        <p className="text-sm text-ink-2">
          Model design, training notebook and the result images: <a className="text-cyan-soft underline underline-offset-2" href="https://github.com/fahad-mughal-rehman/sar-colorization-gan" target="_blank" rel="noreferrer">Fahad Ur Rehman (sar-colorization-gan, MIT)</a>.
          Inference package, API, this workbench, tests and documentation: Priyanshu Parasar.
          Dataset: <a className="text-cyan-soft underline underline-offset-2" href="https://www.kaggle.com/datasets/anjaliikakde/paired-sar-optical-dataset-16k-images" target="_blank" rel="noreferrer">Paired SAR-Optical Dataset (16K) on Kaggle</a>.
        </p>
      </Panel>
    </div>
  );
}

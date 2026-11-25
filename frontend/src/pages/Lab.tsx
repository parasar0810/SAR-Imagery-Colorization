import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ApiError, type ColorizeResult, type ModelInfo, type PreprocessResult, type Sample } from "../api";
import { Badge, CompareSlider, Notice, Panel, Stat, ZoomImage, cx } from "../components/ui";

type Source = { kind: "upload"; file: File; url: string } | { kind: "sample"; sample: Sample; url: string };
type Phase = "idle" | "preprocessing" | "ready" | "running" | "done" | "error";
const ACCEPT = ["image/png", "image/jpeg", "image/tiff", "image/webp", "image/bmp"];

export function Lab({ model }: { model: ModelInfo | null }) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [src, setSrc] = useState<Source | null>(null);
  const [pre, setPre] = useState<PreprocessResult | null>(null);
  const [result, setResult] = useState<ColorizeResult | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [err, setErr] = useState<string | null>(null);
  const [view, setView] = useState<"slider" | "side">("slider");
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const maxMb = model?.limits.max_upload_mb ?? 8;

  useEffect(() => {
    api.samples().then(setSamples).catch(() => setSamples([]));
  }, []);

  const reset = useCallback(() => {
    if (src?.kind === "upload") URL.revokeObjectURL(src.url);
    setSrc(null); setPre(null); setResult(null); setErr(null); setPhase("idle");
    if (input.current) input.current.value = "";
  }, [src]);

  const load = async (blob: Blob, name: string, next: Source) => {
    setSrc(next); setResult(null); setErr(null); setPhase("preprocessing");
    try {
      setPre(await api.preprocess(blob, name));
      setPhase("ready");
    } catch (e) {
      setErr((e as ApiError).message); setPhase("error");
    }
  };

  const onFile = (file: File | undefined) => {
    if (!file) return;
    if (!ACCEPT.includes(file.type)) { setErr(`Unsupported file type ${file.type || "(unknown)"}. Use PNG, JPEG, TIFF, WEBP or BMP.`); setPhase("error"); return; }
    if (file.size > maxMb * 1024 * 1024) { setErr(`File is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${maxMb} MB.`); setPhase("error"); return; }
    if (src?.kind === "upload") URL.revokeObjectURL(src.url);
    const url = URL.createObjectURL(file);
    void load(file, file.name, { kind: "upload", file, url });
  };

  const onSample = async (s: Sample) => {
    const url = api.sampleUrl(s.id, "sar");
    const blob = await fetch(url).then((r) => r.blob());
    void load(blob, `${s.id}.png`, { kind: "sample", sample: s, url });
  };

  const run = async () => {
    if (!src) return;
    setPhase("running"); setErr(null);
    try {
      const blob = src.kind === "upload" ? src.file : await fetch(src.url).then((r) => r.blob());
      setResult(await api.colorize(blob, src.kind === "upload" ? src.file.name : `${src.sample.id}.png`));
      setPhase("done");
    } catch (e) {
      setErr((e as ApiError).message); setPhase("error");
    }
  };

  const live = !!model?.ready;
  const precomputed = !live && src?.kind === "sample" && src.sample.assets.includes("generated") ? api.sampleUrl(src.sample.id, "generated") : null;
  const output = result?.output ?? precomputed;
  const reference = src?.kind === "sample" && src.sample.assets.includes("reference") ? api.sampleUrl(src.sample.id, "reference") : null;

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.14em] text-cyan">Colorization lab</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">Translate a SAR tile into optical-style colour</h1>
          <p className="mt-1 max-w-2xl text-sm text-ink-2">Upload a single-band SAR image or pick a sample. The server applies the training preprocessing, runs the generator once and returns its prediction.</p>
        </div>
        <Badge tone={live ? "live" : "warn"}>{live ? `● Live inference · ${model?.device}` : "○ Live inference offline"}</Badge>
      </header>

      {!live && model && (
        <Notice title="No trained checkpoint on this server">
          {model.reason} Preprocessing still runs live. For the two upstream samples, the output shown is the <strong className="text-ink">original author’s precomputed result</strong>, not a live prediction.
        </Notice>
      )}

      <div className="grid gap-5 lg:grid-cols-[22rem_minmax(0,1fr)]">
        {/* 1 — input */}
        <div className="space-y-5">
          <Panel kicker="Step 1" title="Choose input">
            <div
              onDragOver={(e) => { e.preventDefault(); setOver(true); }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); onFile(e.dataTransfer.files[0]); }}
              className={cx("grid place-items-center rounded-md border border-dashed px-4 py-7 text-center transition-colors", over ? "border-cyan bg-cyan/5" : "border-edge-strong")}
            >
              <p className="text-sm text-ink">Drop a SAR image here</p>
              <p className="mt-1 text-xs text-dim">PNG, JPEG, TIFF, WEBP, BMP · ≤ {maxMb} MB · ≤ {((model?.limits.max_pixels ?? 16777216) / 1e6).toFixed(1)} MP</p>
              <button type="button" onClick={() => input.current?.click()} className="mt-3 rounded-md border border-edge-strong px-3 py-1.5 text-sm hover:border-cyan hover:text-cyan">Browse files</button>
              <input ref={input} type="file" accept={ACCEPT.join(",")} className="sr-only" onChange={(e) => onFile(e.target.files?.[0])} aria-label="Upload SAR image" />
            </div>
            {samples.length > 0 && (
              <div className="mt-4">
                <p className="mb-2 font-mono text-[0.68rem] uppercase tracking-[0.12em] text-dim">Or use a sample</p>
                <div className="grid grid-cols-2 gap-2">
                  {samples.map((s) => (
                    <button key={s.id} type="button" onClick={() => onSample(s)} aria-pressed={src?.kind === "sample" && src.sample.id === s.id}
                      className={cx("overflow-hidden rounded-md border text-left", src?.kind === "sample" && src.sample.id === s.id ? "border-cyan" : "border-edge hover:border-edge-strong")}>
                      <img src={api.sampleUrl(s.id, "sar")} alt={`SAR sample: ${s.title}`} className="aspect-square w-full object-cover" />
                      <span className="block px-2 py-1.5 text-xs text-ink-2">{s.title}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </Panel>

          <Panel kicker="Step 2" title="Preprocessing" actions={pre && <Badge tone="live">live</Badge>}>
            {phase === "preprocessing" && <p className="text-sm text-dim" role="status">Decoding and resizing…</p>}
            {!pre && phase !== "preprocessing" && <p className="text-sm text-dim">Shows the exact tensor the generator receives.</p>}
            {pre && (
              <div className="space-y-3">
                <ZoomImage src={pre.preprocessed} alt="Grayscale 256×256 model input" label={`Model input · ${pre.model_input_size}²`} pixelated />
                <dl className="grid grid-cols-2 gap-2 text-xs">
                  <Stat label="Original" value={`${pre.input.width}×${pre.input.height}`} note={`${pre.input.format} · ${pre.input.mode}`} />
                  <Stat label="Model input" value="1×256×256" note="bicubic · [−1, 1] · channel mean" />
                </dl>
              </div>
            )}
          </Panel>
        </div>

        {/* 3 — run + compare */}
        <div className="space-y-5">
          <Panel
            kicker="Step 3"
            title="Run and compare"
            actions={
              <div className="flex gap-2">
                {src && <button type="button" onClick={reset} className="rounded-md px-3 py-1.5 text-sm text-ink-2 hover:bg-panel-2">Reset</button>}
                <button type="button" onClick={run} disabled={!live || !pre || phase === "running"}
                  className="rounded-md bg-cyan px-3.5 py-1.5 text-sm font-semibold text-night disabled:cursor-not-allowed disabled:bg-edge disabled:text-dim">
                  {phase === "running" ? "Running…" : "Run inference"}
                </button>
              </div>
            }
          >
            {err && <div role="alert" className="mb-4 rounded-md border border-rose/40 bg-rose/10 px-3 py-2 text-sm text-rose">{err}</div>}
            {phase === "running" && (
              <div className="mb-4 h-1 overflow-hidden rounded bg-edge" role="progressbar" aria-label="Inference in progress">
                <div className="h-full w-1/3 animate-pulse bg-cyan" />
              </div>
            )}

            {!src && <div className="grid aspect-[2/1] place-items-center rounded-md border border-edge bg-night/60 text-sm text-dim grid-bg">Choose an input to begin.</div>}

            {src && output && (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    {result ? <Badge tone="live">Live prediction · {result.inference_ms.toFixed(0)} ms</Badge> : <Badge tone="precomputed">Precomputed by original author’s checkpoint</Badge>}
                  </div>
                  <div className="flex rounded-md border border-edge p-0.5 text-xs" role="tablist" aria-label="Comparison view">
                    {(["slider", "side"] as const).map((v) => (
                      <button key={v} role="tab" aria-selected={view === v} type="button" onClick={() => setView(v)} className={cx("rounded px-2.5 py-1", view === v ? "bg-panel-2 text-ink" : "text-dim")}>{v === "slider" ? "Slider" : "Side by side"}</button>
                    ))}
                  </div>
                </div>
                {view === "slider" ? (
                  <div className="mx-auto max-w-xl">
                    <CompareSlider before={src.url} after={output} beforeLabel="SAR input" afterLabel={result ? "Model prediction" : "Precomputed output"} alt={src.kind === "sample" ? src.sample.title : "uploaded image"} />
                  </div>
                ) : (
                  <div className={cx("grid gap-3", reference ? "sm:grid-cols-3" : "sm:grid-cols-2")}>
                    <ZoomImage src={src.url} alt="SAR input" label="SAR input" />
                    <ZoomImage src={output} alt="Generated optical-style image" label={result ? "Model prediction" : "Precomputed output"} />
                    {reference && <ZoomImage src={reference} alt="Paired optical reference" label="Optical reference" />}
                  </div>
                )}
                <p className="text-xs leading-relaxed text-dim">
                  Colours are the generator’s <strong className="text-ink-2">prediction</strong> of an optical-style rendering. SAR measures radar backscatter, not colour; the output is a learned estimate and can be wrong, especially outside the training distribution.
                </p>
                {result && (
                  <div className="flex flex-wrap items-center gap-3">
                    <a href={result.output} download="sar-colorized.png" className="rounded-md border border-edge-strong px-3 py-1.5 text-sm hover:border-cyan hover:text-cyan">Download output (PNG)</a>
                    <span className="font-mono text-xs text-dim">{result.model_input_size}×{result.model_input_size} · {result.inference_ms} ms on {model?.device}</span>
                  </div>
                )}
              </div>
            )}

            {src && !output && phase !== "running" && (
              <div className="grid gap-3 sm:grid-cols-2">
                <ZoomImage src={src.url} alt="SAR input" label="SAR input" />
                <div className="grid aspect-square place-items-center rounded-md border border-edge bg-night/60 p-6 text-center text-sm text-dim grid-bg">
                  {live ? "Press Run inference to generate the prediction." : "Live inference needs a trained checkpoint. Uploaded images can be preprocessed, but not colorized, until one is installed."}
                </div>
              </div>
            )}
          </Panel>

          {src?.kind === "sample" && (
            <Panel kicker="Provenance" title={src.sample.title}>
              <p className="text-sm text-ink-2">{src.sample.note}</p>
              <p className="mt-2 text-xs leading-relaxed text-dim">{src.sample.provenance} {src.sample.resolution_note}</p>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}

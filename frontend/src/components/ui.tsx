import { useEffect, useRef, useState, type ReactNode } from "react";

export function cx(...p: (string | false | null | undefined)[]) {
  return p.filter(Boolean).join(" ");
}

export function Panel({ title, kicker, actions, children, className }: { title?: ReactNode; kicker?: string; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cx("min-w-0 rounded-lg border border-edge bg-panel", className)}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-edge px-4 py-3">
          <div>
            {kicker && <p className="font-mono text-[0.68rem] uppercase tracking-[0.14em] text-cyan">{kicker}</p>}
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Badge({ tone = "neutral", children }: { tone?: "neutral" | "live" | "warn" | "precomputed"; children: ReactNode }) {
  const t = {
    neutral: "border-edge-strong text-ink-2",
    live: "border-cyan/40 bg-cyan/10 text-cyan-soft",
    warn: "border-amber/40 bg-amber/10 text-amber",
    precomputed: "border-violet-400/40 bg-violet-400/10 text-violet-300",
  }[tone];
  return <span className={cx("inline-flex items-center gap-1.5 rounded border px-2 py-0.5 font-mono text-[0.68rem]", t)}>{children}</span>;
}

/** Before/after slider. Native range input → keyboard and screen-reader accessible. */
export function CompareSlider({ before, after, beforeLabel, afterLabel, alt }: { before: string; after: string; beforeLabel: string; afterLabel: string; alt: string }) {
  const [pos, setPos] = useState(50);
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const fromPointer = (clientX: number) => {
    const r = box.current?.getBoundingClientRect();
    if (r) setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  };

  return (
    <div>
      <div
        ref={box}
        className="relative aspect-square w-full touch-none select-none overflow-hidden rounded-md border border-edge bg-night"
        onPointerDown={(e) => { dragging.current = true; (e.target as Element).setPointerCapture?.(e.pointerId); fromPointer(e.clientX); }}
        onPointerMove={(e) => dragging.current && fromPointer(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
      >
        <img src={after} alt={`${afterLabel}: ${alt}`} className="absolute inset-0 h-full w-full object-cover" draggable={false} />
        <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
          <img src={before} alt={`${beforeLabel}: ${alt}`} className="absolute inset-0 h-full w-full object-cover" draggable={false} />
        </div>
        <div className="pointer-events-none absolute inset-y-0" style={{ left: `${pos}%` }} aria-hidden="true">
          <div className="h-full w-px -translate-x-1/2 bg-cyan shadow-[0_0_0_1px_rgba(11,18,32,.6)]" />
          <div className="absolute top-1/2 grid h-8 w-8 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-cyan bg-night/90 font-mono text-xs text-cyan">⇔</div>
        </div>
        <span className="pointer-events-none absolute left-2 top-2 rounded bg-night/80 px-1.5 py-0.5 font-mono text-[0.68rem] text-ink-2">{beforeLabel}</span>
        <span className="pointer-events-none absolute right-2 top-2 rounded bg-night/80 px-1.5 py-0.5 font-mono text-[0.68rem] text-ink-2">{afterLabel}</span>
      </div>
      <label className="mt-3 flex items-center gap-3 text-xs text-dim">
        <span className="shrink-0">Reveal</span>
        <input type="range" min={0} max={100} value={Math.round(pos)} onChange={(e) => setPos(Number(e.target.value))} className="w-full accent-cyan-400" aria-label={`Comparison position between ${beforeLabel} and ${afterLabel}`} />
        <span className="w-10 shrink-0 text-right font-mono tabular-nums">{Math.round(pos)}%</span>
      </label>
    </div>
  );
}

/** Image tile that opens a fullscreen dialog with zoom (1×–4×). */
export function ZoomImage({ src, alt, label, className, pixelated }: { src: string; alt: string; label: string; className?: string; pixelated?: boolean }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [zoom, setZoom] = useState(1);
  useEffect(() => {
    const d = ref.current;
    const onClose = () => setZoom(1);
    d?.addEventListener("close", onClose);
    return () => d?.removeEventListener("close", onClose);
  }, []);
  return (
    <>
      <button type="button" onClick={() => ref.current?.showModal()} className={cx("group relative block w-full overflow-hidden rounded-md border border-edge bg-night text-left", className)} aria-label={`Open ${label} fullscreen`}>
        <img src={src} alt={alt} className={cx("aspect-square w-full object-cover transition-opacity group-hover:opacity-90", pixelated && "pixelated")} />
        <span className="absolute bottom-2 left-2 rounded bg-night/85 px-1.5 py-0.5 font-mono text-[0.68rem] text-ink-2">{label}</span>
        <span className="absolute bottom-2 right-2 rounded bg-night/85 px-1.5 py-0.5 font-mono text-[0.68rem] text-dim opacity-0 group-hover:opacity-100">⤢ zoom</span>
      </button>
      <dialog ref={ref} className="m-auto max-h-[94vh] w-[min(1100px,96vw)] max-w-none rounded-lg border border-edge bg-panel p-0 text-ink backdrop:bg-black/80" aria-label={label} onClick={(e) => e.target === e.currentTarget && ref.current?.close()}>
        <div className="flex items-center justify-between gap-3 border-b border-edge px-4 py-2.5">
          <p className="text-sm font-medium">{label}</p>
          <div className="flex items-center gap-2">
            {[1, 2, 4].map((z) => (
              <button key={z} type="button" onClick={() => setZoom(z)} aria-pressed={zoom === z} className={cx("rounded px-2 py-1 font-mono text-xs", zoom === z ? "bg-cyan/15 text-cyan" : "text-ink-2 hover:bg-panel-2")}>{z}×</button>
            ))}
            <button type="button" onClick={() => ref.current?.close()} className="ml-2 rounded border border-edge px-2.5 py-1 text-xs hover:bg-panel-2" autoFocus>Close</button>
          </div>
        </div>
        <div className="max-h-[80vh] overflow-auto bg-night p-3">
          <img src={src} alt={alt} style={{ width: `${zoom * 100}%`, maxWidth: "none" }} className={cx("mx-auto h-auto", pixelated && "pixelated")} />
        </div>
      </dialog>
    </>
  );
}

export function Stat({ label, value, note }: { label: string; value: ReactNode; note?: string }) {
  return (
    <div className="rounded-md border border-edge bg-panel-2/60 px-3 py-2.5">
      <p className="font-mono text-[0.66rem] uppercase tracking-[0.12em] text-dim">{label}</p>
      <p className="mt-1 font-mono text-base text-ink">{value}</p>
      {note && <p className="mt-0.5 text-[0.72rem] text-dim">{note}</p>}
    </div>
  );
}

export function Notice({ tone = "warn", title, children }: { tone?: "warn" | "info"; title: string; children: ReactNode }) {
  return (
    <div role="note" className={cx("rounded-md border px-4 py-3 text-sm", tone === "warn" ? "border-amber/40 bg-amber/5" : "border-cyan/30 bg-cyan/5")}>
      <p className={cx("font-mono text-[0.68rem] uppercase tracking-[0.12em]", tone === "warn" ? "text-amber" : "text-cyan")}>{title}</p>
      <div className="mt-1 leading-relaxed text-ink-2">{children}</div>
    </div>
  );
}

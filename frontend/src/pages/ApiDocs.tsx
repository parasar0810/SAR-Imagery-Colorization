import type { ModelInfo } from "../api";
import { Panel } from "../components/ui";

const ENDPOINTS: [string, string, string][] = [
  ["GET", "/api/health", "Liveness and whether a checkpoint is loaded."],
  ["GET", "/api/model", "Architecture, I/O contract, device, checkpoint metadata and limits."],
  ["POST", "/api/colorize", "multipart field `file` → base64 PNG prediction, preprocessed input, latency."],
  ["POST", "/api/preprocess", "multipart field `file` → the exact 256×256 grayscale model input. Works without weights."],
  ["GET", "/api/samples", "Bundled sample metadata, with provenance."],
  ["GET", "/api/samples/{id}/{sar|generated|reference}", "Sample image (PNG)."],
  ["GET", "/api/evaluation", "Measured metrics, or an explicit ‘not run’ with upstream-reported values."],
];

const ERRORS: [string, string, string][] = [
  ["400", "invalid_image", "Not decodable, truncated, too small, or over the pixel limit"],
  ["413", "too_large", "Upload exceeds the size limit"],
  ["415", "invalid_image", "Unsupported format"],
  ["429", "rate_limited", "Too many requests from this client"],
  ["503", "model_unavailable", "No checkpoint installed or it failed to load"],
  ["504", "timeout", "Inference exceeded the timeout"],
];

export function ApiDocs({ model }: { model: ModelInfo | null }) {
  const l = model?.limits;
  return (
    <div className="space-y-5">
      <header>
        <p className="font-mono text-xs uppercase tracking-[0.14em] text-cyan">API</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">Inference API</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">JSON over HTTP. Interactive OpenAPI docs are at <a href="/api/docs" className="font-mono text-cyan-soft underline underline-offset-2">/api/docs</a>. Errors share one shape: <code className="font-mono text-ink">{`{"error": {"code", "message"}}`}</code>.</p>
      </header>
      <Panel title="Endpoints">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[620px] text-left text-sm">
            <tbody>{ENDPOINTS.map(([m, p, d]) => (
              <tr key={p} className="border-t border-edge first:border-0 align-top">
                <td className="py-2 pr-3"><span className={`rounded px-1.5 py-0.5 font-mono text-[0.7rem] ${m === "GET" ? "bg-cyan/10 text-cyan-soft" : "bg-amber/10 text-amber"}`}>{m}</span></td>
                <td className="py-2 pr-4 font-mono text-[0.82rem] text-ink">{p}</td>
                <td className="py-2 text-ink-2">{d}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Panel>
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Example">
          <pre className="overflow-x-auto rounded bg-night p-3 font-mono text-xs leading-relaxed text-ink-2">{`curl -F "file=@tile.png" https://<host>/api/colorize

{
  "output": "data:image/png;base64,...",
  "preprocessed": "data:image/png;base64,...",
  "inference_ms": 184.2,
  "input": {"width": 512, "height": 512, "format": "PNG", "mode": "L"},
  "model_input_size": 256,
  "label": "Model prediction — synthetic optical-style colours, not ground truth."
}`}</pre>
        </Panel>
        <Panel title="Limits (from the running server)">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <dt className="text-dim">Max upload</dt><dd className="font-mono">{l ? `${l.max_upload_mb} MB` : "—"}</dd>
            <dt className="text-dim">Max pixels</dt><dd className="font-mono">{l ? l.max_pixels.toLocaleString() : "—"}</dd>
            <dt className="text-dim">Formats</dt><dd className="font-mono">{l?.formats.join(", ") ?? "—"}</dd>
            <dt className="text-dim">Rate limit</dt><dd className="font-mono">{l ? `${l.rate_limit_per_min} / min / client` : "—"}</dd>
            <dt className="text-dim">Concurrency</dt><dd className="font-mono">{l ? `${l.concurrency} inference at a time` : "—"}</dd>
            <dt className="text-dim">Timeout</dt><dd className="font-mono">{l ? `${l.timeout_s} s` : "—"}</dd>
          </dl>
        </Panel>
      </div>
      <Panel title="Error codes">
        <table className="w-full text-left text-sm">
          <tbody>{ERRORS.map(([s, c, d]) => (
            <tr key={s + c} className="border-t border-edge first:border-0"><td className="py-2 pr-3 font-mono text-ink">{s}</td><td className="py-2 pr-3 font-mono text-cyan-soft">{c}</td><td className="py-2 text-ink-2">{d}</td></tr>
          ))}</tbody>
        </table>
      </Panel>
    </div>
  );
}

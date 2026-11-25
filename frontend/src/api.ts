// Typed client for the FastAPI service. All calls are same-origin (/api/...).

export type ApiError = { code: string; message: string; status: number };

export type ModelInfo = {
  ready: boolean;
  reason: string | null;
  architecture: string;
  input: { channels: number; size: [number, number]; range: [number, number]; preprocessing: string };
  output: { channels: number; size: [number, number]; activation: string; meaning: string };
  device: string;
  checkpoint: Record<string, unknown> | null;
  limits: { max_upload_mb: number; max_pixels: number; timeout_s: number; rate_limit_per_min: number; concurrency: number; formats: string[] };
  attribution: string;
};

export type ImageInfo = { width: number; height: number; format: string; mode: string };

export type ColorizeResult = {
  output: string;
  preprocessed: string;
  inference_ms: number;
  input: ImageInfo;
  model_input_size: number;
  label: string;
};

export type PreprocessResult = { preprocessed: string; model_input_size: number; input: ImageInfo };

export type Sample = {
  id: string;
  title: string;
  category: string;
  note: string;
  provenance: string;
  resolution_note: string;
  assets: string[];
};

export type Evaluation =
  | { available: false; message: string; upstream_reported: { psnr_db: string; ssim: string; n_images: number; source: string } }
  | ({ available: true; source: string } & Record<string, unknown>);

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw { code: "network", message: "Could not reach the inference server.", status: 0 } satisfies ApiError;
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const e = body?.error ?? {};
    throw { code: e.code ?? "http_error", message: e.message ?? `Request failed (${res.status}).`, status: res.status } satisfies ApiError;
  }
  return body as T;
}

const form = (file: Blob, name = "upload.png") => {
  const f = new FormData();
  f.append("file", file, name);
  return f;
};

export const api = {
  model: () => request<ModelInfo>("/api/model"),
  samples: () => request<Sample[]>("/api/samples"),
  evaluation: () => request<Evaluation>("/api/evaluation"),
  colorize: (file: Blob, name?: string) => request<ColorizeResult>("/api/colorize", { method: "POST", body: form(file, name) }),
  preprocess: (file: Blob, name?: string) => request<PreprocessResult>("/api/preprocess", { method: "POST", body: form(file, name) }),
  sampleUrl: (id: string, kind: string) => `/api/samples/${encodeURIComponent(id)}/${encodeURIComponent(kind)}`,
};

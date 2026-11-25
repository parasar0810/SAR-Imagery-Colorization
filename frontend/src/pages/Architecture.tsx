import { Panel } from "../components/ui";

// Values below are read from src/sar_colorization/models.py (ngf = 64).
const ENC = [
  { id: "d1", res: 128, ch: 64, note: "no norm" },
  { id: "d2", res: 64, ch: 128 },
  { id: "d3", res: 32, ch: 256 },
  { id: "d4", res: 16, ch: 512, attn: true },
  { id: "d5", res: 8, ch: 512 },
  { id: "d6", res: 4, ch: 512 },
  { id: "d7", res: 2, ch: 512, note: "no norm" },
];
const DEC = [
  { id: "u1", res: 4, ch: 512, skip: "d6", gated: false, drop: true },
  { id: "u2", res: 8, ch: 512, skip: "d5", gated: false, drop: true },
  { id: "u3", res: 16, ch: 512, skip: "d4", gated: true, drop: true },
  { id: "u4", res: 32, ch: 256, skip: "d3", gated: true },
  { id: "u5", res: 64, ch: 128, skip: "d2", gated: true },
  { id: "u6", res: 128, ch: 64, skip: "d1", gated: true },
];

function Generator() {
  const W = 1000, rowH = 44, top = 112;
  const encX = 150, decX = 640, bw = 210;
  const y = (i: number) => top + i * rowH;
  return (
    <svg viewBox={`0 0 ${W} 500`} className="min-w-[760px]" role="img" aria-labelledby="g-t g-d">
      <title id="g-t">Attention U-Net generator</title>
      <desc id="g-d">Seven stride-2 convolution blocks downsample a 1×256×256 SAR image to 2×2. Self-attention is applied at 16×16. Six transposed-convolution blocks upsample back, concatenating encoder skips; the four highest-resolution skips pass through attention gates. A final transposed convolution with tanh outputs 3×256×256.</desc>
      <defs>
        <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="#8b98ae" /></marker>
      </defs>
      <text x={encX} y={22} fill="#8b98ae" fontSize="11" fontFamily="monospace" letterSpacing="1">ENCODER (Conv · InstanceNorm · LeakyReLU)</text>
      <text x={decX} y={22} fill="#8b98ae" fontSize="11" fontFamily="monospace" letterSpacing="1">DECODER (ConvT · InstanceNorm · ReLU)</text>
      <rect x={10} y={y(0) - 30} width={110} height={34} rx={5} fill="#16213a" stroke="#33425f" />
      <text x={22} y={y(0) - 9} fill="#e6edf7" fontSize="12">SAR 1×256²</text>
      <path d={`M120 ${y(0) - 13} H${encX - 4}`} stroke="#8b98ae" markerEnd="url(#ah)" fill="none" />
      {ENC.map((b, i) => (
        <g key={b.id}>
          <rect x={encX} y={y(i) - 16} width={bw} height={32} rx={5} fill={b.attn ? "#0e2f3a" : "#111a2b"} stroke={b.attn ? "#22d3ee" : "#33425f"} />
          <text x={encX + 10} y={y(i) + 4} fill="#e6edf7" fontSize="12" fontFamily="monospace">{b.id}  {b.ch}×{b.res}²</text>
          {b.attn && <text x={encX + bw - 8} y={y(i) + 4} fill="#67e8f9" fontSize="10.5" textAnchor="end">+ self-attention</text>}
          {b.note && !b.attn && <text x={encX + bw - 8} y={y(i) + 4} fill="#8b98ae" fontSize="10" textAnchor="end">{b.note}</text>}
          {i < ENC.length - 1 && <path d={`M${encX + 30} ${y(i) + 16} V${y(i + 1) - 17}`} stroke="#8b98ae" markerEnd="url(#ah)" fill="none" />}
        </g>
      ))}
      <path d={`M${encX + bw} ${y(6)} H${decX - 4} V${y(5) + 18}`} stroke="#8b98ae" markerEnd="url(#ah)" fill="none" />
      {DEC.map((b) => {
        const row = 6 - (DEC.indexOf(b) + 1);
        const si = ENC.findIndex((e) => e.id === b.skip);
        return (
          <g key={b.id}>
            <rect x={decX} y={y(row) - 16} width={bw} height={32} rx={5} fill="#111a2b" stroke="#33425f" />
            <text x={decX + 10} y={y(row) + 4} fill="#e6edf7" fontSize="12" fontFamily="monospace">{b.id}  {b.ch}×{b.res}²{b.drop ? "  drop .5" : ""}</text>
            <path d={`M${encX + bw} ${y(si)} H${decX - 4}`} stroke={b.gated ? "#22d3ee" : "#33425f"} strokeDasharray={b.gated ? undefined : "4 4"} markerEnd="url(#ah)" fill="none" />
            {b.gated && <rect x={(encX + bw + decX) / 2 - 22} y={y(si) - 9} width={44} height={18} rx={4} fill="#0e2f3a" stroke="#22d3ee" />}
            {b.gated && <text x={(encX + bw + decX) / 2} y={y(si) + 4} fill="#67e8f9" fontSize="10" textAnchor="middle" fontFamily="monospace">gate</text>}
            {row > 0 && <path d={`M${decX + 30} ${y(row) - 16} V${y(row - 1) + 17}`} stroke="#8b98ae" markerEnd="url(#ah)" fill="none" />}
          </g>
        );
      })}
      <rect x={decX} y={y(-1) - 16 + 2} width={bw} height={30} rx={5} fill="#16213a" stroke="#33425f" />
      <text x={decX + 10} y={y(-1) + 4} fill="#e6edf7" fontSize="12" fontFamily="monospace">ConvT → tanh  3×256²</text>
      <path d={`M${decX + 30} ${y(0) - 16} V${y(-1) + 16}`} stroke="#8b98ae" markerEnd="url(#ah)" fill="none" />
      <text x={encX} y={488} fill="#8b98ae" fontSize="11">Every encoder block: Conv 4×4 stride 2. Cyan: attention-gated skip (Oktay et al.) · dashed: plain concatenation · d4 self-attention (SAGAN)</text>
    </svg>
  );
}

const LOSSES = [
  ["LSGAN adversarial", "1", "Least-squares GAN loss on each discriminator scale’s patch output"],
  ["Feature matching", "10", "L1 between discriminator features of real and generated pairs (Pix2PixHD stabiliser)"],
  ["L1 reconstruction", "50", "Pixel-wise L1 to the paired optical image"],
  ["VGG19 perceptual", "10", "Weighted L1 over five VGG19 feature layers (ImageNet weights, frozen)"],
  ["SSIM", "10", "1 − SSIM (11×11 Gaussian window), implemented from scratch"],
];

export function Architecture() {
  return (
    <div className="space-y-5">
      <header>
        <p className="font-mono text-xs uppercase tracking-[0.14em] text-cyan">Model architecture</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">Conditional GAN: Attention U-Net generator, multi-scale PatchGAN critic</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">Drawn from the layer definitions in <code className="font-mono text-cyan-soft">models.py</code> (copied verbatim from the original notebook). 42,497,640 generator parameters; 8,297,283 across the three discriminators.</p>
      </header>
      <Panel title="Generator (used at inference)">
        <div className="overflow-x-auto" tabIndex={0} aria-label="Generator diagram, scrollable"><Generator /></div>
      </Panel>
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Discriminator (training only)">
          <p className="text-sm text-ink-2">Three identical PatchGANs see the concatenated <span className="font-mono">[SAR, optical]</span> pair (4 channels) at 256², 128² and 64² (3×3 average-pool downsampling). Each has three stride-2 conv blocks and one stride-1 block (InstanceNorm after the first, LeakyReLU throughout), then a 1-channel patch map. Intermediate features feed the feature-matching loss.</p>
          <div className="mt-4 grid grid-cols-3 gap-2 text-center font-mono text-xs">
            {["D₁ · 256²", "D₂ · 128²", "D₃ · 64²"].map((d) => <div key={d} className="rounded border border-edge bg-panel-2 px-2 py-3 text-ink-2">{d}</div>)}
          </div>
        </Panel>
        <Panel title="Training objective (generator)">
          <table className="w-full text-left text-sm">
            <thead><tr className="text-xs text-dim"><th className="pb-2 font-medium">Term</th><th className="pb-2 font-medium">λ</th><th className="pb-2 font-medium">What it does</th></tr></thead>
            <tbody>{LOSSES.map(([n, l, d]) => <tr key={n} className="border-t border-edge align-top"><td className="py-2 pr-3 text-ink">{n}</td><td className="py-2 pr-3 font-mono text-cyan-soft">{l}</td><td className="py-2 text-ink-2">{d}</td></tr>)}</tbody>
          </table>
          <p className="mt-3 text-xs text-dim">Adam (lr 2e-4, β 0.5/0.999), linear LR decay over the second half of 60 epochs, mixed precision on GPU.</p>
        </Panel>
      </div>
      <Panel title="Why InstanceNorm, not BatchNorm">
        <p className="text-sm text-ink-2">The original author documents that an earlier BatchNorm version silently corrupted its running statistics with NaN during training, which only appeared at evaluation time. InstanceNorm keeps no running statistics, so train and eval behaviour match. The notebook keeps a recovery cell for older checkpoints.</p>
      </Panel>
    </div>
  );
}

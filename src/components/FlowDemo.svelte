<script lang="ts">
  import { onMount } from "svelte";

  let duration = 0.55; // 0..1 slider
  let playing = true;

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;

  // Two panels: Flow Matching (curved) vs Rectified Flow (straight-ish)
  const W = 860;
  const H = 340;

  type Particle = {
    x: number;
    y: number;
    x0: number;
    y0: number;
    x1: number;
    y1: number;
    phase: number;
  };

  let left: Particle[] = [];
  let right: Particle[] = [];

  function randn() {
    // Box-Muller
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function reset() {
    const n = 170;
    left = [];
    right = [];

    // source cluster (top-ish), target cluster (bottom-ish)
    const srcCx = W * 0.25,
      srcCy = H * 0.3;
    const tgtCx = W * 0.25,
      tgtCy = H * 0.76;

    const srcCx2 = W * 0.75,
      srcCy2 = H * 0.3;
    const tgtCx2 = W * 0.75,
      tgtCy2 = H * 0.76;

    for (let i = 0; i < n; i++) {
      // left panel particles
      const x0 = srcCx + randn() * 18;
      const y0 = srcCy + randn() * 18;
      const x1 = tgtCx + randn() * 40;
      const y1 = tgtCy + randn() * 16;

      left.push({ x: x0, y: y0, x0, y0, x1, y1, phase: Math.random() });

      // right panel particles
      const x0r = srcCx2 + randn() * 18;
      const y0r = srcCy2 + randn() * 18;
      const x1r = tgtCx2 + randn() * 40;
      const y1r = tgtCy2 + randn() * 16;

      right.push({ x: x0r, y: y0r, x0: x0r, y0: y0r, x1: x1r, y1: y1r, phase: Math.random() });
    }
  }

  function ease(t: number) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }

  function bezier(p0: number, p1: number, p2: number, p3: number, t: number) {
    const u = 1 - t;
    return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
  }

  function drawAxes() {
    ctx.save();
    ctx.clearRect(0, 0, W, H);

    // background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);

    // divider
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(W / 2, 10);
    ctx.lineTo(W / 2, H - 10);
    ctx.stroke();

    ctx.fillStyle = "#111827";
    ctx.font = "700 20px ui-sans-serif, system-ui";
    ctx.textAlign = "center";
    ctx.fillText("Flow Matching", W * 0.25, 34);
    ctx.fillText("Rectified Flow", W * 0.75, 34);

    ctx.fillStyle = "#6b7280";
    ctx.font = "400 14px ui-sans-serif, system-ui";
    ctx.fillText("Curved Paths Flow Slow", W * 0.25, 56);
    ctx.fillText("Straight Paths Flow Fast", W * 0.75, 56);

    // faint vector field arrows (static)
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    for (let px = 70; px <= W - 70; px += 80) {
      for (let py = 92; py <= H - 55; py += 70) {
        const isLeft = px < W / 2;
        const dx = isLeft ? 14 : 10;
        const dy = isLeft ? 6 : 14;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + dx, py + dy);
        ctx.stroke();

        // arrow head
        const ax = px + dx,
          ay = py + dy;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - 6, ay - 1);
        ctx.lineTo(ax - 2, ay - 6);
        ctx.closePath();
        ctx.fillStyle = "#f59e0b";
        ctx.fill();
      }
    }

    ctx.restore();
  }

  function drawParticles(t01: number) {
    const t = ease(t01);

    // left: curved trajectories via cubic Bezier
    for (const p of left) {
      const bend = 80 * (p.phase - 0.5);
      const cx1 = p.x0 + bend;
      const cy1 = p.y0 + 70;
      const cx2 = p.x1 - bend;
      const cy2 = p.y1 - 70;

      p.x = bezier(p.x0, cx1, cx2, p.x1, t);
      p.y = bezier(p.y0, cy1, cy2, p.y1, t);

      ctx.fillStyle = "rgba(59,130,246,0.35)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // right: mostly straight trajectories
    for (const p of right) {
      const j = 10 * Math.sin(t * 6.283 + p.phase * 10);
      p.x = p.x0 + (p.x1 - p.x0) * t + j * 0.2;
      p.y = p.y0 + (p.y1 - p.y0) * t;

      ctx.fillStyle = "rgba(59,130,246,0.35)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  let t01 = 0;
  let last = 0;

  function loop(ts: number) {
    if (!playing) {
      requestAnimationFrame(loop);
      return;
    }
    if (!last) last = ts;
    const dt = Math.min(0.05, (ts - last) / 1000);
    last = ts;

    const speed = 0.5 + (1 - duration) * 2.2; // 0..1 -> slow..fast
    t01 += dt * speed;
    if (t01 > 1.0) {
      t01 = 0;
      reset();
    }

    drawAxes();
    drawParticles(t01);

    requestAnimationFrame(loop);
  }

  function toggle() {
    playing = !playing;
    if (playing) last = 0;
  }

  onMount(() => {
    ctx = canvas.getContext("2d")!;
    reset();
    requestAnimationFrame(loop);
  });
</script>

<div class="wrap">
  <canvas bind:this={canvas} width={W} height={H} class="cv"></canvas>

  <div class="controls">
    <button class="btn" on:click={toggle} aria-label="Play/Pause">
      {playing ? "⏸" : "▶"}
    </button>

    <input
      class="range"
      type="range"
      min="0"
      max="1"
      step="0.01"
      bind:value={duration}
      aria-label="Sampling Duration"
    />
    <div class="label">Sampling Duration</div>

    <button
      class="btn secondary"
      on:click={() => {
        t01 = 0;
        reset();
      }}
      aria-label="Reset"
    >
      ↻
    </button>
  </div>
</div>

<style>
  .wrap {
    width: 100%;
  }
  .cv {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    background: white;
  }

  .controls {
    display: grid;
    grid-template-columns: 48px 1fr auto 48px;
    align-items: center;
    gap: 10px;
    padding: 10px 4px 0;
  }

  .btn {
    border: 1px solid #e5e7eb;
    background: white;
    border-radius: 10px;
    height: 40px;
    width: 48px;
    cursor: pointer;
    font-size: 16px;
  }
  .btn:hover {
    background: #f9fafb;
  }

  .btn.secondary {
    color: #6b7280;
  }

  .range {
    width: 100%;
  }

  .label {
    color: #9ca3af;
    font-size: 13px;
    white-space: nowrap;
    padding-right: 4px;
  }
</style>

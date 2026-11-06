<script lang="ts">
  import { onMount } from "svelte";

  // Camera motion. Image x points right, image y points down and Z points forward.
  let vx = 0, vy = 0, vz = 0.6;
  let wx = 0, wy = 0, wz = 0;
  let scene = "wall";
  let probe: { x: number; y: number } | null = null;

  let canvas: HTMLCanvasElement;
  const W = 760, H = 500;
  const hx = 1, hy = (hx * H) / W;
  const NX = 15, NY = 10;

  const presets: Record<string, number[]> = {
    Forward: [0, 0, 0.6, 0, 0, 0],
    Sideways: [0.6, 0, 0, 0, 0, 0],
    Pan: [0, 0, 0, 0, 0.5, 0],
    Tilt: [0, 0, 0, 0.5, 0, 0],
    Roll: [0, 0, 0, 0, 0, 0.8],
    "Forward + pan": [0, 0, 0.6, 0, 0.35, 0],
    "Curved path": [0.25, 0, 0.5, 0, -0.3, 0.3],
  };

  function apply(p: number[]) {
    [vx, vy, vz, wx, wy, wz] = p;
  }

  // Scene depth. The ground plane recedes toward the horizon at y = 0.
  function depth(x: number, y: number, s: string) {
    if (s === "ground") return y > 0.02 ? 0.5 / y : 25;
    if (s === "objects") {
      if (Math.abs(x + 0.45) < 0.22 && Math.abs(y) < 0.3) return 0.6;
      if (Math.abs(x - 0.45) < 0.22 && Math.abs(y) < 0.3) return 1.5;
      return 5;
    }
    return 1;
  }

  // The motion field equation.
  function flow(x: number, y: number, s: string) {
    const Z = depth(x, y, s);
    const u = (x * vz - vx) / Z + x * y * wx - (1 + x * x) * wy + y * wz;
    const v = (y * vz - vy) / Z + (1 + y * y) * wx - x * y * wy - x * wz;
    return { u, v, Z };
  }

  const phases = Array.from({ length: NX * NY }, () => Math.random());
  const toX = (x: number) => ((x / hx + 1) * W) / 2;
  const toY = (y: number) => ((y / hy + 1) * H) / 2;

  function arrow(c: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) {
    const a = Math.atan2(y1 - y0, x1 - x0), L = Math.hypot(x1 - x0, y1 - y0);
    const hl = Math.min(11, L * 0.55), hw = hl * 0.48;
    const bx = x1 - hl * Math.cos(a), by = y1 - hl * Math.sin(a);
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(bx, by); c.stroke();
    c.beginPath(); c.moveTo(x1, y1);
    c.lineTo(bx - hw * Math.sin(a), by + hw * Math.cos(a));
    c.lineTo(bx + hw * Math.sin(a), by - hw * Math.cos(a));
    c.closePath(); c.fill();
  }

  function draw(t: number) {
    const c = canvas.getContext("2d");
    if (!c) return;
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, W, H);
    if (scene === "ground") {
      c.fillStyle = "#EEF2F5"; c.fillRect(0, 0, W, H / 2);
      c.fillStyle = "#F1EEE2"; c.fillRect(0, H / 2, W, H / 2);
    }
    if (scene === "objects") {
      c.fillStyle = "#F1EEE2";
      c.fillRect(toX(-0.67), toY(-0.3), toX(-0.23) - toX(-0.67), toY(0.3) - toY(-0.3));
      c.fillRect(toX(0.23), toY(-0.3), toX(0.67) - toX(0.23), toY(0.3) - toY(-0.3));
    }

    const cell = (2 * hx) / NX;
    const hasT = Math.hypot(vx, vy, vz) > 0.001;
    const hasR = Math.hypot(wx, wy, wz) > 0.001;
    const color = hasT && hasR ? "#3B3B33" : hasR ? "#1F6FB2" : "#A63D2F";
    c.lineWidth = 2.6; c.lineCap = "round";

    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const x = ((i + 0.5) / NX) * 2 * hx - hx, y = ((j + 0.5) / NY) * 2 * hy - hy;
        const { u, v } = flow(x, y, scene);
        const sp = Math.hypot(u, v);
        if (sp < 0.004) continue;
        const L = Math.min(sp * 0.2, cell * 0.8);
        const s = (t / 1400 + phases[j * NX + i]) % 1;
        c.globalAlpha = Math.min(1, Math.min(s, 1 - s) / 0.12);
        // Each arrow only ever moves forward along its own vector.
        const off = (s - 0.5) * 0.55 * L;
        const dx = u / sp, dy = v / sp;
        const tx = x + dx * (off - L / 2), ty = y + dy * (off - L / 2);
        c.strokeStyle = color; c.fillStyle = color;
        arrow(c, toX(tx), toY(ty), toX(tx + dx * L), toY(ty + dy * L));
      }
    }
    c.globalAlpha = 1;

    if (Math.abs(vz) > 0.001 && !hasR) {
      const fx = vx / vz, fy = vy / vz;
      if (Math.abs(fx) <= hx && Math.abs(fy) <= hy) {
        c.fillStyle = "#1D1E00";
        c.beginPath(); c.arc(toX(fx), toY(fy), 5.5, 0, 2 * Math.PI); c.fill();
        c.font = "600 14px system-ui, sans-serif";
        c.fillText(vz > 0 ? "FOE" : "FOC", toX(fx) + 10, toY(fy) + 19);
      }
    }
    if (probe) {
      c.strokeStyle = "#1D1E00"; c.lineWidth = 1.4;
      c.beginPath(); c.arc(toX(probe.x), toY(probe.y), 8, 0, 2 * Math.PI); c.stroke();
    }
  }

  function move(e: PointerEvent) {
    const r = canvas.getBoundingClientRect();
    probe = {
      x: (((e.clientX - r.left) / r.width) * 2 - 1) * hx,
      y: (((e.clientY - r.top) / r.height) * 2 - 1) * hy,
    };
  }

  const fmt = (n: number, d = 2) => (Math.abs(n) < 0.005 ? "0" : n.toFixed(d));

  $: point = probe ?? { x: 0, y: 0 };
  $: _dep = [vx, vy, vz, wx, wy, wz, scene];
  $: reading = (_dep, flow(point.x, point.y, scene));
  $: tau = vz > 0.001 ? reading.Z / vz : null;
  $: foe = Math.abs(vz) > 0.001 ? `(${fmt(vx / vz)}, ${fmt(vy / vz)})` : "at infinity";

  onMount(() => {
    let raf = 0, visible = true;
    const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting));
    io.observe(canvas);
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const loop = (t: number) => {
      if (visible) draw(still ? 700 : t);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); io.disconnect(); };
  });
</script>

<div class="fx">
  <div class="fx-presets">
    {#each Object.keys(presets) as name}
      <button type="button" on:click={() => apply(presets[name])}>{name}</button>
    {/each}
  </div>

  <canvas
    bind:this={canvas}
    width={W}
    height={H}
    on:pointermove={move}
    on:pointerleave={() => (probe = null)}
    aria-label="Live motion field computed from the selected camera motion"
  ></canvas>

  <div class="fx-controls">
    <label class="fx-select">
      <span class="fx-title">Scene</span>
      <select bind:value={scene}>
        <option value="wall">Flat wall</option>
        <option value="ground">Ground plane</option>
        <option value="objects">Two objects at different depths</option>
      </select>
    </label>
  </div>

  <dl class="fx-readout">
    <div><dt>{probe ? "Pointer" : "Image center"}</dt><dd>({fmt(point.x)}, {fmt(point.y)})</dd></div>
    <div><dt>Depth Z</dt><dd>{fmt(reading.Z, 1)}</dd></div>
    <div><dt>Flow</dt><dd>({fmt(reading.u)}, {fmt(reading.v)})</dd></div>
    <div><dt>FOE</dt><dd>{foe}</dd></div>
    <div><dt>Time to contact</dt><dd>{tau === null ? "none" : fmt(tau, 1)}</dd></div>
  </dl>
</div>

<style>
  .fx {
    max-width: 108ch;
    margin: 22px auto;
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: 12px;
  }
  .fx-presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
  button {
    padding: 5px 11px;
    font: inherit;
    font-size: 13px;
    color: inherit;
    cursor: pointer;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 999px;
  }
  button:hover { border-color: currentColor; }
  canvas {
    display: block;
    width: 100%;
    height: auto;
    touch-action: pan-y;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .fx-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: 12px 20px;
    margin-top: 14px;
    font-size: 14px;
  }
  .fx-title { display: block; margin-bottom: 4px; font-size: 13px; font-weight: 700; }
  .fx-select { flex: 0 1 300px; }
  .fx-select select {
    width: 100%;
    padding: 5px;
    font: inherit;
    font-size: 13px;
    color: inherit;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  select option { color: #111111; }
  .fx-readout {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin: 14px 0 0;
    font-size: 13px;
  }
  .fx-readout dt { color: var(--muted); }
  .fx-readout dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }
  @media (max-width: 640px) {
    .fx-readout { grid-template-columns: repeat(2, 1fr); }
  }
</style>

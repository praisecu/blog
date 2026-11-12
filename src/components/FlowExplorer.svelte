<script lang="ts">
  import { onMount } from "svelte";

  // Camera motion. Image x points right, image y points down and Z points forward.
  let vx = 0, vy = 0, vz = 0.6;
  let wx = 0, wy = 0, wz = 0;
  let scene = "wall";
  const sceneOptions = [
    ["wall", "Flat wall"],
    ["ground", "Ground plane"],
    ["objects", "Objects at two depths"],
  ];
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

  let active = "Forward";
  function apply(name: string) {
    active = name;
    [vx, vy, vz, wx, wy, wz] = presets[name];
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

  // One fade front that sweeps across the field in the flow direction, so arrows fade
  // systematically (a line, ring or spoke at a time) instead of at random.
  function sweep(s: string) {
    const pts = [];
    for (let j = 0; j < NY; j++)
      for (let i = 0; i < NX; i++) {
        const x = ((i + 0.5) / NX) * 2 * hx - hx, y = ((j + 0.5) / NY) * 2 * hy - hy;
        const { u, v } = flow(x, y, s);
        pts.push({ x, y, u, v, sp: Math.hypot(u, v) });
      }
    const cell = (2 * hx) / NX;
    const meanSp = pts.reduce((a, p) => a + p.sp, 0) / pts.length || 1;
    const mu = pts.reduce((a, p) => a + p.u, 0) / pts.length;
    const mv = pts.reduce((a, p) => a + p.v, 0) / pts.length;
    if (Math.hypot(mu, mv) > 0.5 * meanSp) {
      // Nearly uniform: a straight line moving along the mean flow direction
      const m = Math.hypot(mu, mv), dx = mu / m, dy = mv / m;
      const span = 2 * (hx * Math.abs(dx) + hy * Math.abs(dy));
      return { phase: pts.map((p) => -(p.x * dx + p.y * dy) / span), fade: cell / span };
    }
    // Otherwise centre the front on the slowest point (the FOE or centre of rotation)
    const c = pts.reduce((a, p) => (p.sp < a.sp ? p : a), pts[0]);
    let rad = 0, tan = 0;
    for (const p of pts) {
      const rx = p.x - c.x, ry = p.y - c.y, r = Math.hypot(rx, ry) || 1;
      rad += (rx * p.u + ry * p.v) / r;
      tan += (rx * p.v - ry * p.u) / r;
    }
    if (Math.abs(rad) >= Math.abs(tan)) {
      // Expanding (or contracting) ring
      const rs = pts.map((p) => Math.hypot(p.x - c.x, p.y - c.y));
      const span = Math.max(...rs) + cell;
      return { phase: rs.map((r) => -Math.sign(rad) * r / span), fade: cell / span };
    }
    // Rotating spoke
    return {
      phase: pts.map((p) => Math.sign(tan) * -Math.atan2(p.y - c.y, p.x - c.x) / (2 * Math.PI)),
      fade: 1 / 14,
    };
  }

  function frontFor(s: string) {
    const f = sweep(s);
    const speeds: number[] = [];
    for (let j = 0; j < NY; j++)
      for (let i = 0; i < NX; i++) {
        const { u, v } = flow(((i + 0.5) / NX) * 2 * hx - hx, ((j + 0.5) / NY) * 2 * hy - hy, s);
        speeds.push(Math.hypot(u, v));
      }
    speeds.sort((a, b) => a - b);
    return { ...f, speeds };
  }

  $: front = (vx, vy, vz, wx, wy, wz, frontFor(scene));

  const toX = (x: number) => ((x / hx + 1) * W) / 2;
  const toY = (y: number) => ((y / hy + 1) * H) / 2;

  function arrow(c: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) {
    const a = Math.atan2(y1 - y0, x1 - x0), L = Math.hypot(x1 - x0, y1 - y0);
    const hl = Math.min(10, L * 0.45), hw = hl * 0.62;
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
    const dpr = canvas.width / W;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, W, H);
    if (scene === "ground") {
      const sky = c.createLinearGradient(0, 0, 0, H / 2);
      sky.addColorStop(0, "#E6EDF3"); sky.addColorStop(1, "#F5F8FA");
      c.fillStyle = sky; c.fillRect(0, 0, W, H / 2);
      const gnd = c.createLinearGradient(0, H / 2, 0, H);
      gnd.addColorStop(0, "#F6F2E7"); gnd.addColorStop(1, "#EAE2CC");
      c.fillStyle = gnd; c.fillRect(0, H / 2, W, H / 2);
      c.strokeStyle = "rgba(29, 30, 0, 0.18)"; c.lineWidth = 1;
      c.beginPath(); c.moveTo(0, H / 2); c.lineTo(W, H / 2); c.stroke();
    }
    if (scene === "objects") {
      c.font = "600 12px system-ui, sans-serif";
      for (const [x0, x1, label] of [[-0.67, -0.23, "Z = 0.6"], [0.23, 0.67, "Z = 1.5"]] as const) {
        c.fillStyle = "#F3EEDF";
        c.beginPath();
        c.roundRect(toX(x0), toY(-0.3), toX(x1) - toX(x0), toY(0.3) - toY(-0.3), 12);
        c.fill();
        c.fillStyle = "rgba(29, 30, 0, 0.45)";
        c.fillText(label, toX(x0) + 10, toY(-0.3) + 18);
      }
      c.fillStyle = "rgba(29, 30, 0, 0.35)";
      c.fillText("Background Z = 5", 12, H - 12);
    }

    const cell = (2 * hx) / NX;
    const hasT = Math.hypot(vx, vy, vz) > 0.001;
    const hasR = Math.hypot(wx, wy, wz) > 0.001;
    const color = hasT && hasR ? "#3B3B33" : hasR ? "#1F6FB2" : "#A63D2F";
    c.lineWidth = 2.4; c.lineCap = "round"; c.lineJoin = "round";

    // Arrow lengths are scaled to this field's own speed range, so slow and fast regions
    // both read clearly, and every arrow stays inside its own cell (no overlaps).
    const speeds = front.speeds;
    const ref = speeds.length ? speeds[Math.floor(0.95 * (speeds.length - 1))] || 1 : 1;
    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const x = ((i + 0.5) / NX) * 2 * hx - hx, y = ((j + 0.5) / NY) * 2 * hy - hy;
        const { u, v } = flow(x, y, scene);
        const sp = Math.hypot(u, v);
        const rel = Math.min(1, sp / ref);
        if (rel < 0.08) continue; // too slow to show: no arrow rather than a dot
        const L = cell * 0.62 * Math.pow(rel, 0.6);
        const s = (((t / 4000 + front.phase[j * NX + i]) % 1) + 1) % 1;
        const a = Math.min(1, Math.min(s, 1 - s) / front.fade);
        c.globalAlpha = a * a * (3 - 2 * a);
        // Each arrow only ever moves forward along its own vector, within its cell.
        const off = (s - 0.5) * 0.18 * cell;
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
    // Render at device resolution so arrows stay sharp on high-DPI screens
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
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
      <button
        type="button"
        class:active={active === name}
        aria-pressed={active === name}
        on:click={() => apply(name)}
      >{name}</button>
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
    <div class="fx-scene">
      <span class="fx-title" id="fx-scene-label">Scene</span>
      <div class="fx-seg" role="radiogroup" aria-labelledby="fx-scene-label">
        {#each sceneOptions as [value, label]}
          <button
            type="button"
            role="radio"
            aria-checked={scene === value}
            class:active={scene === value}
            on:click={() => (scene = value)}
          >{label}</button>
        {/each}
      </div>
    </div>
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
    --fx-radius: 18px;
    --fx-tint: color-mix(in srgb, currentColor 5%, transparent);
    max-width: 108ch;
    margin: 24px auto;
    padding: 16px;
    border: 1px solid var(--border);
    border-radius: var(--fx-radius);
    background: color-mix(in srgb, currentColor 2%, transparent);
  }
  button {
    font: inherit;
    font-size: 13px;
    color: inherit;
    cursor: pointer;
    background: transparent;
    -webkit-tap-highlight-color: transparent;
    transition: background-color 160ms ease, border-color 160ms ease, opacity 160ms ease, transform 120ms ease;
  }
  button:active { transform: scale(0.97); }
  button:focus { outline: none; }
  button:focus-visible { outline: 2px solid color-mix(in srgb, currentColor 55%, transparent); outline-offset: 2px; }

  /* Camera motion presets: pills with a clear active state */
  .fx-presets { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
  .fx-presets button {
    padding: 6px 14px;
    border: 1px solid var(--border);
    border-radius: 999px;
  }
  .fx-presets button:hover { border-color: color-mix(in srgb, currentColor 45%, transparent); }
  .fx-presets button.active {
    font-weight: 600;
    color: #ffffff;
    background: #a63d2f;
    border-color: #a63d2f;
  }

  canvas {
    display: block;
    width: 100%;
    height: auto;
    touch-action: pan-y;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06), 0 8px 24px -12px rgba(0, 0, 0, 0.18);
  }

  .fx-controls { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px 20px; margin-top: 16px; }
  .fx-title {
    display: block;
    margin-bottom: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
  }

  /* Scene: segmented control */
  .fx-seg {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 4px;
    padding: 4px;
    border-radius: 12px;
    background: var(--fx-tint);
    border: 1px solid var(--border);
  }
  .fx-seg button {
    padding: 6px 14px;
    border: 0;
    border-radius: 9px;
    opacity: 0.7;
  }
  .fx-seg button:hover { opacity: 1; }
  .fx-seg button.active {
    opacity: 1;
    font-weight: 600;
    background: color-mix(in srgb, currentColor 13%, transparent);
  }

  /* Readout: small cards */
  .fx-readout {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin: 16px 0 0;
  }
  .fx-readout > div {
    padding: 10px 12px;
    border-radius: 12px;
    background: var(--fx-tint);
  }
  .fx-readout dt {
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .fx-readout dd { margin: 2px 0 0; font-size: 14px; font-weight: 600; font-variant-numeric: tabular-nums; }

  @media (max-width: 640px) {
    .fx { padding: 12px; }
    .fx-readout { grid-template-columns: repeat(2, 1fr); }
  }
  @media (prefers-reduced-motion: reduce) {
    button { transition: none; }
    button:active { transform: none; }
  }
</style>

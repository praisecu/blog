<script lang="ts">
  import { onMount } from "svelte";

  let canvas: HTMLCanvasElement;
  const W = 760, H = 330;
  const TARGET = 0.5;

  let state: "idle" | "diving" | "done" = "idle";
  let Z0 = 30, V = 15, tau0 = 2;
  let start = 0, raf = 0;
  let result = "";
  let history: { Z0: number; V: number; miss: number | null }[] = [];

  // A fixed ripple texture, repeated across scales so detail keeps appearing as it expands.
  const ripples = Array.from({ length: 170 }, (_, i) => ({
    x: (Math.random() - 0.5) * 2.4,
    y: (Math.random() - 0.5) * 2.4,
    r: 0.02 + Math.random() * 0.1,
    tone: i % 3,
  }));
  const tones = ["#C7D6E6", "#8FAAC6", "#E7EEF5"];

  function paint(scale: number) {
    const c = canvas.getContext("2d");
    if (!c) return;
    c.fillStyle = "#AFC4D9";
    c.fillRect(0, 0, W, H);
    const unit = H;
    for (const layer of [1 / 16, 1 / 4, 1, 4]) {
      const s = scale * layer;
      if (s < 0.05) continue;
      c.globalAlpha = Math.min(1, s * 2) * Math.min(1, 6 / s);
      for (const p of ripples) {
        const x = W / 2 + p.x * s * unit, y = H / 2 + p.y * s * unit, r = p.r * s * unit;
        if (x + 2 * r < 0 || x - 2 * r > W || y + r < 0 || y - r > H) continue;
        c.fillStyle = tones[p.tone];
        c.beginPath();
        c.ellipse(x, y, 2 * r, r * 0.8, 0, 0, 2 * Math.PI);
        c.fill();
      }
    }
    c.globalAlpha = 1;
  }

  function dive() {
    // Height and speed change each trial. Only their ratio reaches the image.
    Z0 = 15 + Math.random() * 45;
    tau0 = 1.8 + Math.random() * 1.7;
    V = Z0 / tau0;
    state = "diving";
    result = "";
    start = performance.now();
    cancelAnimationFrame(raf);
    const loop = (t: number) => {
      const el = (t - start) / 1000;
      if (el >= tau0) {
        finish(null);
        return;
      }
      // Image scale: Z0 / Z(t) = 1 / (1 - t / tau0), which depends on neither Z0 nor V alone.
      paint(1 / (1 - el / tau0));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
  }

  function finish(left: number | null) {
    cancelAnimationFrame(raf);
    state = "done";
    if (left === null) {
      result = "Too late: the camera reached the surface before you folded.";
      const c = canvas.getContext("2d");
      if (c) { c.fillStyle = "#6F8FAD"; c.fillRect(0, 0, W, H); }
    } else {
      const err = left - TARGET;
      result = `You folded ${left.toFixed(2)} s before contact, ${Math.abs(err).toFixed(2)} s ${err > 0 ? "early" : "late"}.`;
    }
    history = [{ Z0, V, miss: left === null ? null : left - TARGET }, ...history].slice(0, 4);
  }

  function fold() {
    if (state !== "diving") return;
    finish(tau0 - (performance.now() - start) / 1000);
  }

  onMount(() => {
    paint(1);
    return () => cancelAnimationFrame(raf);
  });
</script>

<div class="ct">
  <canvas bind:this={canvas} width={W} height={H} aria-label="View of a water surface expanding as the camera dives toward it"></canvas>

  <div class="ct-row">
    {#if state === "diving"}
      <button class="primary" type="button" on:click={fold}>Fold now</button>
    {:else}
      <button class="primary" type="button" on:click={dive}>{state === "idle" ? "Start dive" : "Dive again"}</button>
    {/if}
    <span class="ct-msg">{state === "diving" ? `Fold ${TARGET} s before contact.` : result || `Press start, then fold ${TARGET} s before contact.`}</span>
  </div>

  {#if history.length}
    <table>
      <thead><tr><th>Height</th><th>Speed</th><th>Time to contact at start</th><th>Your error</th></tr></thead>
      <tbody>
        {#each history as h}
          <tr>
            <td>{h.Z0.toFixed(0)} m</td>
            <td>{h.V.toFixed(0)} m/s</td>
            <td>{(h.Z0 / h.V).toFixed(1)} s</td>
            <td>{h.miss === null ? "missed" : `${h.miss > 0 ? "+" : "-"}${Math.abs(h.miss).toFixed(2)} s`}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>

<style>
  .ct {
    max-width: 108ch;
    margin: 22px auto;
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: 12px;
  }
  canvas {
    display: block;
    width: 100%;
    height: auto;
    border-radius: 8px;
  }
  .ct-row { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 12px; }
  .ct-msg { font-size: 14px; }
  button.primary {
    padding: 7px 16px;
    font: inherit;
    font-size: 14px;
    font-weight: 700;
    color: #1d1e00;
    cursor: pointer;
    background: #cfb87c;
    border: 0;
    border-radius: 999px;
  }
  table { width: 100%; margin: 12px 0 0; font-size: 13px; border-collapse: collapse; }
  th { color: var(--muted); font-weight: 600; text-align: left; }
  th, td { padding: 4px 8px 4px 0; border-top: 1px solid var(--border); }
</style>

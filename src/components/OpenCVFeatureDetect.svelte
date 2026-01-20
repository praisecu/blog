<script lang="ts">
  import { onMount } from "svelte";

  let status = "Idle.";
  let canvas: HTMLCanvasElement | null = null;
  let running = false;

  // GitHub Pages-safe base (Astro sets BASE_URL)
  const BASE = (import.meta as any).env?.BASE_URL ?? "/";

  // You must have BOTH files in: public/vendor/
  //   public/vendor/opencv.js
  //   public/vendor/opencv_js.wasm
  const OPENCV_JS = `${BASE}vendor/opencv.js`;
  const IMG_URL = `${BASE}images/optical-flow.png`;

  function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const t = window.setTimeout(() => reject(new Error(`${label} (timeout after ${ms} ms)`)), ms);
      p.then((v) => {
        clearTimeout(t);
        resolve(v);
      }).catch((e) => {
        clearTimeout(t);
        reject(e);
      });
    });
  }

  function loadScriptOnce(src: string, id: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const existing = document.getElementById(id) as HTMLScriptElement | null;
      if (existing) {
        existing.addEventListener("load", () => resolve(), { once: true });
        existing.addEventListener("error", () => reject(new Error(`Failed to load script: ${src}`)), { once: true });
        return;
      }

      const s = document.createElement("script");
      s.id = id;
      s.src = src;
      s.async = true;
      s.defer = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error(`Failed to load script: ${src}`));
      document.head.appendChild(s);
    });
  }

function waitForCVReady(): Promise<any> {
  return new Promise((resolve, reject) => {
    const w = window as any;

    // If already ready
    if (w.cv?.Mat) return resolve(w.cv);

    const mod = (globalThis as any).Module;

    if (!mod) {
      return reject(new Error("Module is not defined. OpenCV script likely not loaded."));
    }

    const prev = mod.onRuntimeInitialized;
    mod.onRuntimeInitialized = () => {
      try {
        if (typeof prev === "function") prev();
      } catch {}
      if (w.cv?.Mat) resolve(w.cv);
      else reject(new Error("Runtime initialized but window.cv.Mat is missing."));
    };

    // As a fallback, poll briefly for cv.Mat in case the callback already fired
    let tries = 0;
    const timer = window.setInterval(() => {
      tries++;
      if (w.cv?.Mat) {
        window.clearInterval(timer);
        resolve(w.cv);
      } else if (tries > 4000) {
        window.clearInterval(timer);
        reject(new Error("OpenCV init stalled (cv.Mat never appeared)."));
      }
    }, 10);
  });
}


  async function loadOpenCV(): Promise<any> {
    const w = window as any;
    if (w.cv?.Mat) return w.cv;

    status = "Configuring WASM path…";

    // Critical for split builds: tell Emscripten where to fetch the WASM.
    // You have public/vendor/opencv.wasm (NOT opencv_js.wasm).
    (globalThis as any).Module = (globalThis as any).Module ?? {};
    (globalThis as any).Module.locateFile = (path: string) => {
      if (path.endsWith(".wasm")) return `${BASE}vendor/opencv.wasm`;
      return `${BASE}vendor/${path}`;
    };

    status = "Loading OpenCV.js…";
    await loadScriptOnce(OPENCV_JS, "opencvjs-script");

    status = "Initializing OpenCV…";
    return await withTimeout(waitForCVReady(), 40000, "OpenCV runtime init stalled");
  }


  function loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Image failed to load: ${url}`));
      img.src = url;
    });
  }

  async function runFeatureDetect() {
    if (running) return;
    running = true;

    try {
      status = "Starting…";
      if (!canvas) throw new Error("Canvas not bound.");

      const cv = await loadOpenCV();

      status = "Downloading image…";
      const img = await withTimeout(loadImage(IMG_URL), 15000, "Image load stalled");

      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas 2D context unavailable.");

      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      if (!canvas.width || !canvas.height) throw new Error("Image has zero width/height.");

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      status = "Running Shi–Tomasi (goodFeaturesToTrack)…";

      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

      const corners = new cv.Mat();
      const mask = new cv.Mat();

      cv.goodFeaturesToTrack(gray, corners, 400, 0.01, 8, mask, 3, false, 0.04);

      status = `Drawing ${corners.rows} corners…`;
      ctx.lineWidth = 2;

      for (let i = 0; i < corners.rows; i++) {
        const x = corners.data32F[i * 2 + 0];
        const y = corners.data32F[i * 2 + 1];
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.stroke();
      }

      src.delete();
      gray.delete();
      corners.delete();
      mask.delete();

      status = "Done.";
    } catch (e: any) {
      console.error(e);
      status = `Failed: ${e?.message ?? String(e)}`;
    } finally {
      running = false;
    }
  }

  onMount(() => {
    void runFeatureDetect();
  });
</script>

<div class="figure-card">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
    <div style="font-weight:600;">OpenCV Feature Detection (Shi–Tomasi)</div>
    <button type="button" on:click={runFeatureDetect} disabled={running}>
      {running ? "Running…" : "Re-run"}
    </button>
  </div>

  <div style="margin-top:8px; opacity:0.85; font-size:0.95em;">
    {status}
  </div>

  <div style="margin-top:12px;">
    <canvas bind:this={canvas} style="max-width:100%; height:auto; display:block;"></canvas>
  </div>

  <div class="caption" style="margin-top:10px;">
    <b>Figure:</b> Shi–Tomasi corners detected using OpenCV.js and rendered on a canvas.
  </div>
</div>

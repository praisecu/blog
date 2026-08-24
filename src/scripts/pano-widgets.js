// Interactive widgets for the panorama-stitching tutorial.
// Data (corners, matches, RANSAC iterations) is precomputed by
// public/code/gen_pano.py and bundled from src/data/pano-data.json.
// Every widget is optional: if its container is missing, it is skipped.

export function initPanoWidgets(D) {
  const VW = D.vw, VH = D.vh;

  const $ = (id) => document.getElementById(id);

  const loadImg = (src) =>
    new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = rej;
      im.src = src;
    });

  function makeCanvas(container, w, h) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    c.style.maxWidth = '100%';
    c.style.maxHeight = '70vh';      /* keep tall (portrait) canvases from filling the screen */
    c.style.width = 'auto';
    c.style.height = 'auto';
    c.style.display = 'block';
    c.style.margin = '0 auto';
    c.style.borderRadius = '6px';
    container.appendChild(c);
    return c;
  }

  function dot(ctx, x, y, r, fill) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    ctx.stroke();
  }

  /* ---------------- Widget 1: Harris threshold ---------------- */
  function initHarris() {
    const box = $('w-harris');
    if (!box) return;
    loadImg(box.dataset.img).then((im) => {
      const ctx = makeCanvas(box.querySelector('.pano-stage'), VW, VH).getContext('2d');
      const slider = box.querySelector('input[type=range]');
      const readout = box.querySelector('.pano-readout');

      const draw = () => {
        // slider 0..100 -> log10(threshold/Rmax) from -5.5 (keep ~all) to -0.5
        const t = -5.5 + (slider.value / 100) * 5.0;
        ctx.drawImage(im, 0, 0);
        let n = 0;
        for (const [x, y, s] of D.corners) {
          if (s >= t) { dot(ctx, x, y, 4, '#ffd166'); n++; }
        }
        readout.innerHTML =
          'threshold: <b>10<sup>' + t.toFixed(1) +
          '</sup> &times; R<sub>max</sub></b> &nbsp;&rarr;&nbsp; <b>' + n + '</b> corners survive';
      };
      slider.addEventListener('input', draw);
      draw();
    });
  }

  /* ---------------- Widget 2: ANMS vs top-N ---------------- */
  function initAnms() {
    const box = $('w-anms');
    if (!box) return;
    loadImg(box.dataset.img).then((im) => {
      const ctx = makeCanvas(box.querySelector('.pano-stage'), VW, VH).getContext('2d');
      const slider = box.querySelector('input[type=range]');
      const readout = box.querySelector('.pano-readout');

      const mode = () => {
        const r = box.querySelector('input[name=anmsmode]:checked');
        return r ? r.value : 'anms';
      };
      const draw = () => {
        const N = parseInt(slider.value, 10);
        ctx.drawImage(im, 0, 0);
        if (mode() === 'top') {
          for (let i = 0; i < N; i++) dot(ctx, D.corners[i][0], D.corners[i][1], 4.5, '#ff5964');
        } else {
          for (let j = 0; j < N; j++) {
            const k = D.anmsOrder[j];
            dot(ctx, D.corners[k][0], D.corners[k][1], 4.5, '#35d07f');
          }
        }
        readout.innerHTML =
          '<b>' + N + '</b> corners, picked by <b>' +
          (mode() === 'top' ? 'raw strength (watch them clump!)' : 'ANMS (spread out)') + '</b>';
      };
      slider.addEventListener('input', draw);
      box.querySelectorAll('input[name=anmsmode]').forEach((r) => r.addEventListener('change', draw));
      draw();
    });
  }

  /* ---------------- shared match drawing ---------------- */
  function drawMatchBase(ctx, imL, imR) {
    ctx.drawImage(imL, 0, 0);
    ctx.drawImage(imR, VW, 0);
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.fillRect(VW - 2, 0, 4, VH);
  }
  function matchLine(ctx, m1, m2, color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 1.6;
    ctx.beginPath();
    ctx.moveTo(m1[0], m1[1]);
    ctx.lineTo(m2[0] + VW, m2[1]);
    ctx.stroke();
    dot(ctx, m1[0], m1[1], 4, color);
    dot(ctx, m2[0] + VW, m2[1], 4, color);
  }

  /* ---------------- Widget 3: ratio-test matching ---------------- */
  function initMatch() {
    const box = $('w-match');
    if (!box) return;
    Promise.all([loadImg(box.dataset.imgL), loadImg(box.dataset.imgR)]).then((ims) => {
      const ctx = makeCanvas(box.querySelector('.pano-stage'), VW * 2, VH).getContext('2d');
      const slider = box.querySelector('input[type=range]');
      const reveal = box.querySelector('input[type=checkbox]');
      const readout = box.querySelector('.pano-readout');
      const M = D.matches;

      const draw = () => {
        const thr = slider.value / 100;
        drawMatchBase(ctx, ims[0], ims[1]);
        let kept = 0, good = 0, bad = 0;
        for (let i = 0; i < M.ratio.length; i++) {
          if (M.ratio[i] >= thr) continue;
          kept++;
          let col = '#ffb020';
          if (reveal.checked) {
            col = M.inlier[i] ? '#2ecc71' : '#e74c3c';
            if (M.inlier[i]) good++; else bad++;
          }
          matchLine(ctx, M.p1[i], M.p2[i], col);
        }
        let msg = 'ratio &lt; <b>' + thr.toFixed(2) + '</b> &nbsp;&rarr;&nbsp; <b>' + kept + '</b> matches kept';
        if (reveal.checked) {
          msg += ' &nbsp;(<b style="color:#2ecc71">' + good + ' correct</b>, <b style="color:#e74c3c">' + bad + ' outliers</b>)';
        }
        readout.innerHTML = msg;
      };
      slider.addEventListener('input', draw);
      reveal.addEventListener('change', draw);
      draw();
    });
  }

  /* ---------------- Widget 4: RANSAC slot machine ---------------- */
  function initRansac() {
    const box = $('w-ransac');
    if (!box) return;
    Promise.all([loadImg(box.dataset.imgL), loadImg(box.dataset.imgR)]).then((ims) => {
      const ctx = makeCanvas(box.querySelector('.pano-stage'), VW * 2, VH).getContext('2d');
      const readout = box.querySelector('.pano-readout');
      const M = D.matches, S = D.strictIdx, R = D.ransac;
      let order = [], round = 0, best = 0, cur = null;

      const reshuffle = () => {
        order = R.iters.map((_, i) => i);
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [order[i], order[j]] = [order[j], order[i]];
        }
      };
      reshuffle();

      const draw = () => {
        drawMatchBase(ctx, ims[0], ims[1]);
        for (let i = 0; i < S.length; i++) {
          const mi = S[i];
          let col = 'rgba(200,200,200,0.35)';
          if (cur && cur.mask.charAt(i) === '1') col = 'rgba(46,204,113,0.85)';
          if (!cur) col = 'rgba(255,176,32,0.45)';
          matchLine(ctx, M.p1[mi], M.p2[mi], col, 1.4);
        }
        if (cur) {
          for (const k of cur.pick) matchLine(ctx, M.p1[S[k]], M.p2[S[k]], '#ffd166', 4);
          const verdict =
            cur.count >= R.bestPossible * 0.9 ? ' &nbsp;Near-optimal consensus.' :
            cur.count < 20 ? ' &nbsp;An outlier in the sample spoiled this fit.' : '';
          readout.innerHTML =
            'Round <b>' + round + '</b>: sampled <b style="color:#d4a017">4</b> matches, fit a homography &rarr; <b style="color:#2ecc71">' +
            cur.count + '</b> of ' + S.length + ' matches agree. Best so far: <b>' + best + '</b>.' + verdict;
        } else {
          readout.innerHTML =
            'Run an iteration: each of the <b>' + S.length + '</b> candidate matches votes on the fitted homography. (Best possible consensus: <b>' +
            R.bestPossible + '</b>.)';
        }
      };
      const spin = () => {
        if (round >= order.length) reshuffle();
        cur = R.iters[order[round % order.length]];
        round++;
        if (cur.count > best) best = cur.count;
        draw();
      };
      box.querySelector('.pano-spin').addEventListener('click', spin);
      box.querySelector('.pano-spin10').addEventListener('click', () => {
        let n = 0;
        const id = setInterval(() => { spin(); if (++n >= 10) clearInterval(id); }, 130);
      });
      box.querySelector('.pano-reset').addEventListener('click', () => {
        round = 0; best = 0; cur = null; reshuffle(); draw();
      });
      draw();
    });
  }

  /* ---------------- Widget 5: before/after blend wipe ---------------- */
  function initBlend() {
    const box = $('w-blend');
    if (!box) return;
    const wrap = box.querySelector('.pano-ba');
    const topLayer = box.querySelector('.pano-ba-top');
    const topImg = topLayer.querySelector('img');
    const line = box.querySelector('.pano-ba-line');
    const slider = box.querySelector('input[type=range]');

    const sync = () => {
      topImg.style.width = wrap.clientWidth + 'px';
      topLayer.style.width = slider.value + '%';
      line.style.left = slider.value + '%';
    };
    slider.addEventListener('input', sync);
    window.addEventListener('resize', sync);
    wrap.querySelectorAll('img').forEach((im) => {
      if (!im.complete) im.addEventListener('load', sync);
    });
    sync();
  }

  initHarris();
  initAnms();
  initMatch();
  initRansac();
  initBlend();
}

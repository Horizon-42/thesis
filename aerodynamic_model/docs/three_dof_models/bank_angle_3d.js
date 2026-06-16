(function () {
  const TAU = Math.PI * 2;
  const DEG = Math.PI / 180;

  function vec(x, y, z) {
    return { x, y, z };
  }

  function add(a, b) {
    return vec(a.x + b.x, a.y + b.y, a.z + b.z);
  }

  function sub(a, b) {
    return vec(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  function scale(a, k) {
    return vec(a.x * k, a.y * k, a.z * k);
  }

  function dot(a, b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  function cross(a, b) {
    return vec(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
    );
  }

  function length(a) {
    return Math.hypot(a.x, a.y, a.z);
  }

  function normalize(a) {
    const len = length(a);
    return len > 1e-9 ? scale(a, 1 / len) : vec(0, 0, 0);
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function formatDeg(value) {
    return `${Math.round(value)} deg`;
  }

  function setupDemo(root) {
    const symbol = root.dataset.symbol || "mu";
    const forceLabel = root.dataset.force || "normal force";
    const initialBank = Number(root.dataset.bank || 35);

    root.innerHTML = `
      <div class="bank-demo__stage">
        <canvas aria-label="3D bank angle direction diagram"></canvas>
      </div>
      <div class="bank-demo__controls">
        <label>
          <span>Bank ${symbol}</span>
          <input data-bank-slider type="range" min="-60" max="60" step="1" value="${initialBank}">
          <output data-bank-output>${formatDeg(initialBank)}</output>
        </label>
        <label>
          <span>View pitch</span>
          <input data-pitch-slider type="range" min="-10" max="65" step="1" value="28">
          <output data-pitch-output>28 deg</output>
        </label>
        <button type="button" data-reset-view>Reset view</button>
      </div>
      <div class="bank-demo__legend" aria-hidden="true">
        <span><i class="bank-demo__swatch bank-demo__swatch--speed"></i>speed axis eV</span>
        <span><i class="bank-demo__swatch bank-demo__swatch--normal"></i>zero-bank normal e_gamma</span>
        <span><i class="bank-demo__swatch bank-demo__swatch--turn"></i>turn direction e_psi</span>
        <span><i class="bank-demo__swatch bank-demo__swatch--lift"></i>banked ${forceLabel}</span>
      </div>
    `;

    const canvas = root.querySelector("canvas");
    const bankSlider = root.querySelector("[data-bank-slider]");
    const bankOutput = root.querySelector("[data-bank-output]");
    const pitchSlider = root.querySelector("[data-pitch-slider]");
    const pitchOutput = root.querySelector("[data-pitch-output]");
    const resetButton = root.querySelector("[data-reset-view]");
    const ctx = canvas.getContext("2d");
    const state = {
      bank: initialBank,
      yaw: -38,
      pitch: 28,
      dragging: false,
      lastX: 0,
      lastY: 0,
    };

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function rotateForView(p) {
      const yaw = state.yaw * DEG;
      const pitch = state.pitch * DEG;
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const cp = Math.cos(pitch);
      const sp = Math.sin(pitch);

      const x1 = cy * p.x - sy * p.y;
      const y1 = sy * p.x + cy * p.y;
      const z1 = p.z;
      const y2 = cp * y1 - sp * z1;
      const z2 = sp * y1 + cp * z1;
      return { x: x1, y: y2, z: z2 };
    }

    function project(p) {
      const rect = canvas.getBoundingClientRect();
      const scalePx = Math.min(rect.width, rect.height) * 0.25;
      const rotated = rotateForView(p);
      return {
        x: rect.width * 0.5 + rotated.x * scalePx,
        y: rect.height * 0.57 - rotated.z * scalePx,
        depth: rotated.y,
      };
    }

    function drawLabel(text, p, color, dx = 8, dy = -8) {
      const q = project(p);
      ctx.save();
      ctx.font = "600 13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.textBaseline = "middle";
      const width = ctx.measureText(text).width + 12;
      const x = q.x + dx;
      const y = q.y + dy;
      ctx.fillStyle = "rgba(255,255,255,0.86)";
      ctx.strokeStyle = "rgba(203,215,226,0.9)";
      ctx.lineWidth = 1;
      roundRect(x - 6, y - 10, width, 20, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.fillText(text, x, y + 1);
      ctx.restore();
    }

    function roundRect(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r);
      ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }

    function drawLine(a, b, color, width = 2, dash = []) {
      const p = project(a);
      const q = project(b);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(q.x, q.y);
      ctx.stroke();
      ctx.restore();
    }

    function drawArrow(a, b, color, width = 2, dash = []) {
      drawLine(a, b, color, width, dash);
      const p = project(a);
      const q = project(b);
      const angle = Math.atan2(q.y - p.y, q.x - p.x);
      const size = 8 + width;
      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(q.x, q.y);
      ctx.lineTo(q.x - size * Math.cos(angle - Math.PI / 7), q.y - size * Math.sin(angle - Math.PI / 7));
      ctx.lineTo(q.x - size * Math.cos(angle + Math.PI / 7), q.y - size * Math.sin(angle + Math.PI / 7));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function drawPolyline(points, color, width = 2, dash = []) {
      if (points.length < 2) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      points.forEach((point, index) => {
        const p = project(point);
        if (index === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
      ctx.restore();
    }

    function drawGrid() {
      const color = "rgba(90,106,121,0.18)";
      for (let i = -2; i <= 2; i += 1) {
        drawLine(vec(-1.4, i * 0.7, 0), vec(1.4, i * 0.7, 0), color, 1);
        drawLine(vec(i * 0.7, -1.4, 0), vec(i * 0.7, 1.4, 0), color, 1);
      }
    }

    function drawAircraft(eV, lift) {
      const nose = scale(eV, 0.78);
      const tail = scale(eV, -0.36);
      const span = normalize(cross(lift, eV));
      const wingA = add(scale(eV, 0.04), scale(span, 0.58));
      const wingB = add(scale(eV, 0.04), scale(span, -0.58));
      const tailA = add(scale(eV, -0.26), scale(span, 0.26));
      const tailB = add(scale(eV, -0.26), scale(span, -0.26));

      drawLine(tail, nose, "#24344a", 5);
      drawArrow(scale(eV, 0.54), nose, "#24344a", 3);
      drawLine(wingA, wingB, "#24344a", 6);
      drawLine(tailA, tailB, "#596879", 4);
    }

    function draw() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);

      const psi = 34 * DEG;
      const gamma = 13 * DEG;
      const bank = state.bank * DEG;
      const up = vec(0, 0, 1);
      const eV = normalize(vec(Math.cos(gamma) * Math.cos(psi), Math.cos(gamma) * Math.sin(psi), Math.sin(gamma)));
      const eGamma = normalize(sub(up, scale(eV, dot(up, eV))));
      const ePsi = normalize(cross(eGamma, eV));
      const lift = normalize(add(scale(eGamma, Math.cos(bank)), scale(ePsi, Math.sin(bank))));
      const gammaComponent = scale(eGamma, Math.cos(bank));
      const psiComponent = scale(ePsi, Math.sin(bank));

      drawGrid();
      drawArrow(vec(0, 0, 0), vec(1.45, 0, 0), "#7d8996", 1.5);
      drawArrow(vec(0, 0, 0), vec(0, 1.45, 0), "#7d8996", 1.5);
      drawArrow(vec(0, 0, 0), vec(0, 0, 1.25), "#7d8996", 1.5);
      drawLabel("X", vec(1.5, 0, 0), "#5a6a79", 4, 0);
      drawLabel("Y", vec(0, 1.5, 0), "#5a6a79", 4, 0);
      drawLabel("h", vec(0, 0, 1.3), "#5a6a79", 4, 0);

      const arc = [];
      const steps = 42;
      for (let i = 0; i <= steps; i += 1) {
        const t = bank * (i / steps);
        arc.push(scale(add(scale(eGamma, Math.cos(t)), scale(ePsi, Math.sin(t))), 0.72));
      }

      const circle = [];
      for (let i = 0; i <= 96; i += 1) {
        const t = TAU * (i / 96);
        circle.push(scale(add(scale(eGamma, Math.cos(t)), scale(ePsi, Math.sin(t))), 0.75));
      }
      drawPolyline(circle, "rgba(111,72,168,0.24)", 1.5, [5, 6]);
      drawPolyline(arc, "#6f48a8", 3);

      drawArrow(vec(0, 0, 0), scale(eV, 1.35), "#24344a", 3);
      drawArrow(vec(0, 0, 0), scale(eGamma, 1.0), "#2f62b7", 2.5, [7, 5]);
      drawArrow(vec(0, 0, 0), scale(ePsi, 1.0), "#007c89", 2.5, [7, 5]);
      drawArrow(vec(0, 0, 0), scale(lift, 1.24), "#b7352d", 4);
      drawArrow(vec(0, 0, 0), gammaComponent, "#2f62b7", 2);
      drawArrow(gammaComponent, add(gammaComponent, psiComponent), "#007c89", 2);

      drawAircraft(eV, lift);

      drawLabel("eV", scale(eV, 1.38), "#24344a");
      drawLabel("zero bank: e_gamma", scale(eGamma, 1.02), "#2f62b7");
      drawLabel("+psi turn: e_psi", scale(ePsi, 1.03), "#007c89");
      drawLabel(`${forceLabel}`, scale(lift, 1.28), "#b7352d");
      drawLabel(`${symbol} = ${formatDeg(state.bank)}`, scale(arc[Math.floor(arc.length * 0.62)] || lift, 1.08), "#6f48a8", 6, -18);

      ctx.save();
      ctx.fillStyle = "#5a6a79";
      ctx.font = "500 12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText("drag the scene to rotate view", 16, rect.height - 18);
      ctx.restore();
    }

    function updateBank(value) {
      state.bank = Number(value);
      bankOutput.value = formatDeg(state.bank);
      draw();
    }

    function updatePitch(value) {
      state.pitch = Number(value);
      pitchOutput.value = formatDeg(state.pitch);
      draw();
    }

    bankSlider.addEventListener("input", (event) => updateBank(event.target.value));
    pitchSlider.addEventListener("input", (event) => updatePitch(event.target.value));
    resetButton.addEventListener("click", () => {
      state.yaw = -38;
      state.pitch = 28;
      pitchSlider.value = state.pitch;
      pitchOutput.value = formatDeg(state.pitch);
      draw();
    });

    canvas.addEventListener("pointerdown", (event) => {
      state.dragging = true;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!state.dragging) return;
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      state.yaw += dx * 0.42;
      state.pitch = clamp(state.pitch - dy * 0.28, -10, 65);
      pitchSlider.value = Math.round(state.pitch);
      pitchOutput.value = formatDeg(state.pitch);
      draw();
    });

    canvas.addEventListener("pointerup", (event) => {
      state.dragging = false;
      canvas.releasePointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointercancel", () => {
      state.dragging = false;
    });

    if ("ResizeObserver" in window) {
      new ResizeObserver(resizeCanvas).observe(canvas);
    } else {
      window.addEventListener("resize", resizeCanvas);
    }

    resizeCanvas();
  }

  function init() {
    document.querySelectorAll("[data-bank-demo]").forEach(setupDemo);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

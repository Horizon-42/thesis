(function () {
  const DEG = Math.PI / 180;
  const COLORS = {
    axis: "#24344a",
    muted: "#5a6a79",
    speed: "#24344a",
    thrust: "#a55212",
    drag: "#6f48a8",
    normal: "#b7352d",
    component: "#2f62b7",
    turn: "#007c89",
    weight: "#344254",
    grid: "rgba(90,106,121,0.18)",
  };

  const SCENES = {
    "alpha-overview": {
      title: "完整模型总览",
      caption: "只保留四个主力：T、D、N=L+Tsinα、mg。",
      legend: [
        ["speed", "speed axis eV"],
        ["thrust", "T near eV"],
        ["normal", "N banked normal"],
        ["weight", "mg"],
      ],
      draw: drawAlphaOverview,
    },
    "simplified-overview": {
      title: "简化模型总览",
      caption: "只保留 T、D、L=nmg、mg；分量放到后续推导段。",
      legend: [
        ["speed", "speed axis eV"],
        ["thrust", "T"],
        ["normal", "L = nmg"],
        ["weight", "mg"],
      ],
      draw: drawSimplifiedOverview,
    },
    "alpha-vdot": {
      title: "Vdot 切向力",
      caption: "只画沿速度方向的贡献：Tcosα、D、mg sinγ。",
      legend: [
        ["speed", "speed axis eV"],
        ["thrust", "T and Tcosα"],
        ["drag", "D"],
        ["weight", "mg sinγ"],
      ],
      draw: drawAlphaVdot,
    },
    "simplified-vdot": {
      title: "Vdot 切向力",
      caption: "α≈0 时推力直接沿 eV，切向只剩 T、D、mg sinγ。",
      legend: [
        ["speed", "speed axis eV"],
        ["thrust", "T"],
        ["drag", "D"],
        ["weight", "mg sinγ"],
      ],
      draw: drawSimplifiedVdot,
    },
    "alpha-gammadot": {
      title: "gammadot 竖直法向力",
      caption: "只画 eγ 方向上的 Ncosφ 与反向 mg cosγ。",
      legend: [
        ["normal", "N"],
        ["component", "N cosφ"],
        ["weight", "mg cosγ"],
      ],
      draw: drawAlphaGammadot,
    },
    "simplified-gammadot": {
      title: "gammadot 竖直法向力",
      caption: "只画 eγ 方向上的 nmg cosμ 与反向 mg cosγ。",
      legend: [
        ["normal", "L = nmg"],
        ["component", "nmg cosμ"],
        ["weight", "mg cosγ"],
      ],
      draw: drawSimplifiedGammadot,
    },
    "alpha-mdot": {
      title: "mdot 推力幅值",
      caption: "质量方程只需要推力大小 T，不需要再做空间分量分解。",
      legend: [
        ["speed", "speed axis eV"],
        ["thrust", "T magnitude"],
      ],
      draw: drawAlphaMdot,
    },
  };

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

  function setupForceDemo(root) {
    const scene = SCENES[root.dataset.scene];
    if (!scene) return;

    root.innerHTML = `
      <div class="force-demo__stage">
        <canvas aria-label="${scene.title} 3D force diagram"></canvas>
      </div>
      <div class="force-demo__controls">
        <span>${scene.caption}</span>
        <button type="button" data-reset-view>Reset view</button>
      </div>
      <div class="force-demo__legend" aria-hidden="true">
        ${scene.legend.map(([kind, label]) => `<span><i class="force-demo__swatch force-demo__swatch--${kind}"></i>${label}</span>`).join("")}
      </div>
    `;

    const canvas = root.querySelector("canvas");
    const resetButton = root.querySelector("[data-reset-view]");
    const ctx = canvas.getContext("2d");
    const state = {
      yaw: -38,
      pitch: 28,
      dragging: false,
      lastX: 0,
      lastY: 0,
    };

    const api = {
      ctx,
      canvas,
      state,
      project,
      drawArrow,
      drawLine,
      drawLabel,
      drawGrid,
      drawAircraft,
      drawPanelText,
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

    function draw() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      scene.draw(api);
      ctx.save();
      ctx.fillStyle = COLORS.muted;
      ctx.font = "500 12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText("drag the scene to rotate view", 16, rect.height - 18);
      ctx.restore();
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
      roundRect(ctx, x - 6, y - 10, width, 20, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.fillText(text, x, y + 1);
      ctx.restore();
    }

    function drawPanelText(lines) {
      const rect = canvas.getBoundingClientRect();
      const x = rect.width - 246;
      const y = 22;
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,0.88)";
      ctx.strokeStyle = "rgba(203,215,226,0.95)";
      ctx.lineWidth = 1;
      roundRect(ctx, x, y, 224, 30 + lines.length * 20, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = COLORS.axis;
      ctx.font = "600 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      lines.forEach((line, index) => ctx.fillText(line, x + 12, y + 24 + index * 20));
      ctx.restore();
    }

    function drawGrid() {
      for (let i = -2; i <= 2; i += 1) {
        drawLine(vec(-1.4, i * 0.7, 0), vec(1.4, i * 0.7, 0), COLORS.grid, 1);
        drawLine(vec(i * 0.7, -1.4, 0), vec(i * 0.7, 1.4, 0), COLORS.grid, 1);
      }
      drawArrow(vec(0, 0, 0), vec(1.45, 0, 0), "#7d8996", 1.5);
      drawArrow(vec(0, 0, 0), vec(0, 1.45, 0), "#7d8996", 1.5);
      drawArrow(vec(0, 0, 0), vec(0, 0, 1.25), "#7d8996", 1.5);
      drawLabel("X", vec(1.5, 0, 0), COLORS.muted, 4, 0);
      drawLabel("Y", vec(0, 1.5, 0), COLORS.muted, 4, 0);
      drawLabel("h", vec(0, 0, 1.3), COLORS.muted, 4, 0);
    }

    function drawAircraft(eV, normal) {
      const nose = scale(eV, 0.78);
      const tail = scale(eV, -0.36);
      const span = normalize(cross(normal, eV));
      const wingA = add(scale(eV, 0.04), scale(span, 0.58));
      const wingB = add(scale(eV, 0.04), scale(span, -0.58));
      const tailA = add(scale(eV, -0.26), scale(span, 0.26));
      const tailB = add(scale(eV, -0.26), scale(span, -0.26));
      drawLine(tail, nose, COLORS.axis, 5);
      drawArrow(scale(eV, 0.54), nose, COLORS.axis, 3);
      drawLine(wingA, wingB, COLORS.axis, 6);
      drawLine(tailA, tailB, "#596879", 4);
    }

    resetButton.addEventListener("click", () => {
      state.yaw = -38;
      state.pitch = 28;
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

  function roundRect(ctx, x, y, w, h, r) {
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

  function modelBasis(bankDeg = 35) {
    const psi = 34 * DEG;
    const gamma = 13 * DEG;
    const bank = bankDeg * DEG;
    const up = vec(0, 0, 1);
    const eV = normalize(vec(Math.cos(gamma) * Math.cos(psi), Math.cos(gamma) * Math.sin(psi), Math.sin(gamma)));
    const eGamma = normalize(sub(up, scale(eV, dot(up, eV))));
    const ePsi = normalize(cross(eGamma, eV));
    const normal = normalize(add(scale(eGamma, Math.cos(bank)), scale(ePsi, Math.sin(bank))));
    return { eV, eGamma, ePsi, normal, bank };
  }

  function drawBase(api, bankDeg = 35) {
    const b = modelBasis(bankDeg);
    api.drawGrid();
    api.drawAircraft(b.eV, b.normal);
    api.drawArrow(vec(0, 0, 0), scale(b.eV, 1.35), COLORS.speed, 3);
    api.drawLabel("eV", scale(b.eV, 1.38), COLORS.speed);
    return b;
  }

  function drawAlphaOverview(api) {
    const b = drawBase(api, 35);
    const alpha = 16 * DEG;
    const thrustDir = normalize(add(scale(b.eV, Math.cos(alpha)), scale(b.eGamma, Math.sin(alpha))));
    api.drawArrow(vec(0, 0, 0), scale(thrustDir, 1.02), COLORS.thrust, 3);
    api.drawArrow(vec(0, 0, 0), scale(b.eV, -0.72), COLORS.drag, 3);
    api.drawArrow(vec(0, 0, 0), scale(b.normal, 1.08), COLORS.normal, 4);
    api.drawArrow(vec(0, 0, 0), vec(0, 0, -0.92), COLORS.weight, 3);
    api.drawLabel("T", scale(thrustDir, 1.05), COLORS.thrust);
    api.drawLabel("D", scale(b.eV, -0.75), COLORS.drag, -42, 4);
    api.drawLabel("N = L + T sin alpha", scale(b.normal, 1.12), COLORS.normal);
    api.drawLabel("mg", vec(0, 0, -0.96), COLORS.weight);
    api.drawPanelText(["normal plane rotates", "around eV by phi"]);
  }

  function drawSimplifiedOverview(api) {
    const b = drawBase(api, 35);
    api.drawArrow(vec(0, 0, 0), scale(b.eV, 1.0), COLORS.thrust, 3);
    api.drawArrow(vec(0, 0, 0), scale(b.eV, -0.72), COLORS.drag, 3);
    api.drawArrow(vec(0, 0, 0), scale(b.normal, 1.08), COLORS.normal, 4);
    api.drawArrow(vec(0, 0, 0), vec(0, 0, -0.92), COLORS.weight, 3);
    api.drawLabel("T", scale(b.eV, 1.04), COLORS.thrust);
    api.drawLabel("D", scale(b.eV, -0.75), COLORS.drag, -42, 4);
    api.drawLabel("L = nmg", scale(b.normal, 1.12), COLORS.normal);
    api.drawLabel("mg", vec(0, 0, -0.96), COLORS.weight);
    api.drawPanelText(["alpha approx 0", "L is commanded by n"]);
  }

  function drawAlphaVdot(api) {
    const b = drawBase(api, 0);
    const alpha = 18 * DEG;
    const thrustDir = normalize(add(scale(b.eV, Math.cos(alpha)), scale(b.eGamma, Math.sin(alpha))));
    const offset = scale(b.ePsi, -0.16);
    api.drawArrow(vec(0, 0, 0), scale(thrustDir, 0.92), COLORS.thrust, 3);
    api.drawArrow(add(offset, vec(0, 0, 0)), add(offset, scale(b.eV, 0.72)), COLORS.component, 3);
    api.drawArrow(scale(b.ePsi, 0.12), add(scale(b.ePsi, 0.12), scale(b.eV, -0.62)), COLORS.drag, 3);
    api.drawArrow(scale(b.ePsi, 0.3), add(scale(b.ePsi, 0.3), scale(b.eV, -0.42)), COLORS.weight, 2.5);
    api.drawLabel("T", scale(thrustDir, 0.96), COLORS.thrust);
    api.drawLabel("T cos alpha", add(offset, scale(b.eV, 0.74)), COLORS.component);
    api.drawLabel("-D", add(scale(b.ePsi, 0.12), scale(b.eV, -0.66)), COLORS.drag, -38, 3);
    api.drawLabel("-mg sin gamma", add(scale(b.ePsi, 0.3), scale(b.eV, -0.45)), COLORS.weight, -82, 18);
    api.drawPanelText(["sum F_eV", "= T cos alpha - D", "  - mg sin gamma"]);
  }

  function drawSimplifiedVdot(api) {
    const b = drawBase(api, 0);
    api.drawArrow(vec(0, 0, 0), scale(b.eV, 0.9), COLORS.thrust, 3);
    api.drawArrow(scale(b.ePsi, 0.12), add(scale(b.ePsi, 0.12), scale(b.eV, -0.62)), COLORS.drag, 3);
    api.drawArrow(scale(b.ePsi, 0.3), add(scale(b.ePsi, 0.3), scale(b.eV, -0.42)), COLORS.weight, 2.5);
    api.drawLabel("T", scale(b.eV, 0.94), COLORS.thrust);
    api.drawLabel("-D", add(scale(b.ePsi, 0.12), scale(b.eV, -0.66)), COLORS.drag, -38, 3);
    api.drawLabel("-mg sin gamma", add(scale(b.ePsi, 0.3), scale(b.eV, -0.45)), COLORS.weight, -82, 18);
    api.drawPanelText(["sum F_eV", "= T - D", "  - mg sin gamma"]);
  }

  function drawAlphaGammadot(api) {
    drawGammadot(api, "N", "N cos phi", "phi");
  }

  function drawSimplifiedGammadot(api) {
    drawGammadot(api, "L = nmg", "nmg cos mu", "mu");
  }

  function drawGammadot(api, normalLabel, componentLabel, symbol) {
    const b = drawBase(api, 35);
    const weightOffset = scale(b.ePsi, 0.18);
    api.drawArrow(vec(0, 0, 0), scale(b.eGamma, 1.0), COLORS.component, 2.5, [7, 5]);
    api.drawArrow(vec(0, 0, 0), scale(b.normal, 1.05), COLORS.normal, 4);
    api.drawArrow(vec(0, 0, 0), scale(b.eGamma, 0.72), COLORS.component, 3);
    api.drawArrow(weightOffset, add(weightOffset, scale(b.eGamma, -0.52)), COLORS.weight, 3);
    api.drawLabel("e_gamma", scale(b.eGamma, 1.04), COLORS.component);
    api.drawLabel(normalLabel, scale(b.normal, 1.09), COLORS.normal);
    api.drawLabel(componentLabel, scale(b.eGamma, 0.75), COLORS.component);
    api.drawLabel("-mg cos gamma", add(weightOffset, scale(b.eGamma, -0.55)), COLORS.weight, -96, 18);
    api.drawPanelText([`${componentLabel}`, "- mg cos gamma", `bank ${symbol} only scales`, "the e_gamma part"]);
  }

  function drawAlphaMdot(api) {
    const b = drawBase(api, 0);
    api.drawArrow(vec(0, 0, 0), scale(b.eV, 1.02), COLORS.thrust, 4);
    api.drawLabel("T", scale(b.eV, 1.06), COLORS.thrust);
    api.drawPanelText(["mdot = -c_T T / g", "T is used as magnitude", "not a spatial projection"]);
  }

  function init() {
    document.querySelectorAll("[data-force-demo]").forEach(setupForceDemo);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

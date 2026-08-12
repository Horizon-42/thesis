/**
 * Tiny purpose-built WebGL renderer for the evaluation report's 3D deviation
 * scatter. It keeps static geometry in two GPU buffers and redraws only when
 * the camera or canvas size changes.
 */

import type { EvaluationVerdict } from "../data/evaluationReport";

export interface DeviationScatterDatum {
  label: string;
  lateral: number;
  vertical: number;
  verdict: EvaluationVerdict;
}

export interface DeviationOrbitView {
  yaw: number;
  pitch: number;
  distance: number;
}

export interface DeviationScatterHit extends DeviationScatterDatum {
  index: number;
  screenX: number;
  screenY: number;
}

export interface DeviationScatterRenderer {
  resize(width: number, height: number): void;
  draw(view: DeviationOrbitView): void;
  hitTest(x: number, y: number, view: DeviationOrbitView): DeviationScatterHit | null;
  dispose(): void;
}

type Vec3 = [number, number, number];
type Color = [number, number, number, number];

interface PreparedPoint extends DeviationScatterDatum {
  index: number;
  world: Vec3;
}

const HALF_EXTENT = 0.82;
const FIELD_OF_VIEW_RAD = (52 * Math.PI) / 180;
const VERTEX_STRIDE_FLOATS = 7;

const VERTEX_SHADER = `
  attribute vec3 a_position;
  attribute vec4 a_color;
  uniform float u_yaw;
  uniform float u_pitch;
  uniform float u_distance;
  uniform float u_aspect;
  uniform float u_point_size;
  varying vec4 v_color;

  void main() {
    float cy = cos(u_yaw);
    float sy = sin(u_yaw);
    vec3 yawed = vec3(
      cy * a_position.x + sy * a_position.z,
      a_position.y,
      -sy * a_position.x + cy * a_position.z
    );

    float cp = cos(u_pitch);
    float sp = sin(u_pitch);
    vec3 camera = vec3(
      yawed.x,
      cp * yawed.y - sp * yawed.z,
      sp * yawed.y + cp * yawed.z - u_distance
    );

    float near_plane = 0.1;
    float far_plane = 20.0;
    float focal = 1.0 / tan(0.453785606);
    gl_Position = vec4(
      camera.x * focal / u_aspect,
      camera.y * focal,
      ((far_plane + near_plane) / (near_plane - far_plane)) * camera.z
        + ((2.0 * far_plane * near_plane) / (near_plane - far_plane)),
      -camera.z
    );
    gl_PointSize = u_point_size;
    v_color = a_color;
  }
`;

const FRAGMENT_SHADER = `
  precision mediump float;
  uniform float u_round_points;
  varying vec4 v_color;

  void main() {
    if (u_round_points > 0.5) {
      float radius = length(gl_PointCoord - vec2(0.5));
      if (radius > 0.5) discard;
    }
    gl_FragColor = v_color;
  }
`;

const FRAME_COLOR: Color = [0.42, 0.5, 0.62, 0.26];
const LATERAL_AXIS_COLOR: Color = [0.38, 0.68, 1, 0.95];
const VERTICAL_AXIS_COLOR: Color = [1, 0.72, 0.3, 0.95];
const FLIGHT_AXIS_COLOR: Color = [0.72, 0.56, 1, 0.95];
const GATE_COLOR: Color = [0.25, 0.75, 0.45, 0.72];
const REFERENCE_COLOR: Color = [0.88, 0.92, 0.98, 0.48];
const STEM_COLOR: Color = [0.58, 0.66, 0.78, 0.56];
const PASS_COLOR: Color = [0.25, 0.75, 0.45, 0.96];
const FAIL_COLOR: Color = [0.88, 0.36, 0.36, 0.96];
const INDETERMINATE_COLOR: Color = [0.6, 0.62, 0.65, 0.96];

export function deviationScatterColor(verdict: EvaluationVerdict): Color {
  return verdict === "pass"
    ? PASS_COLOR
    : verdict === "fail"
      ? FAIL_COLOR
      : INDETERMINATE_COLOR;
}

function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string,
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl: WebGLRenderingContext): WebGLProgram | null {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  if (!vertex || !fragment) {
    if (vertex) gl.deleteShader(vertex);
    if (fragment) gl.deleteShader(fragment);
    return null;
  }

  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    return null;
  }
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

function pushVertex(vertices: number[], point: Vec3, color: Color) {
  vertices.push(point[0], point[1], point[2], color[0], color[1], color[2], color[3]);
}

function pushLine(vertices: number[], start: Vec3, end: Vec3, color: Color) {
  pushVertex(vertices, start, color);
  pushVertex(vertices, end, color);
}

function pushBoxEdges(vertices: number[], low: Vec3, high: Vec3, color: Color) {
  const [x0, y0, z0] = low;
  const [x1, y1, z1] = high;
  const corners: Vec3[] = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  for (const [start, end] of edges) {
    pushLine(vertices, corners[start], corners[end], color);
  }
}

function rotateToCamera(point: Vec3, view: DeviationOrbitView): Vec3 {
  const cy = Math.cos(view.yaw);
  const sy = Math.sin(view.yaw);
  const yawedX = cy * point[0] + sy * point[2];
  const yawedZ = -sy * point[0] + cy * point[2];
  const cp = Math.cos(view.pitch);
  const sp = Math.sin(view.pitch);
  return [
    yawedX,
    cp * point[1] - sp * yawedZ,
    sp * point[1] + cp * yawedZ - view.distance,
  ];
}

function projectToScreen(
  point: Vec3,
  view: DeviationOrbitView,
  width: number,
  height: number,
): { x: number; y: number } | null {
  const camera = rotateToCamera(point, view);
  if (camera[2] >= -0.1) return null;
  const focal = 1 / Math.tan(FIELD_OF_VIEW_RAD / 2);
  const aspect = width / Math.max(1, height);
  const ndcX = (camera[0] * focal / aspect) / -camera[2];
  const ndcY = (camera[1] * focal) / -camera[2];
  return {
    x: ((ndcX + 1) / 2) * width,
    y: ((1 - ndcY) / 2) * height,
  };
}

export function createDeviationScatterRenderer(
  canvas: HTMLCanvasElement,
  points: DeviationScatterDatum[],
  lateralGate: number,
  verticalBand: [number, number],
): DeviationScatterRenderer | null {
  const gl = canvas.getContext("webgl", {
    alpha: false,
    antialias: true,
    depth: true,
    powerPreference: "low-power",
    preserveDrawingBuffer: false,
  });
  if (!gl) return null;

  const program = createProgram(gl);
  if (!program) return null;
  const positionLocation = gl.getAttribLocation(program, "a_position");
  const colorLocation = gl.getAttribLocation(program, "a_color");
  const yawLocation = gl.getUniformLocation(program, "u_yaw");
  const pitchLocation = gl.getUniformLocation(program, "u_pitch");
  const distanceLocation = gl.getUniformLocation(program, "u_distance");
  const aspectLocation = gl.getUniformLocation(program, "u_aspect");
  const pointSizeLocation = gl.getUniformLocation(program, "u_point_size");
  const roundPointsLocation = gl.getUniformLocation(program, "u_round_points");
  if (
    positionLocation < 0 ||
    colorLocation < 0 ||
    !yawLocation ||
    !pitchLocation ||
    !distanceLocation ||
    !aspectLocation ||
    !pointSizeLocation ||
    !roundPointsLocation
  ) {
    gl.deleteProgram(program);
    return null;
  }

  const lateralMax =
    Math.max(1, lateralGate, ...points.map((point) => point.lateral)) * 1.1;
  const rawVerticalLow = Math.min(
    0,
    verticalBand[0],
    ...points.map((point) => point.vertical),
  );
  const rawVerticalHigh = Math.max(
    0,
    verticalBand[1],
    ...points.map((point) => point.vertical),
  );
  const rawVerticalSpan = Math.max(1, rawVerticalHigh - rawVerticalLow);
  const verticalLow = rawVerticalLow - rawVerticalSpan * 0.1;
  const verticalHigh = rawVerticalHigh + rawVerticalSpan * 0.1;
  const verticalSpan = verticalHigh - verticalLow;
  const worldX = (lateral: number) =>
    -HALF_EXTENT + (lateral / lateralMax) * HALF_EXTENT * 2;
  const worldY = (vertical: number) =>
    -HALF_EXTENT + ((vertical - verticalLow) / verticalSpan) * HALF_EXTENT * 2;
  const worldZ = (index: number) =>
    points.length > 1
      ? -HALF_EXTENT + (index / (points.length - 1)) * HALF_EXTENT * 2
      : 0;

  const preparedPoints: PreparedPoint[] = points.map((point, index) => ({
    ...point,
    index,
    world: [worldX(point.lateral), worldY(point.vertical), worldZ(index)],
  }));

  const lineVertices: number[] = [];
  pushBoxEdges(
    lineVertices,
    [-HALF_EXTENT, -HALF_EXTENT, -HALF_EXTENT],
    [HALF_EXTENT, HALF_EXTENT, HALF_EXTENT],
    FRAME_COLOR,
  );
  const axisOrigin: Vec3 = [-HALF_EXTENT, -HALF_EXTENT, -HALF_EXTENT];
  pushLine(
    lineVertices,
    axisOrigin,
    [HALF_EXTENT, -HALF_EXTENT, -HALF_EXTENT],
    LATERAL_AXIS_COLOR,
  );
  pushLine(
    lineVertices,
    axisOrigin,
    [-HALF_EXTENT, HALF_EXTENT, -HALF_EXTENT],
    VERTICAL_AXIS_COLOR,
  );
  pushLine(
    lineVertices,
    axisOrigin,
    [-HALF_EXTENT, -HALF_EXTENT, HALF_EXTENT],
    FLIGHT_AXIS_COLOR,
  );
  pushBoxEdges(
    lineVertices,
    [worldX(0), worldY(verticalBand[0]), -HALF_EXTENT],
    [worldX(lateralGate), worldY(verticalBand[1]), HALF_EXTENT],
    GATE_COLOR,
  );
  pushLine(
    lineVertices,
    [worldX(0), worldY(0), -HALF_EXTENT],
    [worldX(0), worldY(0), HALF_EXTENT],
    REFERENCE_COLOR,
  );
  for (const point of preparedPoints) {
    pushLine(
      lineVertices,
      [worldX(0), worldY(0), point.world[2]],
      point.world,
      STEM_COLOR,
    );
  }

  const pointVertices: number[] = [];
  for (const point of preparedPoints) {
    pushVertex(pointVertices, point.world, deviationScatterColor(point.verdict));
  }

  const createBuffer = (data: number[]) => {
    const buffer = gl.createBuffer();
    if (!buffer) return null;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
    return buffer;
  };
  const lineBuffer = createBuffer(lineVertices);
  const pointBuffer = createBuffer(pointVertices);
  if (!lineBuffer || !pointBuffer) {
    if (lineBuffer) gl.deleteBuffer(lineBuffer);
    if (pointBuffer) gl.deleteBuffer(pointBuffer);
    gl.deleteProgram(program);
    return null;
  }

  const bindBuffer = (buffer: WebGLBuffer) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    const stride = VERTEX_STRIDE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, stride, 0);
    gl.vertexAttribPointer(
      colorLocation,
      4,
      gl.FLOAT,
      false,
      stride,
      3 * Float32Array.BYTES_PER_ELEMENT,
    );
    gl.enableVertexAttribArray(positionLocation);
    gl.enableVertexAttribArray(colorLocation);
  };

  gl.useProgram(program);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0.035, 0.06, 0.11, 1);

  return {
    resize(width, height) {
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      gl.viewport(0, 0, canvas.width, canvas.height);
    },

    draw(view) {
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(program);
      gl.uniform1f(yawLocation, view.yaw);
      gl.uniform1f(pitchLocation, view.pitch);
      gl.uniform1f(distanceLocation, view.distance);
      gl.uniform1f(aspectLocation, canvas.width / Math.max(1, canvas.height));

      bindBuffer(lineBuffer);
      gl.uniform1f(pointSizeLocation, 1);
      gl.uniform1f(roundPointsLocation, 0);
      gl.drawArrays(gl.LINES, 0, lineVertices.length / VERTEX_STRIDE_FLOATS);

      bindBuffer(pointBuffer);
      const pixelRatio = canvas.width / Math.max(1, canvas.clientWidth);
      gl.uniform1f(pointSizeLocation, Math.min(14, 7 * pixelRatio));
      gl.uniform1f(roundPointsLocation, 1);
      gl.drawArrays(gl.POINTS, 0, preparedPoints.length);
    },

    hitTest(x, y, view) {
      const width = canvas.clientWidth || canvas.width;
      const height = canvas.clientHeight || canvas.height;
      let closest: DeviationScatterHit | null = null;
      let closestDistance = 10;
      for (const point of preparedPoints) {
        const screen = projectToScreen(point.world, view, width, height);
        if (!screen) continue;
        const distance = Math.hypot(screen.x - x, screen.y - y);
        if (distance > closestDistance) continue;
        closestDistance = distance;
        closest = {
          label: point.label,
          lateral: point.lateral,
          vertical: point.vertical,
          verdict: point.verdict,
          index: point.index,
          screenX: screen.x,
          screenY: screen.y,
        };
      }
      return closest;
    },

    dispose() {
      gl.deleteBuffer(lineBuffer);
      gl.deleteBuffer(pointBuffer);
      gl.deleteProgram(program);
    },
  };
}

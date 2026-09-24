/**
 * The Training flight in the 3D scene, on real Cesium entities (no WebGL canvas): the envelopes
 * the switches ask for are built once; the SELECTED word — one column's, chosen in the sentence
 * bar — lights up its own envelope and nothing of another column's, and is restored without
 * rebuilding anything; a new flight is framed once; leaving Training removes the layer.
 */
import { useLayoutEffect } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import * as Cesium from "cesium";
import { AppProvider, useApp } from "../../context/AppContext";
import { parseTrainingSample } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";
import useTrainingTrackLayer, { TRAINING_ENTITY } from "../../hooks/useTrainingTrackLayer";
import {
  TRAINING_CORRIDOR_COLOR,
  TRAINING_FUNNEL_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_TURN_COLOR,
  TRAINING_WORD_COLOR,
} from "../../utils/trainingWordColors";
import TrainingSentenceBar from "../TrainingSentenceBar";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({
    defaultAirport: "KXXX",
    airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }],
  }),
}));

/** The regions and the funnels are off by default; most tests look at every envelope. */
async function setup(position = 0, edit: (raw: any) => void = () => undefined, everyEnvelope = true) {
  const raw = mockSample();
  edit(raw);
  const parsed = parseTrainingSample(raw);
  if (!parsed.ok) throw new Error(parsed.problem);
  const selection = {
    vocabulary: parsed.value.vocabulary, candidates: parsed.value.candidates, flight: parsed.value.flights[position],
  };
  const entities = new Cesium.EntityCollection();
  const requestRender = vi.fn();
  const camera = { heading: 0, flyToBoundingSphere: vi.fn() };
  const viewer = { entities, scene: { requestRender }, camera, isDestroyed: () => false } as unknown as Cesium.Viewer;
  let app: ReturnType<typeof useApp>;
  function Scene() {
    app = useApp();
    useTrainingTrackLayer();
    useLayoutEffect(() => {
      app.setViewer(viewer);
      app.setMode("training");
      if (everyEnvelope) {
        app.setTrainingLayer("turnRegions", true);
        app.setTrainingLayer("holdFunnels", true);
      }
      app.setTrainingSelection(selection);
    }, []);
    return <TrainingSentenceBar />;
  }
  const view = render(<AppProvider><Scene /></AppProvider>);
  await waitFor(() => expect(app!.airports).toHaveLength(1));
  const colourOf = (id: string) => {
    const entity = entities.getById(id)!;
    const graphics = entity.polygon ?? entity.wall ?? entity.polyline;
    return graphics!.material!.getValue(Cesium.JulianDate.now()).color as Cesium.Color;
  };
  const css = (value: string, alpha = 1) => Cesium.Color.fromCssColorString(value).withAlpha(alpha);
  return { ...view, entities, selection, requestRender, camera, colourOf, css, app: () => app! };
}

const band = (name: RegExp) => screen.getByLabelText(name);
const HEADING_180 = /^heading 180° — a turn/;
const DESCEND_TO_LAND = /^altitude descend to land — a new target/;

describe("Training envelopes in the 3D scene", () => {
  it("builds the track, the lateral envelopes, the tubes, every candidate and the issue points", async () => {
    const scene = await setup();
    for (const id of [
      TRAINING_ENTITY.track, TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis, TRAINING_ENTITY.captureTurn,
      TRAINING_ENTITY.turn(1), TRAINING_ENTITY.turnEnd(1), TRAINING_ENTITY.funnel(0), TRAINING_ENTITY.funnel(1), TRAINING_ENTITY.tube(0), TRAINING_ENTITY.tube(1),
      TRAINING_ENTITY.centreline("09"), TRAINING_ENTITY.runway("27"), TRAINING_ENTITY.issue(0), TRAINING_ENTITY.issue(1),
      TRAINING_ENTITY.clearance, TRAINING_ENTITY.capture, TRAINING_ENTITY.end, TRAINING_ENTITY.groundTrace,
      TRAINING_ENTITY.edge(TRAINING_ENTITY.funnel(1)), TRAINING_ENTITY.edge(TRAINING_ENTITY.corridor),
      TRAINING_ENTITY.edge(TRAINING_ENTITY.tube(1), "upper"), TRAINING_ENTITY.edge(TRAINING_ENTITY.tube(1), "lower"),
      TRAINING_ENTITY.path(TRAINING_ENTITY.turn(1), "fast"), TRAINING_ENTITY.path(TRAINING_ENTITY.turn(1), "slow"),
      TRAINING_ENTITY.path(TRAINING_ENTITY.captureTurn, "fast"), TRAINING_ENTITY.path(TRAINING_ENTITY.captureTurn, "slow"),
    ]) {
      expect(scene.entities.getById(id), id).toBeDefined();
    }
    // a word the flight was already holding at entry has no turn region
    expect(scene.entities.getById(TRAINING_ENTITY.turn(0))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.turnEnd(0))).toBeUndefined();
    // the lateral envelopes are draped on the ground: they give no height; their edges are draped too
    expect(scene.entities.getById(TRAINING_ENTITY.funnel(1))!.polygon!.height).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.edge(TRAINING_ENTITY.funnel(1)))!.polyline!.clampToGround!
      .getValue(Cesium.JulianDate.now())).toBe(true);
  });

  it("carries the labeller's verdict on the edges, as the plan view does", async () => {
    const scene = await setup();
    const { heading } = scene.selection.flight.envelopes;
    const failed = heading.findIndex((item) => item.holdCheck !== null && item.holdCheck.inside < item.holdCheck.rows);
    const passed = heading.findIndex((item) => item.holdCheck !== null && item.holdCheck.inside === item.holdCheck.rows);
    // (this failed hold is also a weak one: dotted, at the dash lines' opacity)
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.funnel(failed)))).toEqual(scene.css(TRAINING_OUTSIDE_COLOR, 0.85));
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.funnel(passed)))).toEqual(scene.css(TRAINING_FUNNEL_COLOR));
    // a tube with rows outside is edged in the verdict colour, not its own
    const tubes = scene.selection.flight.envelopes.altitude;
    const outside = tubes.findIndex((tube) => !tube.check.contained);
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.tube(outside), "upper")))
      .toEqual(scene.css(TRAINING_OUTSIDE_COLOR, 0.9));
  });

  it("walls each altitude tube over the aircraft's own ground track, between its exported edges", async () => {
    const scene = await setup();
    const wall = scene.entities.getById(TRAINING_ENTITY.tube(1))!.wall!;
    const tube = scene.selection.flight.envelopes.altitude[1];
    expect(wall.minimumHeights!.getValue(Cesium.JulianDate.now())).toEqual(tube.lowerHaeM);
    expect(wall.maximumHeights!.getValue(Cesium.JulianDate.now())).toEqual(tube.upperHaeM);
    const first = Cesium.Cartographic.fromCartesian(wall.positions!.getValue(Cesium.JulianDate.now())[0]);
    expect(Cesium.Math.toDegrees(first.longitude)).toBeCloseTo(scene.selection.flight.signals.lon[tube.row], 9);
  });

  it("highlights nothing until a word is selected", async () => {
    const scene = await setup();
    expect(scene.entities.getById(TRAINING_ENTITY.focusStretch)).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.focusIssue)).toBeUndefined();
    expect(scene.colourOf(TRAINING_ENTITY.funnel(0))).toEqual(scene.css(TRAINING_FUNNEL_COLOR, 0.18));
  });

  it("lights up ONLY the selected column's word — its own hue deepened, a yellow edge, its rows", async () => {
    const scene = await setup();
    const original = [...scene.entities.values];
    fireEvent.click(band(HEADING_180));
    expect(scene.colourOf(TRAINING_ENTITY.turn(1))).toEqual(scene.css(TRAINING_TURN_COLOR, 0.45));
    expect(scene.colourOf(TRAINING_ENTITY.funnel(1))).toEqual(scene.css(TRAINING_FUNNEL_COLOR, 0.45));
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.funnel(1)))).toEqual(scene.css(TRAINING_WORD_COLOR));
    // the weak hold's edge keeps its dots when it is lit
    const lit = scene.entities.getById(TRAINING_ENTITY.edge(TRAINING_ENTITY.funnel(1)))!.polyline!.material as Cesium.PolylineDashMaterialProperty;
    expect(lit.dashPattern!.getValue(Cesium.JulianDate.now())).toBe(0x3333);
    expect(scene.colourOf(TRAINING_ENTITY.path(TRAINING_ENTITY.turn(1), "slow"))).toEqual(scene.css(TRAINING_WORD_COLOR));
    // the altitude word in force at the same step is another column's: never lit, and it recedes
    // with every other word's envelope, keeping its hue
    expect(scene.colourOf(TRAINING_ENTITY.tube(0))).toEqual(scene.css(TRAINING_TUBE_COLOR, 0.28 * 0.3));
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.tube(0), "upper"))).toEqual(scene.css(TRAINING_TUBE_COLOR, 0.9 * 0.3));
    expect(scene.colourOf(TRAINING_ENTITY.corridor)).toEqual(scene.css(TRAINING_CORRIDOR_COLOR, 0.3 * 0.3));
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.corridor))).toEqual(scene.css(TRAINING_CORRIDOR_COLOR, 0.3));
    // its two turn paths named where they end, and its turn as flown marked on the track
    const text = (id: string) => scene.entities.getById(id)!.label!.text!.getValue(Cesium.JulianDate.now());
    expect(text(TRAINING_ENTITY.focusPathLabel("fast"))).toBe("fastest: ≤ 4.7°/s, ≤ 32° bank");
    expect(text(TRAINING_ENTITY.focusPathLabel("slow"))).toBe("slowest: 0.5°/s, begun 10.5 s late");
    expect(text(TRAINING_ENTITY.focusFlown("starts"))).toBe("turn flown starts · step 10");
    expect(text(TRAINING_ENTITY.focusFlown("ends"))).toBe("turn flown ends · step 16");
    const label = scene.entities.getById(TRAINING_ENTITY.focusIssue)!.label!.text!.getValue(Cesium.JulianDate.now());
    expect(label).toBe("heading 180° · step 10");
    expect(scene.entities.getById(TRAINING_ENTITY.focusStretch)).toBeDefined();

    fireEvent.click(band(DESCEND_TO_LAND));
    expect(scene.colourOf(TRAINING_ENTITY.tube(1))).toEqual(scene.css(TRAINING_TUBE_COLOR, 0.45));
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.tube(1), "lower"))).toEqual(scene.css(TRAINING_WORD_COLOR));
    // the heading word now recedes like every other word's, and its marks are gone
    expect(scene.colourOf(TRAINING_ENTITY.funnel(1))).toEqual(scene.css(TRAINING_FUNNEL_COLOR, 0.18 * 0.3));
    expect(scene.entities.getById(TRAINING_ENTITY.focusFlown("starts"))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.focusPathLabel("fast"))).toBeUndefined();
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.funnel(1)))).not.toEqual(scene.css(TRAINING_WORD_COLOR));
    expect(scene.colourOf(TRAINING_ENTITY.path(TRAINING_ENTITY.turn(1), "slow"))).toEqual(scene.css(TRAINING_TURN_COLOR, 0.9 * 0.3));
    // nothing was rebuilt
    original.forEach((entity) => expect(scene.entities.getById(entity.id)).toBe(entity));

    // a second click on the selected word clears it: every envelope back at rest
    fireEvent.click(band(DESCEND_TO_LAND));
    expect(scene.colourOf(TRAINING_ENTITY.tube(1))).toEqual(scene.css(TRAINING_TUBE_COLOR, 0.28));
    expect(scene.colourOf(TRAINING_ENTITY.funnel(1))).toEqual(scene.css(TRAINING_FUNNEL_COLOR, 0.18));
    expect(scene.colourOf(TRAINING_ENTITY.path(TRAINING_ENTITY.turn(1), "slow"))).toEqual(scene.css(TRAINING_TURN_COLOR, 0.9));
    expect(scene.entities.getById(TRAINING_ENTITY.focusIssue)).toBeUndefined();
    expect(scene.requestRender).toHaveBeenCalled();
  });

  it("gives the clearance its capture turn and corridor; a dashed edge turns yellow, dashed, and comes back", async () => {
    const scene = await setup();
    const edgeId = TRAINING_ENTITY.edge(TRAINING_ENTITY.captureTurn);
    const edge = () => scene.entities.getById(edgeId)!.polyline!;
    const now = Cesium.JulianDate.now();
    const restColour = scene.colourOf(edgeId);
    expect(edge().material).toBeInstanceOf(Cesium.PolylineDashMaterialProperty);
    expect(edge().width!.getValue(now)).toBe(1.5);
    fireEvent.click(band(/^approach cleared — cleared to join the final/));
    expect(edge().material).toBeInstanceOf(Cesium.PolylineDashMaterialProperty);
    expect(scene.colourOf(edgeId)).toEqual(scene.css(TRAINING_WORD_COLOR));
    expect(edge().width!.getValue(now)).toBe(3);
    expect(scene.colourOf(TRAINING_ENTITY.corridor)).toEqual(scene.css(TRAINING_CORRIDOR_COLOR, 0.45));
    expect(scene.colourOf(TRAINING_ENTITY.corridorAxis)).toEqual(scene.css(TRAINING_WORD_COLOR));
    // the heading words are another column's: they recede; the clearance names its capture turn's
    // paths but has no turn flown of its own to mark
    expect(scene.colourOf(TRAINING_ENTITY.funnel(1))).toEqual(scene.css(TRAINING_FUNNEL_COLOR, 0.18 * 0.3));
    expect(scene.entities.getById(TRAINING_ENTITY.focusPathLabel("slow"))).toBeDefined();
    expect(scene.entities.getById(TRAINING_ENTITY.focusFlown("starts"))).toBeUndefined();
    fireEvent.click(band(/^approach cleared — cleared to join the final/));
    expect(edge().material).toBeInstanceOf(Cesium.PolylineDashMaterialProperty);
    expect(scene.colourOf(edgeId)).toEqual(restColour);
    expect(edge().width!.getValue(now)).toBe(1.5);
  });

  it("finds the clearance at step 0 on a straight-in flight", async () => {
    const scene = await setup(1);
    fireEvent.click(band(/^approach cleared — cleared to join the final, issued at step 0/));
    expect(scene.colourOf(TRAINING_ENTITY.corridor)).toEqual(scene.css(TRAINING_CORRIDOR_COLOR, 0.45));
    expect(scene.entities.getById(TRAINING_ENTITY.focusIssue)!.label!.text!.getValue(Cesium.JulianDate.now()))
      .toBe("approach cleared · step 0");
  });

  it("dashes a slowest turn that does not finish, and draws no line for a fastest path of one point", async () => {
    const unfinished = await setup(0, (raw) => { raw.flights[0].envelopes.approach.captureTurn.turn.slowFinished = false; });
    const slow = unfinished.entities.getById(TRAINING_ENTITY.path(TRAINING_ENTITY.captureTurn, "slow"))!.polyline!;
    expect(slow.material).toBeInstanceOf(Cesium.PolylineDashMaterialProperty);
    unfinished.unmount();
    const within = await setup(0, (raw) => {
      const turn = raw.flights[0].envelopes.approach.captureTurn.turn;
      const start = { eM: [turn.region.eM[0]], nM: [turn.region.nM[0]], lon: [turn.region.lon[0]], lat: [turn.region.lat[0]] };
      turn.fastPath = start;
      turn.headingFast = { tS: [turn.headingFast.tS[0]], deg: [270] };
      turn.headingRegion = { tS: turn.headingRegion.tS.slice(2), deg: turn.headingRegion.deg.slice(2) };
    });
    expect(within.entities.getById(TRAINING_ENTITY.path(TRAINING_ENTITY.captureTurn, "fast"))).toBeUndefined();
    expect(within.entities.getById(TRAINING_ENTITY.path(TRAINING_ENTITY.captureTurn, "slow"))).toBeDefined();
    // a path that is not drawn is not named either
    fireEvent.click(band(/^approach cleared — cleared to join the final/));
    expect(within.entities.getById(TRAINING_ENTITY.focusPathLabel("fast"))).toBeUndefined();
    expect(within.entities.getById(TRAINING_ENTITY.focusPathLabel("slow"))).toBeDefined();
    expect(within.entities.getById(TRAINING_ENTITY.path(TRAINING_ENTITY.turn(1), "fast"))).toBeDefined();
  });

  it("repaints nothing while the cursor moves inside the selected word", async () => {
    const scene = await setup();
    fireEvent.click(band(DESCEND_TO_LAND));
    const marker = scene.entities.getById(TRAINING_ENTITY.focusIssue);
    const renders = scene.requestRender.mock.calls.length;
    act(() => scene.app().setTrainingCursorS(80));
    expect(scene.entities.getById(TRAINING_ENTITY.focusIssue)).toBe(marker);
    expect(scene.requestRender.mock.calls.length).toBe(renders);
  });

  it("frames the selected flight once, not on a switch", async () => {
    const scene = await setup();
    expect(scene.camera.flyToBoundingSphere).toHaveBeenCalledTimes(1);
    act(() => scene.app().setTrainingLayer("corridor", false));
    expect(scene.camera.flyToBoundingSphere).toHaveBeenCalledTimes(1);
  });

  it("draws, by default, the turns and where they end — not the regions between them, nor the funnels", async () => {
    const scene = await setup(0, () => undefined, false);
    for (const id of [TRAINING_ENTITY.path(TRAINING_ENTITY.turn(1), "fast"), TRAINING_ENTITY.path(TRAINING_ENTITY.turn(1), "slow"),
      TRAINING_ENTITY.turnEnd(1), TRAINING_ENTITY.captureTurnEnd, TRAINING_ENTITY.corridor, TRAINING_ENTITY.tube(0)]) {
      expect(scene.entities.getById(id), id).toBeDefined();
    }
    for (const id of [TRAINING_ENTITY.turn(1), TRAINING_ENTITY.captureTurn, TRAINING_ENTITY.funnel(0), TRAINING_ENTITY.funnel(1)]) {
      expect(scene.entities.getById(id), id).toBeUndefined();
    }
  });

  it("dots the edge of a judged hold whose funnel starts too wide to say much", async () => {
    const scene = await setup();
    const now = Cesium.JulianDate.now();
    // the fixture's second hold starts 12.4 km wide; the first is a cone from a point
    const wide = scene.entities.getById(TRAINING_ENTITY.edge(TRAINING_ENTITY.funnel(1)))!.polyline!.material;
    expect(wide).toBeInstanceOf(Cesium.PolylineDashMaterialProperty);
    expect((wide as Cesium.PolylineDashMaterialProperty).dashPattern!.getValue(now)).toBe(0x3333);
    expect(scene.entities.getById(TRAINING_ENTITY.edge(TRAINING_ENTITY.funnel(0)))!.polyline!.material)
      .toBeInstanceOf(Cesium.ColorMaterialProperty);
  });

  it("follows the switches, and leaves nothing behind outside Training", async () => {
    const scene = await setup();
    act(() => scene.app().setTrainingLayer("turnPaths", false));
    expect(scene.entities.getById(TRAINING_ENTITY.path(TRAINING_ENTITY.turn(1), "fast"))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.turnEnd(1))).toBeUndefined();   // where it ends goes with the turns
    expect(scene.entities.getById(TRAINING_ENTITY.turn(1))).toBeDefined();   // the region stays
    act(() => scene.app().setTrainingLayer("turnPaths", true));
    act(() => scene.app().setTrainingLayer("turnRegions", false));
    expect(scene.entities.getById(TRAINING_ENTITY.turn(1))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.funnel(1))).toBeDefined();
    act(() => scene.app().setTrainingLayer("holdFunnels", false));
    expect(scene.entities.getById(TRAINING_ENTITY.funnel(1))).toBeUndefined();
    act(() => scene.app().setTrainingLayer("corridor", false));
    expect(scene.entities.getById(TRAINING_ENTITY.corridor)).toBeUndefined();
    // the paths have a switch of their own
    expect(scene.entities.getById(TRAINING_ENTITY.path(TRAINING_ENTITY.turn(1), "fast"))).toBeDefined();
    expect(scene.entities.getById(TRAINING_ENTITY.tube(0))).toBeDefined();
    act(() => scene.app().setTrainingLayer("vertical", false));
    expect(scene.entities.getById(TRAINING_ENTITY.tube(0))).toBeUndefined();
    act(() => scene.app().setTrainingLayer("candidates", false));
    expect(scene.entities.getById(TRAINING_ENTITY.runway("27"))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.runway("09"))).toBeDefined();   // the designated one stays
    act(() => scene.app().setMode("observe"));
    expect(scene.entities.values).toHaveLength(0);
    scene.unmount();
  });
});

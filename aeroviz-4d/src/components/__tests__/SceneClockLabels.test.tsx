import { render } from "@testing-library/react";
import * as Cesium from "cesium";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { SceneTime } from "../../utils/sceneTime";

const { app } = vi.hoisted(() => ({
  app: { viewer: null as unknown, sceneTime: null as unknown, mode: "optimize", layers: { trajectories: false } },
}));
vi.mock("../../context/AppContext", () => ({ useApp: () => app }));

import SceneClockLabels from "../SceneClockLabels";

const EPOCH = "2026-04-01T08:00:00Z";
const SCENE: SceneTime = { startUtc: "2026-05-21T17:47:18.959Z", epoch: EPOCH };
const at = (seconds: number) => Cesium.JulianDate.addSeconds(Cesium.JulianDate.fromIso8601(EPOCH), seconds, new Cesium.JulianDate());

/** The viewer's two clock widgets: Cesium's own animation view model (its real formatters) and a timeline double. */
function viewerWithWidgets() {
  const clock = new Cesium.Clock({ currentTime: at(0), shouldAnimate: false });
  const animation = new Cesium.AnimationViewModel(new Cesium.ClockViewModel(clock));
  const timeline = {
    makeLabel: vi.fn((time: Cesium.JulianDate) => `default ${Cesium.JulianDate.toIso8601(time)}`),
    zoomTo: vi.fn(), _startJulian: at(0), _endJulian: at(7200) };
  return { clock, animation: { viewModel: animation }, timeline };
}

beforeEach(() => {
  app.sceneTime = SCENE;
  app.mode = "optimize";
  app.layers = { trajectories: false };
});

describe("SceneClockLabels", () => {
  it("makes the animation widget and the timeline say the scene's real UTC time, not the display epoch's", () => {
    const viewer = viewerWithWidgets();
    const defaultDate = viewer.animation.viewModel.dateFormatter(at(125), viewer.animation.viewModel);
    app.viewer = viewer;
    render(<SceneClockLabels />);

    const model = viewer.animation.viewModel;
    expect(model.dateFormatter(at(0), model)).toBe("2026-05-21");                       // the scene's start, not 2026-04-01
    expect(model.timeFormatter(at(0), model)).toBe("17:47:18 UTC");
    expect(model.timeFormatter(at(125), model)).toBe("17:49:23 UTC");                  // start + (clock − epoch)
    expect(model.dateFormatter(at(86400), model)).toBe("2026-05-22");
    expect(viewer.timeline.makeLabel(at(125))).toBe("2026-05-21 17:49:23 UTC");
    // its ticks are drawn again, over the range it shows, with the new labels (`resize` would do nothing: the size is unchanged)
    expect(viewer.timeline.zoomTo).toHaveBeenLastCalledWith(viewer.timeline._startJulian, viewer.timeline._endJulian);
    expect(model.dateFormatter(at(125), model)).not.toBe(defaultDate);
  });

  it("changes the tick labels of the REAL timeline widget, and puts them back", () => {
    // jsdom has no canvas: the widget only clears its (empty) track canvas
    const canvas = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      { clearRect: vi.fn() } as unknown as CanvasRenderingContext2D);
    const container = document.createElement("div");
    document.body.appendChild(container);
    const clock = new Cesium.Clock({ startTime: at(0), stopTime: at(7200), currentTime: at(0), shouldAnimate: false });
    const timeline = new Cesium.Timeline(container, clock);
    timeline.resize();                                     // as the browser does once: later `resize()` calls return at once
    timeline.zoomTo(at(0), at(7200));
    const labels = () => Array.from(container.querySelectorAll(".cesium-timeline-ticLabel")).map((el) => el.textContent ?? "");
    const before = labels();
    expect(before.length).toBeGreaterThan(0);
    expect(before.every((text) => /^Apr 1 2026 \d\d:\d\d:\d\d UTC$/.test(text))).toBe(true);        // the widget's own, on the display epoch

    const viewer = { clock, animation: { viewModel: new Cesium.AnimationViewModel(new Cesium.ClockViewModel(clock)) }, timeline };
    app.viewer = viewer;
    const view = render(<SceneClockLabels />);
    const real = labels();
    expect(real.length).toBeGreaterThan(0);
    // the scene's real time (a tick at the epoch's 04:00 is 4 h before its start, 17:47:18), not the epoch's 1 Apr
    expect(real).toEqual(["2026-05-21 13:47:18 UTC"]);

    view.unmount();
    expect(labels()).toEqual(before);                                                // and the widget's own labels come back
    container.remove();
    canvas.mockRestore();
  });

  it("shows the real time on the widgets' own labels as the clock moves", () => {
    const viewer = viewerWithWidgets();
    app.viewer = viewer;
    render(<SceneClockLabels />);
    viewer.clock.currentTime = at(3600);
    viewer.clock.tick();
    expect(viewer.animation.viewModel.timeLabel).toBe("18:47:18 UTC");
    expect(viewer.animation.viewModel.dateLabel).toBe("2026-05-21");
  });

  it("puts the widgets' own formatters back when the scene goes, the panel unmounts or the scene is not drawn", () => {
    const viewer = viewerWithWidgets();
    const model = viewer.animation.viewModel;
    const original = { date: model.dateFormatter, time: model.timeFormatter, label: viewer.timeline.makeLabel };
    app.viewer = viewer;
    const view = render(<SceneClockLabels />);
    expect(model.timeFormatter).not.toBe(original.time);

    app.sceneTime = null;
    view.rerender(<SceneClockLabels />);
    expect(model.dateFormatter).toBe(original.date);
    expect(model.timeFormatter).toBe(original.time);
    expect(viewer.timeline.makeLabel).toBe(original.label);

    app.sceneTime = SCENE;
    view.rerender(<SceneClockLabels />);
    expect(model.timeFormatter).not.toBe(original.time);
    view.unmount();
    expect(model.timeFormatter).toBe(original.time);
    expect(viewer.timeline.makeLabel).toBe(original.label);

    app.mode = "evaluation";                                                            // a published scene needs its layer on
    const evaluation = render(<SceneClockLabels />);
    expect(model.timeFormatter).toBe(original.time);
    app.layers = { trajectories: true };
    evaluation.rerender(<SceneClockLabels />);
    expect(model.timeFormatter).not.toBe(original.time);
  });

  it("does not touch the widgets of a viewer that was destroyed before the panel unmounted (the app's teardown order)", () => {
    // `useCesiumViewer` destroys the viewer, then React unmounts the components: the widgets are gone
    const viewer = viewerWithWidgets();
    let destroyed = false;
    Object.assign(viewer, { isDestroyed: () => destroyed });
    viewer.timeline.zoomTo = vi.fn(() => { if (destroyed) throw new Error("the timeline was destroyed"); });
    app.viewer = viewer;
    const view = render(<SceneClockLabels />);
    destroyed = true;
    viewer.animation.viewModel.dateFormatter = () => { throw new Error("the animation widget was destroyed"); };
    expect(() => view.unmount()).not.toThrow();
  });

  it("does not touch the widgets of a REAL viewer's destroyed timeline either", () => {
    const canvas = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      { clearRect: vi.fn() } as unknown as CanvasRenderingContext2D);
    const container = document.createElement("div");
    const clock = new Cesium.Clock({ startTime: at(0), stopTime: at(7200), currentTime: at(0), shouldAnimate: false });
    const timeline = new Cesium.Timeline(container, clock);
    timeline.zoomTo(at(0), at(7200));
    let destroyed = false;
    app.viewer = { clock, isDestroyed: () => destroyed,
      animation: { viewModel: new Cesium.AnimationViewModel(new Cesium.ClockViewModel(clock)) }, timeline };
    const view = render(<SceneClockLabels />);
    timeline.destroy();                                                                 // what `viewer.destroy()` does to it
    destroyed = true;
    expect(() => view.unmount()).not.toThrow();
    canvas.mockRestore();
  });

  it("leaves the widgets alone when there is no viewer", () => {
    app.viewer = null;
    expect(() => render(<SceneClockLabels />)).not.toThrow();
  });
});

/**
 * ExperimentPicker.tsx
 * --------------------
 * The Experiments source's model picker. A native <select> gives each run one line of text,
 * which is why the old picker read as a wall of run-grammar strings with no structure and no
 * reason; this is a trigger that opens a browser instead: each campaign a collapsible heading
 * WITH its question, each run WITH its intent, and the hovered run's full structured
 * parameters beside the list (`ExperimentDetails`).
 *
 * Portalled into document.body: the left dock carries `backdrop-filter`, which makes it the
 * containing block of — and clips — any `position: fixed` descendant.
 */

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, RefObject } from "react";
import { createPortal } from "react-dom";
import ExperimentDetails from "./ExperimentDetails";
import {
  experimentGroups,
  experimentMatches,
  type ExperimentOption,
} from "../utils/trajectoryResultSources";

interface BrowserPlacement {
  left: number;
  top: number;
  width: number;
  height: number;
}

const BROWSER_MAX_WIDTH = 860;
const BROWSER_MAX_HEIGHT = 660;
const VIEWPORT_MARGIN = 12;
/** Past the trigger's edge: the left dock's inner padding + scrollbar, so the browser opens
 *  beside the panel instead of covering its edge. */
const PANEL_CLEARANCE = 28;

/** Beside the panel when the viewport has room, else pinned inside the right edge. */
function placeBrowser(trigger: HTMLElement): BrowserPlacement {
  const rect = trigger.getBoundingClientRect();
  const width = Math.min(BROWSER_MAX_WIDTH, window.innerWidth - 2 * VIEWPORT_MARGIN);
  const height = Math.min(BROWSER_MAX_HEIGHT, window.innerHeight - 2 * VIEWPORT_MARGIN);
  const beside = rect.right + PANEL_CLEARANCE;
  const left = beside + width <= window.innerWidth - VIEWPORT_MARGIN
    ? beside
    : Math.max(VIEWPORT_MARGIN, window.innerWidth - VIEWPORT_MARGIN - width);
  const top = Math.min(
    Math.max(VIEWPORT_MARGIN, rect.top - 48),
    window.innerHeight - VIEWPORT_MARGIN - height,
  );
  return { left, top, width, height };
}

interface ExperimentPickerProps {
  /** Already ordered: by campaign, then by the ranking metric or the name. */
  experiments: ExperimentOption[];
  activeId: string;
  onSelect: (id: string) => void;
  /** The ranking metric as text ("ADE mean 1,322 m"), or null when the list is unranked. */
  metricText: (value: number | null | undefined) => string | null;
}

export default function ExperimentPicker({
  experiments,
  activeId,
  onSelect,
  metricText,
}: ExperimentPickerProps) {
  const labelId = useId();
  const valueId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [placement, setPlacement] = useState<BrowserPlacement | null>(null);
  const active = experiments.find((experiment) => experiment.id === activeId) ?? null;
  const activeMetric = active ? metricText(active.metricValue) : null;

  const open = placement !== null;

  const close = useCallback(() => {
    setPlacement(null);
    triggerRef.current?.focus();
  }, []);

  useEffect(() => {
    const trigger = triggerRef.current;
    if (!open || !trigger) return undefined;
    const reposition = () => setPlacement(placeBrowser(trigger));
    window.addEventListener("resize", reposition);
    return () => window.removeEventListener("resize", reposition);
  }, [open]);

  return (
    <div className="experiment-picker">
      <span id={labelId} className="experiment-picker-label">Experiment model</span>
      <button
        ref={triggerRef}
        type="button"
        className="experiment-picker-trigger"
        aria-labelledby={`${labelId} ${valueId}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={(event) => setPlacement(open ? null : placeBrowser(event.currentTarget))}
      >
        <span id={valueId} className="experiment-picker-trigger-text">
          {active ? (
            <>
              <span className="experiment-picker-trigger-group">{active.groupTitle}</span>
              <span className="experiment-picker-trigger-run">{active.runName}</span>
              {active.variantLabel ? (
                <span className="experiment-picker-trigger-variant">{active.variantLabel}</span>
              ) : null}
              {activeMetric ? (
                <span className="experiment-picker-trigger-metric">{activeMetric}</span>
              ) : null}
            </>
          ) : (
            <span className="experiment-picker-trigger-run">Choose an experiment…</span>
          )}
        </span>
        <span className="experiment-picker-trigger-chevron" aria-hidden="true">›</span>
      </button>
      {placement
        ? createPortal(
            <ExperimentBrowser
              experiments={experiments}
              activeId={activeId}
              placement={placement}
              trigger={triggerRef}
              metricText={metricText}
              onSelect={(id) => {
                onSelect(id);
                close();
              }}
              onClose={close}
            />,
            document.body,
          )
        : null}
    </div>
  );
}

interface ExperimentBrowserProps {
  experiments: ExperimentOption[];
  activeId: string;
  placement: BrowserPlacement;
  trigger: RefObject<HTMLButtonElement>;
  metricText: (value: number | null | undefined) => string | null;
  onSelect: (id: string) => void;
  onClose: () => void;
}

function ExperimentBrowser({
  experiments,
  activeId,
  placement,
  trigger,
  metricText,
  onSelect,
  onClose,
}: ExperimentBrowserProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [previewId, setPreviewId] = useState(activeId);
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => {
    const activeGroup = experiments.find((experiment) => experiment.id === activeId)?.group;
    return new Set(activeGroup ? [activeGroup] : []);
  });
  const filtering = query.trim() !== "";
  const campaignCount = useMemo(() => experimentGroups(experiments).length, [experiments]);
  const groups = useMemo(
    () => experimentGroups(experiments.filter((experiment) => experimentMatches(experiment, query))),
    [experiments, query],
  );
  // The previewed run follows the list: a run the list no longer shows (filtered out, or in a
  // collapsed campaign) is not described.
  const visible = groups
    .filter((group) => filtering || expanded.has(group.id))
    .flatMap((group) => group.experiments);
  const preview = visible.find((experiment) => experiment.id === previewId) ?? visible[0] ?? null;

  // Open on the run on the map: its campaign is expanded, and may sit far down the list.
  useEffect(() => {
    searchRef.current?.focus();
    const list = listRef.current;
    const active = list?.querySelector<HTMLElement>(".experiment-browser-run.is-active");
    if (list && active) list.scrollTop = Math.max(0, active.offsetTop - list.clientHeight / 3);
  }, []);

  // Escape, or a press anywhere outside the browser and its trigger, closes. `pointerdown` in
  // the CAPTURE phase, not `mousedown`: Cesium cancels the canvas's pointerdown, and a browser
  // then never fires the compatibility mousedown — a press on the map would not close it.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    function onPointerDown(event: Event) {
      const target = event.target as Node;
      if (dialogRef.current?.contains(target) || trigger.current?.contains(target)) return;
      onClose();
    }
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown, true);
    };
  }, [onClose, trigger]);

  function toggleGroup(id: string): void {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  // Up/Down walks the visible runs: Down from the filter box or a campaign heading goes to the
  // next run below it, Up from the first run returns to the filter box.
  function onKeyDown(event: ReactKeyboardEvent<HTMLDivElement>): void {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const runs = [
      ...(dialogRef.current?.querySelectorAll<HTMLButtonElement>(".experiment-browser-run") ?? []),
    ];
    const focused = document.activeElement;
    const current = runs.indexOf(focused as HTMLButtonElement);
    let next: HTMLElement | undefined;
    if (current >= 0) {
      next = event.key === "ArrowDown"
        ? runs[current + 1]
        : runs[current - 1] ?? searchRef.current ?? undefined;
    } else if (event.key === "ArrowDown" && focused) {
      next = runs.find(
        (run) => focused.compareDocumentPosition(run) & Node.DOCUMENT_POSITION_FOLLOWING,
      );
    }
    if (next) {
      event.preventDefault();
      next.focus();
    }
  }

  return (
    <div
      ref={dialogRef}
      className="experiment-browser"
      role="dialog"
      aria-label="Experiment browser"
      style={placement}
      onKeyDown={onKeyDown}
    >
      <header className="experiment-browser-header">
        <div className="experiment-browser-titles">
          <h3>Experiments</h3>
          <p>
            {experiments.length} runs in {campaignCount} campaigns · hover or arrow to preview,
            click to show on the map
          </p>
        </div>
        <input
          ref={searchRef}
          type="search"
          className="experiment-browser-search"
          placeholder="Filter by run, intent or parameter…"
          aria-label="Filter experiments"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <button type="button" className="experiment-browser-close" onClick={onClose}>
          Close
        </button>
      </header>
      <div className="experiment-browser-body">
        <nav ref={listRef} className="experiment-browser-list" aria-label="Experiment campaigns">
          {groups.length === 0 ? (
            <p className="experiment-browser-empty">No run matches “{query.trim()}”.</p>
          ) : null}
          {groups.map((group) => {
            const open = filtering || expanded.has(group.id);
            return (
              <section
                key={group.id}
                className={`experiment-browser-group${open ? " is-open" : ""}`}
              >
                <button
                  type="button"
                  className="experiment-browser-group-header"
                  aria-expanded={open}
                  onClick={() => toggleGroup(group.id)}
                >
                  <span className="experiment-browser-chevron" aria-hidden="true">›</span>
                  <span className="experiment-browser-group-title">
                    {group.title}
                    {group.title !== group.id ? <code>{group.id}</code> : null}
                  </span>
                  <span className="experiment-browser-group-count">{group.experiments.length}</span>
                </button>
                {open ? (
                  <>
                    <p className="experiment-browser-group-intent">
                      {group.intent ?? "No intent recorded for this campaign."}
                    </p>
                    <ul className="experiment-browser-runs">
                      {group.experiments.map((experiment) => {
                        const metric = metricText(experiment.metricValue);
                        const classes = ["experiment-browser-run"];
                        if (experiment.id === activeId) classes.push("is-active");
                        if (experiment.id === preview?.id) classes.push("is-preview");
                        return (
                          <li key={experiment.id}>
                            <button
                              type="button"
                              className={classes.join(" ")}
                              aria-current={experiment.id === activeId ? "true" : undefined}
                              onMouseEnter={() => setPreviewId(experiment.id)}
                              onFocus={() => setPreviewId(experiment.id)}
                              onClick={() => onSelect(experiment.id)}
                            >
                              <span className="experiment-browser-run-name">{experiment.runName}</span>
                              {experiment.variantLabel ? (
                                <span className="experiment-browser-run-variant">
                                  {experiment.variantLabel}
                                </span>
                              ) : null}
                              <span className="experiment-browser-run-intent">
                                {experiment.intent?.variant ?? experiment.intent?.run ??
                                  "No intent recorded."}
                              </span>
                              {metric ? (
                                <span className="experiment-browser-run-metric">{metric}</span>
                              ) : null}
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </>
                ) : null}
              </section>
            );
          })}
        </nav>
        <div className="experiment-browser-detail">
          {preview ? <ExperimentDetails experiment={preview} /> : null}
        </div>
      </div>
    </div>
  );
}

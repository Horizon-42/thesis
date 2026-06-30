import { AEROVIZ_BACKEND_URL } from "./pilotClient";

/**
 * Worker-session lifecycle client.
 *
 * The backend keeps its casadi solver worker resident (warm) while the Optimize
 * or Compare tab is open, and decommissions it (freeing hundreds of MB) when the
 * tab closes. The frontend signals that lifecycle: open on tab enter, close on
 * tab leave, plus a `sendBeacon` close on page unload (which fires even when the
 * tab/window is closed and React's unmount cleanup would not run).
 *
 * These calls are best-effort hints: a missed open just means the next solve
 * pays the usual spawn cost, and a missed close is reclaimed by the backend's
 * idle watchdog. So failures are swallowed rather than surfaced to the user.
 */
export type WorkerSessionKind = "optimizer" | "comparison";

const SESSION_PATHS: Record<WorkerSessionKind, { open: string; close: string }> = {
  optimizer: {
    open: "/optimization/session/open",
    close: "/optimization/session/close",
  },
  comparison: {
    open: "/dynamics-comparison/session/open",
    close: "/dynamics-comparison/session/close",
  },
};

export async function openWorkerSession(kind: WorkerSessionKind): Promise<void> {
  await postSession(SESSION_PATHS[kind].open);
}

export async function closeWorkerSession(kind: WorkerSessionKind): Promise<void> {
  await postSession(SESSION_PATHS[kind].close);
}

/**
 * Fire a close on page unload. `navigator.sendBeacon` is delivered as the page
 * goes away (unlike a normal `fetch`, which the browser may cancel), so a closed
 * tab/window still releases the worker. Sent with no body — a "simple" POST that
 * needs no CORS preflight; the backend ignores the (empty) payload.
 */
export function beaconCloseWorkerSession(kind: WorkerSessionKind): void {
  if (typeof navigator === "undefined" || !navigator.sendBeacon) {
    return;
  }
  navigator.sendBeacon(`${AEROVIZ_BACKEND_URL}${SESSION_PATHS[kind].close}`);
}

async function postSession(path: string): Promise<void> {
  try {
    await fetch(`${AEROVIZ_BACKEND_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
      // Let the request outlive a same-tick unmount (best effort).
      keepalive: true,
    });
  } catch {
    // Best-effort lifecycle hint; the backend idle watchdog is the leak backstop.
  }
}

/**
 * useMeasuredWidth.ts
 * -------------------
 * The width of an element, followed as it resizes — for SVG drawn one unit to one pixel (the Training sentence bar and
 * windows). A callback ref, so an element that mounts later (a view that renders nothing until it has data) is measured
 * when it appears. ``fallback`` until measured, and in jsdom, which has no layout; never below ``min``.
 */

import { useLayoutEffect, useState } from "react";

export default function useMeasuredWidth(min: number, fallback: number): [(node: HTMLElement | null) => void, number] {
  const [node, setNode] = useState<HTMLElement | null>(null);
  const [width, setWidth] = useState<number>(fallback);
  useLayoutEffect(() => {
    if (node === null || typeof ResizeObserver === "undefined") return;
    const measure = () => setWidth(Math.max(node.clientWidth, min));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [node, min]);
  return [setNode, width];
}

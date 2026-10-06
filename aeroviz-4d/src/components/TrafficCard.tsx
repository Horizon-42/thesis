/**
 * TrafficCard.tsx
 * ---------------
 * One two-line card of the Optimize task's multi-aircraft panel: a row of its scenario list or of its result. The dock is 242 px
 * wide, so a row is two lines of text — no columns — that wrap at spaces and never mid-word. A `li` of a listbox: a click, or
 * Enter or Space, picks it.
 */

import type { KeyboardEvent } from "react";
import TypeCode from "./TypeCode";

interface TrafficCardProps {
  lines: readonly [string, string];
  /** The text of its tooltip. */
  title: string;
  selected: boolean;
  /** Called by a click and by Enter or Space. */
  pick: () => void;
  /** The class of the second line (the result's is a smaller, muted one). */
  detailClass?: string;
  /** The tail of the second line that must stay on one line (the runways: `23L, 23R, 32` never wraps alone). */
  keep?: string;
  /** The ICAO type code in the first line (` · B738`) and its plain name, when the app has one: the code gets the name as its title. */
  typeCode?: string;
  typeName?: string;
}

export default function TrafficCard({ lines, title, selected, pick, detailClass = "", keep, typeCode, typeName }: TrafficCardProps) {
  const marker = typeCode === undefined ? null : ` · ${typeCode}`;
  const at = marker === null ? -1 : lines[0].indexOf(marker);
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      pick();
    }
  };
  return (
    <li
      role="option"
      aria-selected={selected}
      tabIndex={0}
      className={`traffic-job-card${selected ? " selected" : ""}`}
      title={title}
      onClick={pick}
      onKeyDown={onKeyDown}
    >
      <span className="traffic-job-card-line">
        {marker === null || typeName === undefined || at < 0
          ? lines[0]
          : <>{lines[0].slice(0, at)} · <TypeCode code={typeCode!} name={typeName} />{lines[0].slice(at + marker.length)}</>}
      </span>
      <span className={`traffic-job-card-line ${detailClass}`.trim()}>
        {keep !== undefined && lines[1].endsWith(keep)
          ? <>{lines[1].slice(0, -keep.length)}<span className="traffic-job-keep">{keep}</span></>
          : lines[1]}
      </span>
    </li>
  );
}

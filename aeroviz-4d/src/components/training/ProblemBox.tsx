/**
 * ProblemBox.tsx
 * --------------
 * What failed, in the Training panel: a title in words and the reader's own refusal beneath it, selectable for a bug
 * report — never a blank where a file should be.
 */

import type { ReactNode } from "react";

export default function ProblemBox({ title, detail, children }: { title: string; detail: string; children?: ReactNode }) {
  return (
    <div className="training-problem" role="alert">
      <p className="training-empty-title">{title}</p>
      <p className="training-problem-detail">{detail}</p>
      {children}
    </div>
  );
}

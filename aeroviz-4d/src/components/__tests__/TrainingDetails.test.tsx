/**
 * TrainingDetails: the details page's shell — a tab per section, a section with nothing to show disabled with its reason,
 * Up / Down between the sections that have a body, the rest of the page inert while it is open, closed by Escape, the ×
 * or the backdrop (never by a click inside), and the focus back on the control that opened it.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import TrainingDetails, { type TrainingDetailsSection } from "../training/TrainingDetails";

const SECTIONS: TrainingDetailsSection[] = [
  { id: "a", title: "First", body: <p>first body</p> },
  { id: "b", title: "Second", body: null, absent: "none published for this set" },
  { id: "c", title: "Third", body: <p>third body</p> },
];

function Page({ onClose, opener, sections = SECTIONS, start = "a" }: {
  onClose: () => void; opener: HTMLElement; sections?: TrainingDetailsSection[]; start?: string;
}) {
  const [section, setSection] = useState(start);
  return (
    <>
      <span data-testid="shown">{section}</span>
      <TrainingDetails context="KXXX · set" sections={sections} sectionId={section} onSection={setSection} onClose={onClose}
        opener={opener} />
    </>
  );
}

afterEach(() => {
  document.querySelectorAll("button[data-opener]").forEach((button) => button.remove());
});

function opener(): HTMLButtonElement {
  const button = document.createElement("button");
  button.dataset.opener = "";
  document.body.appendChild(button);
  return button;
}

describe("TrainingDetails", () => {
  it("shows the chosen section, and says why a section without a body is disabled", () => {
    render(<Page onClose={vi.fn()} opener={opener()} />);
    expect(screen.getByRole("tabpanel").textContent).toBe("Firstfirst body");
    const second = screen.getByRole("tab", { name: /Second/ }) as HTMLButtonElement;
    expect(second.disabled).toBe(true);
    expect(second.title).toBe("none published for this set");
  });

  it("moves between the sections with a body on Up / Down, wrapping", () => {
    render(<Page onClose={vi.fn()} opener={opener()} />);
    fireEvent.keyDown(screen.getByRole("tab", { name: "First" }), { key: "ArrowDown" });
    expect(screen.getByRole("tabpanel").textContent).toBe("Thirdthird body");
    expect(document.activeElement).toBe(screen.getByRole("tab", { name: "Third" }));
    fireEvent.keyDown(screen.getByRole("tab", { name: "Third" }), { key: "ArrowDown" });
    expect(screen.getByRole("tabpanel").textContent).toBe("Firstfirst body");
    fireEvent.keyDown(screen.getByRole("tab", { name: "First" }), { key: "ArrowUp" });
    expect(screen.getByRole("tabpanel").textContent).toBe("Thirdthird body");
  });

  it("gives way to the first section when the one asked for has no body, and says so to its owner", () => {
    render(<Page onClose={vi.fn()} opener={opener()} start="b" />);
    expect(screen.getByRole("tabpanel").textContent).toBe("Firstfirst body");
    expect(screen.getByTestId("shown").textContent).toBe("a");
  });

  it("makes the rest of the page inert while open, and gives the focus back to its opener when closed", () => {
    const button = opener();
    const { unmount } = render(<Page onClose={vi.fn()} opener={button} />);
    expect(document.activeElement).toBe(screen.getByRole("dialog"));
    expect(button.hasAttribute("inert")).toBe(true);
    expect(screen.getByRole("dialog").closest("[inert]")).toBeNull();
    unmount();
    expect(button.hasAttribute("inert")).toBe(false);
    expect(document.activeElement).toBe(button);
  });

  it("closes on the backdrop, the × and Escape — not on a click inside, nor on another button", () => {
    const onClose = vi.fn();
    render(<Page onClose={onClose} opener={opener()} />);
    const backdrop = document.querySelector(".training-details-backdrop")!;
    fireEvent.mouseDown(screen.getByRole("tabpanel"));
    fireEvent.mouseDown(backdrop, { button: 2 });
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.mouseDown(backdrop);
    fireEvent.click(screen.getByRole("button", { name: "Close the details" }));
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(3);
  });
});

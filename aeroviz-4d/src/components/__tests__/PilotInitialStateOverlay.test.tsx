import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import PilotInitialStateOverlay from "../PilotInitialStateOverlay";

const defaultState = {
  lon: -78.7873,
  lat: 35.878659,
  altM: 1000,
  speedMps: 120,
  headingDeg: 245,
  flightPathDeg: -3.2,
  massKg: 10000,
};

describe("PilotInitialStateOverlay", () => {
  it("renders initial aircraft controls outside the flight ops panel flow", () => {
    const { container } = renderOverlay();
    const overlay = screen.getByRole("complementary", { name: "Initial aircraft setup" });

    expect(overlay.classList.contains("pilot-initial-overlay")).toBe(true);
    expect(container.contains(overlay)).toBe(false);
    expect(document.body.contains(overlay)).toBe(true);
    expect(
      (screen.getByRole("button", { name: "Place Aircraft" }) as HTMLButtonElement).disabled,
    ).toBe(false);
    expect(
      (screen.getByRole("textbox", { name: "Gamma deg" }) as HTMLInputElement).value,
    ).toBe("-3.2");
  });

  it("normalizes decimal input to English period notation", () => {
    const onFieldChange = vi.fn();
    renderOverlay({ onFieldChange });

    const gammaInput = screen.getByRole("textbox", { name: "Gamma deg" });
    fireEvent.change(gammaInput, { target: { value: "-4,5" } });
    expect((gammaInput as HTMLInputElement).value).toBe("-4.5");

    fireEvent.blur(gammaInput);
    expect(onFieldChange).toHaveBeenCalledWith("flightPathDeg", -4.5, -15, 15);
  });
});

function renderOverlay({
  onFieldChange = vi.fn(),
}: {
  onFieldChange?: ReturnType<typeof vi.fn>;
} = {}) {
  return render(
    <PilotInitialStateOverlay
      open
      isPlacing={false}
      state={defaultState}
      disabled={false}
      onClose={vi.fn()}
      onPlaceToggle={vi.fn()}
      onFieldChange={onFieldChange}
    />,
  );
}

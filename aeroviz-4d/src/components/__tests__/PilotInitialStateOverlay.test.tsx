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
    expect(
      (screen.getByRole("textbox", { name: "Mass kg" }) as HTMLInputElement).value,
    ).toBe("10000");
  });

  it("normalizes decimal input to English period notation", () => {
    const onFieldChange = vi.fn();
    renderOverlay({ onFieldChange });

    const gammaInput = screen.getByRole("textbox", { name: "Gamma deg" });
    fireEvent.change(gammaInput, { target: { value: "-4,5" } });
    expect((gammaInput as HTMLInputElement).value).toBe("-4.5");

    fireEvent.blur(gammaInput);
    expect(onFieldChange).toHaveBeenCalledWith(
      "flightPathDeg",
      -4.5,
      Number.NEGATIVE_INFINITY,
      Number.POSITIVE_INFINITY,
    );
  });

  it("increments initial state fields with keyboard arrows", () => {
    const onFieldChange = vi.fn();
    renderOverlay({ onFieldChange });

    const altInput = screen.getByRole("textbox", { name: "Alt m" });
    const psiInput = screen.getByRole("textbox", { name: "Psi deg" });
    const gammaInput = screen.getByRole("textbox", { name: "Gamma deg" });
    const massInput = screen.getByRole("textbox", { name: "Mass kg" });

    expect(altInput.getAttribute("step")).toBe("1");
    expect(psiInput.getAttribute("step")).toBe("1");
    expect(gammaInput.getAttribute("step")).toBe("1");
    expect(massInput.getAttribute("step")).toBe("1");

    fireEvent.keyDown(altInput, { key: "ArrowUp" });
    fireEvent.keyDown(psiInput, { key: "ArrowDown" });
    fireEvent.keyDown(gammaInput, { key: "ArrowUp" });
    fireEvent.keyDown(massInput, { key: "ArrowUp" });

    expect(onFieldChange).toHaveBeenCalledWith("altM", 1001, -500, 14000);
    expect(onFieldChange).toHaveBeenCalledWith(
      "headingDeg",
      244,
      Number.NEGATIVE_INFINITY,
      Number.POSITIVE_INFINITY,
    );
    expect(onFieldChange).toHaveBeenCalledWith(
      "flightPathDeg",
      -2.2,
      Number.NEGATIVE_INFINITY,
      Number.POSITIVE_INFINITY,
    );
    expect(onFieldChange).toHaveBeenCalledWith(
      "massKg",
      10001,
      1,
      Number.POSITIVE_INFINITY,
    );
  });

  it("does not clamp psi or gamma values", () => {
    const onFieldChange = vi.fn();
    renderOverlay({ onFieldChange });

    const psiInput = screen.getByRole("textbox", { name: "Psi deg" });
    const gammaInput = screen.getByRole("textbox", { name: "Gamma deg" });

    fireEvent.change(psiInput, { target: { value: "-12" } });
    fireEvent.blur(psiInput);
    fireEvent.change(gammaInput, { target: { value: "24" } });
    fireEvent.blur(gammaInput);

    expect(onFieldChange).toHaveBeenCalledWith(
      "headingDeg",
      -12,
      Number.NEGATIVE_INFINITY,
      Number.POSITIVE_INFINITY,
    );
    expect(onFieldChange).toHaveBeenCalledWith(
      "flightPathDeg",
      24,
      Number.NEGATIVE_INFINITY,
      Number.POSITIVE_INFINITY,
    );
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

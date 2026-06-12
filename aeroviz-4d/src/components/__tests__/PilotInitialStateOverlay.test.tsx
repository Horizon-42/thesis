import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import type { PilotAircraftConfig, PilotResetState } from "../../pilot/pilotClient";
import PilotInitialStateOverlay from "../PilotInitialStateOverlay";

const defaultState: PilotResetState = {
  lon: -78.7873,
  lat: 35.878659,
  altM: 1000,
  speedMps: 120,
  headingDeg: 245,
  flightPathDeg: -3.2,
  massKg: 78000,
  aircraftType: "A320",
};

const aircraftConfigs: PilotAircraftConfig[] = [
  {
    code: "A320",
    name: "Airbus A320-200",
    category: "narrow_body",
    massKg: 78000,
    wingAreaM2: 122.6,
  },
  {
    code: "B77W",
    name: "Boeing 777-300ER",
    category: "wide_body",
    massKg: 351530,
    wingAreaM2: 436.8,
  },
];

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
      (screen.getByRole("combobox", { name: "Type" }) as HTMLSelectElement).value,
    ).toBe("A320");
    expect(screen.getByText("78000")).toBeTruthy();
  });

  it("emits the selected aircraft type", () => {
    const onAircraftTypeChange = vi.fn();
    renderOverlay({ onAircraftTypeChange });

    fireEvent.change(screen.getByRole("combobox", { name: "Type" }), {
      target: { value: "B77W" },
    });

    expect(onAircraftTypeChange).toHaveBeenCalledWith("B77W");
  });

  it("can display the backend-configured mass for the selected aircraft type", () => {
    function Wrapper() {
      const [state, setState] = useState(defaultState);

      return (
        <PilotInitialStateOverlay
          open
          isPlacing={false}
          state={state}
          aircraftConfigs={aircraftConfigs}
          disabled={false}
          onClose={vi.fn()}
          onPlaceToggle={vi.fn()}
          onFieldChange={vi.fn()}
          onAircraftTypeChange={(aircraftType) => {
            const aircraft = aircraftConfigs.find((config) => config.code === aircraftType);
            if (!aircraft) return;
            setState((current) => ({
              ...current,
              aircraftType: aircraft.code,
              massKg: aircraft.massKg,
            }));
          }}
        />
      );
    }

    render(<Wrapper />);

    expect(screen.getByText("78000")).toBeTruthy();
    fireEvent.change(screen.getByRole("combobox", { name: "Type" }), {
      target: { value: "B77W" },
    });

    expect(screen.getByText("351530")).toBeTruthy();
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

    expect(altInput.getAttribute("step")).toBe("1");
    expect(psiInput.getAttribute("step")).toBe("1");
    expect(gammaInput.getAttribute("step")).toBe("1");

    fireEvent.keyDown(altInput, { key: "ArrowUp" });
    fireEvent.keyDown(psiInput, { key: "ArrowDown" });
    fireEvent.keyDown(gammaInput, { key: "ArrowUp" });

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
  onAircraftTypeChange = vi.fn(),
}: {
  onFieldChange?: ReturnType<typeof vi.fn>;
  onAircraftTypeChange?: ReturnType<typeof vi.fn>;
} = {}) {
  return render(
    <PilotInitialStateOverlay
      open
      isPlacing={false}
      state={defaultState}
      aircraftConfigs={aircraftConfigs}
      disabled={false}
      onClose={vi.fn()}
      onPlaceToggle={vi.fn()}
      onFieldChange={onFieldChange}
      onAircraftTypeChange={onAircraftTypeChange}
    />,
  );
}

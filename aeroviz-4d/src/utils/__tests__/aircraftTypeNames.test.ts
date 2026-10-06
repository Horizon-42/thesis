import { describe, expect, it } from "vitest";
import { NO_AIRCRAFT_TYPE_NAMES, aircraftTypeNames, typeNameOf } from "../aircraftTypeNames";

describe("aircraftTypeNames", () => {
  const names = aircraftTypeNames([{ code: "A320", name: "Airbus A320-200" }, { code: "C172", name: "Cessna 172" }]);

  it("maps the ICAO code of every type the aircraft catalog lists to its plain name", () => {
    expect(typeNameOf(names, "A320")).toBe("Airbus A320-200");
    expect(typeNameOf(names, "C172")).toBe("Cessna 172");
  });

  it("has no name for a type the catalog does not list, an untyped aircraft, or a code that is a property of every object", () => {
    expect(typeNameOf(names, "B738")).toBeUndefined();
    expect(typeNameOf(names, null)).toBeUndefined();
    expect(typeNameOf(names, "constructor")).toBeUndefined();
    expect(typeNameOf(NO_AIRCRAFT_TYPE_NAMES, "A320")).toBeUndefined();
  });
});

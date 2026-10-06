import { NO_AIRCRAFT_TYPE_NAMES } from "../aircraftTypeNames";
import { describe, expect, it } from "vitest";
import type { TrafficScenarioArrival, TrafficScenarioBlock, TrafficScenarioCatalog } from "../../data/trafficJobs";
import catalogMirror from "../../data/__tests__/fixtures/trafficScenarioCatalog.json";
import {
  CATALOG_MEANING,
  DEFAULT_TRAFFIC_SORT,
  TRAFFIC_LOSS_KINDS,
  LIGHT_AIRCRAFT_CATEGORY,
  LIST_PAGE_ROWS,
  NO_COMMANDABLE_REASON,
  arrivalDetail,
  TRAFFIC_SORT_CHOICES,
  CONTROLLABLE_TITLE,
  LIGHT_AIRCRAFT_TITLE,
  arrivalCardLines,
  blockCardLines,
  blockRunwaysText,
  blockDetail,
  blocksHeader,
  catalogHeader,
  catalogLimitNote,
  listedRows,
  lossKindName,
  lossKindsText,
  lossSecondsTitle,
  secondsInLoss,
  sortChoiceValue,
  percentOfMinimum,
  runwayChoices,
  selectionText,
  utcMinute,
  utcStamp,
} from "../trafficScenarios";

// the fixture mirrors what `traffic_scenarios.py` writes (see `data/__tests__/trafficJobs.test.ts`)
const catalog = catalogMirror as TrafficScenarioCatalog;
const m1 = catalog.m1;
const hour = catalog.m2["3600"];

const keys = (rows: readonly TrafficScenarioArrival[]) => rows.map((r) => r.callsign ?? r.flightKey.split("_")[0]);
const m1Rows = (runway: string, sort = DEFAULT_TRAFFIC_SORT) =>
  listedRows(m1, (r) => [r.runway], (r) => r.landingUtc, runway, sort);
const blockRows = (rows: readonly TrafficScenarioBlock[], runway: string, sort = DEFAULT_TRAFFIC_SORT) =>
  listedRows(rows, (r) => r.runways, (r) => r.startUtc, runway, sort);

describe("the names of the losses", () => {
  it("name every kind the judge reports in plain words, and show an unknown one as it is", () => {
    expect([...TRAFFIC_LOSS_KINDS]).toEqual(["in_trail", "diagonal", "radar_or_vertical", "at_threshold"]);
    for (const kind of TRAFFIC_LOSS_KINDS) expect(lossKindName(kind)).not.toBe(kind);
    expect(lossKindName("something_new")).toBe("something_new");
    expect(lossKindsText(["in_trail", "radar_or_vertical"])).toBe(
      "too close in trail on one final; under the radar minimum and not vertically separated");
  });
});

describe("the header", () => {
  it("says how many of the judged arrivals have a loss of separation they answer for, how often the records were checked and when", () => {
    expect(catalogHeader(catalog)).toBe(
      "5 of 6 arrivals have a loss of separation they answer for in their record (the records were checked every 1 s, 2026-10-06T09:11:42Z)");
    expect(CATALOG_MEANING).toContain("recorded aircraft as flown");
    expect(CATALOG_MEANING).toContain("not a promise that the optimized flight has one");
  });

  it("says, for blocks, how many blocks of the length hold an arrival and in what order they come", () => {
    expect(blocksHeader(catalog, 1800)).toBe(
      "4 blocks of 30 min with an arrival; sorted by the seconds their arrivals spend in loss of separation (the records were checked every 1 s)");
    expect(blocksHeader(catalog, 900)).toContain("5 blocks of 15 min with an arrival; sorted by the seconds their arrivals spend in loss of separation");
    expect(blocksHeader(catalog, 3600)).toContain("4 blocks of 60 min");
  });

  it("states a partial census, and only a partial one", () => {
    expect(catalogLimitNote(catalog)).toBeNull();
    expect(catalogLimitNote({ ...catalog, config: { ...catalog.config, limit: 300 } })).toContain("first 300 arrivals");
  });
});

describe("the light aircraft and the pages of a list", () => {
  it("names the lightest CWT category and a page of 100 rows (a MIRROR of the CWT categories, pinned)", () => {
    expect(LIGHT_AIRCRAFT_CATEGORY).toBe("I");
    expect(m1.filter((r) => r.category === LIGHT_AIRCRAFT_CATEGORY).map((r) => r.callsign)).toEqual(["N2412P"]);
    expect(LIST_PAGE_ROWS).toBe(100);
  });

  it("makes an M1 card of two lines: who it is, then when it lands and its seconds in loss", () => {
    const [light, swa, nameless] = m1;
    expect(arrivalCardLines(swa, 1)).toEqual(["SWA3131 · B38M · 05L", "2026-05-01 00:06 UTC · 28 seconds in loss"]);
    expect(arrivalCardLines(nameless, 1)).toEqual(["N123AB · — · 05R", "2026-05-01 00:21 UTC · 6 seconds in loss"]);   // no callsign, no type
    expect(arrivalCardLines(light, 1)[0]).toBe("N2412P · C172 · 05L");
    expect(arrivalCardLines({ ...swa, lossInstants: 1 }, 1)[1]).toBe("2026-05-01 00:06 UTC · 1 second in loss");
    // a census that checked every 2 s: 28 check instants are 56 seconds
    expect(arrivalCardLines(swa, 2)[1]).toBe("2026-05-01 00:06 UTC · 56 seconds in loss");
    expect(secondsInLoss(28, 0.5)).toBe(14);
  });

  it("makes an M2 card of two lines: its span, then arrivals, how many are controllable, seconds in loss and runways", () => {
    expect(blockCardLines(hour[1], 3600, 1)).toEqual([
      "2026-05-01 00:00–01:00 UTC", "2 arrivals · 2 controllable · 34 seconds in loss · 05L, 05R"]);
    expect(blockCardLines({ ...hour[0], arrivals: 1, commandable: 0, lossInstants: 1, runways: ["32"] }, 900, 1)).toEqual([
      "2026-05-01 12:00–12:15 UTC", "1 arrival · 0 controllable · 1 second in loss · 32"]);
    expect(blockCardLines({ ...hour[0], startUtc: "2026-05-21T23:45:00Z" }, 1800, 1)[0]).toBe(
      "2026-05-21 23:45–2026-05-22 00:15 UTC");                                                    // past midnight: both days
  });

  it("joins a block's runways with commas, in one piece the card keeps on one line", () => {
    const three = { ...hour[0], runways: ["23L", "23R", "32"] };
    expect(blockRunwaysText(three)).toBe("23L, 23R, 32");
    expect(blockCardLines(three, 3600, 1)[1].endsWith(" · 23L, 23R, 32")).toBe(true);
  });
});

describe("the jargon, in plain words with its title", () => {
  it("says what seconds in loss, light aircraft and controllable are", () => {
    expect(lossSecondsTitle(1)).toBe(
      "Seconds in loss: check instants, 1 s apart, at which the aircraft is closer than its separation minimum and answers " +
      "for it (a block adds up its arrivals)");
    expect(LIGHT_AIRCRAFT_TITLE).toBe("CWT category I: the lightest wake-turbulence category");
    expect(CONTROLLABLE_TITLE).toContain("has an aircraft dynamics model");
  });
});

describe("the formats", () => {
  it("shows the tightest loss as a percentage of the minimum", () => {
    expect(percentOfMinimum(0.648716318740827)).toBe("65 %");
    expect(percentOfMinimum(1)).toBe("100 %");
  });

  it("shows a UTC stamp to the second or the minute", () => {
    expect(utcStamp("2026-05-01T00:06:59Z")).toBe("2026-05-01 00:06:59");
    expect(utcMinute("2026-05-01T00:06:59Z")).toBe("2026-05-01 00:06");
  });
});

describe("what the narrow tables leave to the row's title and the Selected line", () => {
  const [, swa, nameless] = m1;                           // the light aircraft (40 instants) is first

  it("holds, for an arrival, its type, tightest loss, kinds, runway, landing and recorded aircraft", () => {
    expect(arrivalDetail(swa, 1)).toBe(
      "28 seconds in loss, tightest 65 % of the minimum (under the radar minimum and not vertically separated)" +
      " · type B38M · runway 05L · landing 2026-05-01 00:06:59 UTC · 12 recorded aircraft in its window");
    expect(arrivalDetail(nameless, 1)).toContain("type unknown");
    expect(arrivalDetail({ ...swa, lossInstants: 1, recordedAircraft: 1 }, 1)).toContain("1 second in loss, ");
    expect(arrivalDetail({ ...swa, lossInstants: 1, recordedAircraft: 1 }, 1)).toContain("1 recorded aircraft in its window");
  });

  it("names the type in the selection and the details where the app has a plain name for it, and only then", () => {
    const names = new Map([[swa.type!, "Boeing 737-800"]]);
    expect(arrivalDetail(swa, 1, "Boeing 737-800")).toContain(` · type ${swa.type} (Boeing 737-800) · runway `);
    expect(arrivalDetail(swa, 1)).toContain(` · type ${swa.type} · runway `);
    expect(selectionText("m1", swa, undefined, 1800, 1, names)).toBe(`Selected: SWA3131 — ${arrivalDetail(swa, 1, "Boeing 737-800")}`);
    expect(selectionText("m1", swa, undefined, 1800, 1, NO_AIRCRAFT_TYPE_NAMES)).toBe(`Selected: SWA3131 — ${arrivalDetail(swa, 1)}`);
    expect(selectionText("m1", nameless, undefined, 1800, 1, names)).toContain("type unknown");
  });

  it("holds, for a block, its end, arrivals, how many are controllable, seconds in loss and runways", () => {
    expect(blockDetail(hour[1], 3600, 1)).toBe(
      "Block 2026-05-01 00:00–01:00 UTC — 2 arrivals, 2 controllable, 34 seconds in loss, runways 05L, 05R");
  });

  it("names the row picked in full, or says what to pick", () => {
    expect(selectionText("m1", nameless, undefined, 1800, 1, NO_AIRCRAFT_TYPE_NAMES)).toBe(`Selected: N123AB — ${arrivalDetail(nameless, 1)}`);
    expect(selectionText("m2", undefined, hour[2], 3600, 1, NO_AIRCRAFT_TYPE_NAMES)).toBe(`Selected: ${blockDetail(hour[2], 3600, 1)}`);
    expect(selectionText("m1", undefined, undefined, 1800, 1, NO_AIRCRAFT_TYPE_NAMES)).toBe("Pick the arrival to optimize in its recorded traffic.");
    expect(selectionText("m2", undefined, undefined, 1800, 1, NO_AIRCRAFT_TYPE_NAMES)).toBe("Pick the block to optimize.");
    expect(NO_COMMANDABLE_REASON).toBe("no arrival in this block is controllable (has an aircraft dynamics model)");
  });
});

describe("the sort and the runway filter", () => {
  it("starts in the catalog's order: the most loss instants first", () => {
    expect(m1Rows("").map((r) => r.lossInstants)).toEqual([40, 28, 6, 6, 3]);
    expect(keys(m1Rows(""))).toEqual(["N2412P", "SWA3131", "N123AB", "DAL88", "AAL2634"]);
  });

  it("sorts by loss instants either way, equal rows keeping the catalog's order", () => {
    expect(keys(m1Rows("", { key: "loss", descending: false }))).toEqual(["AAL2634", "N123AB", "DAL88", "SWA3131", "N2412P"]);
  });

  it("sorts by landing time either way", () => {
    expect(keys(m1Rows("", { key: "time", descending: false }))).toEqual(["SWA3131", "N123AB", "N2412P", "DAL88", "AAL2634"]);
    expect(keys(m1Rows("", { key: "time", descending: true }))).toEqual(["AAL2634", "DAL88", "N2412P", "N123AB", "SWA3131"]);
  });

  it("leaves the catalog's rows alone", () => {
    const before = keys(m1);
    m1Rows("", { key: "time", descending: true });
    expect(keys(m1)).toEqual(before);
  });

  it("filters the arrivals by their runway and the blocks by any runway they hold", () => {
    expect(keys(m1Rows("05L"))).toEqual(["N2412P", "SWA3131", "AAL2634"]);
    expect(keys(m1Rows("32"))).toEqual(["DAL88"]);
    expect(m1Rows("09")).toEqual([]);
    expect(blockRows(hour, "32").map((b) => b.startUtc)).toEqual(["2026-05-01T14:00:00Z"]);
    expect(blockRows(hour, "05R").map((b) => b.startUtc)).toEqual(["2026-05-01T00:00:00Z", "2026-05-02T09:00:00Z"]);
    expect(blockRows(hour, "").map((b) => b.lossInstants)).toEqual([40, 34, 9, 0]);
  });

  it("lists the runways of the rows, sorted", () => {
    expect(runwayChoices(m1, (r) => [r.runway])).toEqual(["05L", "05R", "32"]);
    expect(runwayChoices(hour, (r) => r.runways)).toEqual(["05L", "05R", "32"]);
  });

  it("offers the sorts the select shows, each a column and a direction, and names a sort by its value", () => {
    expect(TRAFFIC_SORT_CHOICES.map((c) => [c.value, c.label])).toEqual([
      ["loss-desc", "Most losses"], ["loss-asc", "Fewest losses"], ["time-asc", "Earliest"], ["time-desc", "Latest"]]);
    // the select is half a 242 px dock: a label past 13 characters would be clipped
    expect(Math.max(...TRAFFIC_SORT_CHOICES.map((c) => c.label.length))).toBeLessThanOrEqual(13);
    expect(TRAFFIC_SORT_CHOICES[0].sort).toEqual(DEFAULT_TRAFFIC_SORT);                           // the catalog's own order first
    for (const choice of TRAFFIC_SORT_CHOICES) expect(sortChoiceValue(choice.sort)).toBe(choice.value);
  });
});

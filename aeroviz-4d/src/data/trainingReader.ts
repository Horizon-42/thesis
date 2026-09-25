/**
 * trainingReader.ts
 * -----------------
 * The ONE reader every Training file is parsed with: the manifest and a set's sample (`trainingSample.ts`), the
 * overlays drawn over a set (`trainingOverlays.ts`) and the live executor's answer (`trainingAutopilot.ts`). A field is
 * read or refused by its path — `sample.flights[3].signals.tS has 40 values, expected 41` — and `attempt` turns a
 * refusal into a `Parsed` problem. One `Refusal` class, so a refusal thrown by one file's reader inside another's
 * `attempt` is a problem named on screen, never a crash.
 *
 * Nothing here knows what a word is: the files' own rules live beside their shapes.
 */

export type Parsed<T> = { ok: true; value: T } | { ok: false; problem: string };

/** A field refused by name. */
export class Refusal extends Error {}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Reads one object, naming the path of every field it refuses. */
export class Reader {
  constructor(private readonly source: Record<string, unknown>, readonly where: string) {}

  static of(value: unknown, where: string): Reader {
    if (!isRecord(value)) throw new Refusal(`${where} is not an object`);
    return new Reader(value, where);
  }

  fail(message: string): never {
    throw new Refusal(`${this.where}: ${message}`);
  }

  at(key: string): string {
    return `${this.where}.${key}`;
  }

  raw(key: string): unknown {
    return this.source[key];
  }

  private refuse(key: string, what: string): never {
    throw new Refusal(`${this.at(key)} is ${JSON.stringify(this.source[key])}, ${what}`);
  }

  child(key: string): Reader {
    return Reader.of(this.source[key], this.at(key));
  }

  nullableChild(key: string): Reader | null {
    return this.source[key] === null ? null : this.child(key);
  }

  /** Each item of a list, read as an object. */
  children(key: string): Reader[] {
    return this.list(key).map((item, index) => Reader.of(item, this.at(`${key}[${index}]`)));
  }

  string(key: string): string {
    const value = this.source[key];
    if (typeof value !== "string" || value.length === 0) this.refuse(key, "not a non-empty string");
    return value;
  }

  nullableString(key: string): string | null {
    return this.source[key] === null ? null : this.string(key);
  }

  /** One of a closed list of names. */
  oneOf<T extends string>(key: string, names: readonly T[]): T {
    const value = this.string(key);
    if (!(names as readonly string[]).includes(value)) this.refuse(key, `not one of ${names.join(", ")}`);
    return value as T;
  }

  number(key: string): number {
    const value = this.source[key];
    if (!isNumber(value)) this.refuse(key, "not a number");
    return value;
  }

  nullableNumber(key: string): number | null {
    return this.source[key] === null ? null : this.number(key);
  }

  integer(key: string, low: number, high: number): number {
    const value = this.number(key);
    if (!Number.isInteger(value) || value < low || value > high) this.refuse(key, `not a whole number in ${low}…${high}`);
    return value;
  }

  nullableInteger(key: string, low: number, high: number): number | null {
    return this.source[key] === null ? null : this.integer(key, low, high);
  }

  /** A whole number of at least ``low`` — a count, with no upper bound to pretend to. */
  count(key: string, low = 0): number {
    const value = this.number(key);
    if (!Number.isInteger(value) || value < low) this.refuse(key, `not a whole number of at least ${low}`);
    return value;
  }

  nullableCount(key: string, low = 0): number | null {
    return this.source[key] === null ? null : this.count(key, low);
  }

  /** A share of a whole: a number in [0, 1]. */
  share(key: string): number {
    const value = this.number(key);
    if (value < 0 || value > 1) this.refuse(key, "not a share in [0, 1]");
    return value;
  }

  nullableShare(key: string): number | null {
    return this.source[key] === null ? null : this.share(key);
  }

  boolean(key: string): boolean {
    const value = this.source[key];
    if (typeof value !== "boolean") this.refuse(key, "not true/false");
    return value;
  }

  nullableBoolean(key: string): boolean | null {
    return this.source[key] === null ? null : this.boolean(key);
  }

  numbers(key: string, length?: number): number[] {
    const value = this.source[key];
    if (!Array.isArray(value) || !value.every(isNumber)) throw new Refusal(`${this.at(key)} is missing or not a list of numbers`);
    if (length !== undefined && value.length !== length) {
      throw new Refusal(`${this.at(key)} has ${value.length} values, expected ${length}`);
    }
    return value;
  }

  nullableNumbers(key: string, length?: number): number[] | null {
    return this.source[key] === null ? null : this.numbers(key, length);
  }

  /** 0/1 per row, as booleans. */
  flags(key: string, length: number): boolean[] {
    return this.numbers(key, length).map((value, index) => {
      if (value !== 0 && value !== 1) throw new Refusal(`${this.at(key)}[${index}] is ${value}, not 0/1`);
      return value === 1;
    });
  }

  /** Probabilities: numbers in [0, 1]. */
  probabilities(key: string, length: number): number[] {
    const values = this.numbers(key, length);
    const outside = values.findIndex((value) => value < 0 || value > 1);
    if (outside >= 0) throw new Refusal(`${this.at(key)}[${outside}] is ${values[outside]}, not a probability`);
    return values;
  }

  /** An ascending [low, high] pair. */
  range(key: string): [number, number] {
    const values = this.numbers(key, 2);
    if (values[0] > values[1]) throw new Refusal(`${this.at(key)} is inverted: ${values[0]} above ${values[1]}`);
    return [values[0], values[1]];
  }

  strings(key: string): string[] {
    const value = this.source[key];
    if (!Array.isArray(value) || !value.every((item) => typeof item === "string" && item.length > 0)) {
      throw new Refusal(`${this.at(key)} is missing or not a list of names`);
    }
    return value as string[];
  }

  /** A list of names that must be exactly ``names``, in that order. */
  sameNames(key: string, names: readonly string[]): void {
    const found = this.strings(key);
    if (found.length !== names.length || found.some((name, index) => name !== names[index])) {
      throw new Refusal(`${this.at(key)} are [${found.join(", ")}], expected [${names.join(", ")}] in that order`);
    }
  }

  list(key: string): unknown[] {
    const value = this.source[key];
    if (!Array.isArray(value)) throw new Refusal(`${this.at(key)} is missing or not a list`);
    return value;
  }

  record<T>(key: string, read: (value: unknown, where: string) => T): Record<string, T> {
    return recordOf(this.source[key], this.at(key), read);
  }
}

/** An object whose every value is read by ``read``, keyed as it is; an empty key is refused. */
export function recordOf<T>(value: unknown, where: string, read: (value: unknown, where: string) => T): Record<string, T> {
  if (!isRecord(value)) throw new Refusal(`${where} is not an object`);
  return Object.fromEntries(Object.entries(value).map(([key, item]) => {
    if (key.length === 0) throw new Refusal(`${where} has an empty key`);
    return [key, read(item, `${where}.${key}`)];
  }));
}

export function asNumber(value: unknown, where: string): number {
  if (!isNumber(value)) throw new Refusal(`${where} is ${JSON.stringify(value)}, not a number`);
  return value;
}

export function asCount(value: unknown, where: string): number {
  if (!isNumber(value) || !Number.isInteger(value) || value < 0) throw new Refusal(`${where} is ${JSON.stringify(value)}, not a count`);
  return value;
}

export function attempt<T>(read: () => T): Parsed<T> {
  try {
    return { ok: true, value: read() };
  } catch (error) {
    if (error instanceof Refusal) return { ok: false, problem: error.message };
    throw error;
  }
}

/**
 * A manifest: its schema and airport, then its entries, each read on its own — a bad entry is rejected with the field
 * that failed and the rest are kept, instead of emptying the airport (the comparison picker's `.every(...)` did that
 * twice — AV6). Only a manifest that is not one fails the call.
 */
export function parseManifest<T>(
  raw: unknown,
  manifest: { name: string; schema: string; listKey: string; entryName: string },
  read: (entry: Reader) => T,
): Parsed<{ airport: string; entries: T[]; rejected: Array<{ id: string; problem: string }> }> {
  if (!isRecord(raw)) return { ok: false, problem: `the ${manifest.name} is not an object` };
  if (raw.schema !== manifest.schema) {
    return { ok: false, problem: `schema is ${JSON.stringify(raw.schema)}, expected ${JSON.stringify(manifest.schema)}` };
  }
  if (typeof raw.airport !== "string" || raw.airport.length === 0) return { ok: false, problem: "airport is missing" };
  const list = raw[manifest.listKey];
  if (!Array.isArray(list)) return { ok: false, problem: `${manifest.listKey} is not an array` };
  const entries: T[] = [];
  const rejected: Array<{ id: string; problem: string }> = [];
  list.forEach((item, position) => {
    const id = isRecord(item) && typeof item.id === "string" && item.id.length > 0 ? item.id : `${manifest.listKey}[${position}]`;
    const entry = attempt(() => read(Reader.of(item, `${manifest.entryName} ${id}`)));
    if (entry.ok) entries.push(entry.value);
    else rejected.push({ id, problem: entry.problem });
  });
  return { ok: true, value: { airport: raw.airport, entries, rejected } };
}

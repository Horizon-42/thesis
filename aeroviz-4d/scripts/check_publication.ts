/**
 * check_publication.ts — does the picker load what was just published?
 *
 *   npm run check-publication -- --airport KSMF [--airport KRDU …] [--server http://localhost:5173]
 *   npm run check-publication                      # every airport under public/data/airports
 *
 * Two layers, each answering a question a bare `curl … 200` cannot:
 *   1. DISK — `comparison/categories.json` and every drawable category's
 *      `comparison_index.json` are run through the frontend's OWN type guards
 *      (`src/utils/checkPublication.ts`); the CZML files and the evaluation report each index
 *      names must exist. A category the picker would reject is named with the field and the
 *      value, because one such category empties the whole airport's list.
 *   2. SERVER (with `--server`) — the same files fetched from the RUNNING dev server: a JSON
 *      file must come back as JSON, every CZML file must not come back as HTML. A server
 *      started before the publication answers a new category directory with the SPA fallback
 *      (`aeroviz-4d/CLAUDE.md`, "a RUNNING dev server never sees a newly published category"),
 *      which this layer reports as "restart the dev server".
 *
 * Exit status 1 on any error-level finding, 2 on a usage error. Runs under vite-node (no
 * build, no browser); `npm run typecheck:scripts` type-checks it.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  isComparisonCategoriesManifest,
  isComparisonIndex,
  isDrawableComparisonCategory,
} from "../src/data/airportData";
import {
  checkCategoriesManifest,
  checkComparisonIndex,
  checkTrainingIndex,
  checkTrainingSample,
  checkTrainingSetAgrees,
  indexCzmlFiles,
  type PublicationFinding,
} from "../src/utils/checkPublication";
import { parseTrainingIndex, parseTrainingSample } from "../src/data/trainingSample";

const FRONTEND_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const AIRPORTS_ROOT = path.join(FRONTEND_ROOT, "public", "data", "airports");
const USAGE =
  "usage: npm run check-publication -- [--airport ICAO]... [--server http://localhost:5173]\n" +
  "  no --airport: every airport under public/data/airports with a comparison/categories.json";

class UsageError extends Error {}

interface Options {
  airports: string[];
  server: string | null;
}

function flagValue(argv: string[], position: number): string {
  const value = argv[position + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new UsageError(`${argv[position]} needs a value`);
  }
  return value;
}

function parseArgs(argv: string[]): Options {
  const options: Options = { airports: [], server: null };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--airport") {
      options.airports.push(flagValue(argv, i));
      i += 1;
    } else if (arg === "--server") {
      options.server = flagValue(argv, i).replace(/\/+$/, "");
      i += 1;
    } else {
      throw new UsageError(arg === "--help" || arg === "-h" ? "" : `unknown argument ${arg}`);
    }
  }
  if (options.airports.length === 0) {
    options.airports = readdirSync(AIRPORTS_ROOT, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && (
        existsSync(path.join(AIRPORTS_ROOT, entry.name, "comparison", "categories.json"))
        || existsSync(path.join(AIRPORTS_ROOT, entry.name, "training", "index.json"))
      ))
      .map((entry) => entry.name)
      .sort();
  }
  return options;
}

function readJson(file: string): unknown {
  return JSON.parse(readFileSync(file, "utf8"));
}

/**
 * The first bytes of a response. vite's static server honours the Range header (206, 256
 * bytes); the SPA fallback and a compressing proxy ignore it, so a full body is read only
 * when it is small, else one chunk of the stream.
 */
async function responseHead(response: Response): Promise<string> {
  const length = Number(response.headers.get("content-length") ?? NaN);
  if (response.status === 206 || (Number.isFinite(length) && length <= 4096)) {
    return (await response.text()).trimStart();
  }
  const reader = response.body?.getReader();
  if (!reader) return "";
  const { value } = await reader.read();
  await reader.cancel();
  return new TextDecoder().decode(value ?? new Uint8Array()).trimStart();
}

/** What the running server answers for one published file: null, or why the app would fail. */
async function served(url: string, kind: "json" | "czml"): Promise<string | null> {
  let response: Response;
  try {
    response = await fetch(url, { headers: { Range: "bytes=0-255" } });
  } catch (error) {
    return `unreachable: ${error instanceof Error ? error.message : String(error)}`;
  }
  const contentType = response.headers.get("content-type") ?? "";
  const head = await responseHead(response);
  if (!response.ok) return `HTTP ${response.status}`;
  if (contentType.includes("text/html") || head.startsWith("<")) {
    return "answered with HTML (the SPA fallback): the dev server booted before this file existed — restart it";
  }
  if (kind === "json" && !contentType.includes("json")) return `content-type ${contentType || "(none)"}, not JSON`;
  if (kind === "czml" && !head.startsWith("[")) return "does not start like a CZML array";
  return null;
}

interface AirportReport {
  listed: number;
  trainingSets: number;
  findings: PublicationFinding[];
}

/**
 * The Training export (design §7, T8). It is OPTIONAL — `public/data` is git-ignored and most
 * airports have none — so its absence is a count of zero, never a finding. What is a finding is
 * an export that exists and is half-written: the panel greys a bad set out and carries on
 * (§4.5 ③), so nothing on screen shouts, and this is what shouts.
 */
async function checkTraining(airport: string, server: string | null): Promise<AirportReport> {
  const trainingDir = path.join(AIRPORTS_ROOT, airport, "training");
  const manifestFile = path.join(trainingDir, "index.json");
  if (!existsSync(manifestFile)) return { listed: 0, trainingSets: 0, findings: [] };

  const findings: PublicationFinding[] = [];
  const manifest = readJson(manifestFile);
  findings.push(...checkTrainingIndex(manifest));
  const parsed = parseTrainingIndex(manifest);
  if (!parsed.ok) return { listed: 0, trainingSets: 0, findings };

  const serverRoot = server ? `${server}/data/airports/${airport}/training` : null;
  if (serverRoot) {
    const problem = await served(`${serverRoot}/index.json`, "json");
    if (problem) findings.push({ level: "error", message: `server: training/index.json ${problem}` });
  }

  for (const entry of parsed.value.sets) {
    const sampleFile = path.join(trainingDir, entry.file);
    if (!existsSync(sampleFile)) {
      findings.push({ level: "error", category: entry.id, message: `${entry.file} is listed but missing on disk` });
      continue;
    }
    const sample = readJson(sampleFile);
    findings.push(...checkTrainingSample(entry.id, sample));
    const read = parseTrainingSample(sample);
    if (read.ok) findings.push(...checkTrainingSetAgrees(entry, read.value));
    if (serverRoot) {
      const problem = await served(`${serverRoot}/${entry.file}`, "json");
      if (problem) findings.push({ level: "error", category: entry.id, message: `server: ${entry.file} ${problem}` });
    }
  }
  return { listed: 0, trainingSets: parsed.value.sets.length, findings };
}

async function checkAirport(airport: string, server: string | null): Promise<AirportReport> {
  const findings: PublicationFinding[] = [];
  const comparisonDir = path.join(AIRPORTS_ROOT, airport, "comparison");
  const manifestFile = path.join(comparisonDir, "categories.json");
  if (!existsSync(manifestFile)) {
    // An airport can be published with a Training export and no comparison at all; only an
    // airport asked for BY NAME with neither is a mistake, and `checkTraining` says so.
    const level = existsSync(path.join(AIRPORTS_ROOT, airport, "training", "index.json")) ? "warn" : "error";
    return { listed: 0, trainingSets: 0, findings: [{ level, message: `${manifestFile} does not exist` }] };
  }
  const manifest = readJson(manifestFile);
  findings.push(...checkCategoriesManifest(manifest));
  if (!isComparisonCategoriesManifest(manifest)) return { listed: 0, trainingSets: 0, findings };

  const serverRoot = server ? `${server}/data/airports/${airport}/comparison` : null;
  if (serverRoot) {
    const problem = await served(`${serverRoot}/categories.json`, "json");
    if (problem) findings.push({ level: "error", message: `server: categories.json ${problem}` });
  }

  for (const category of manifest.categories) {
    // The picker filters with the same guard: a category with no groups (the observed
    // baseline row) is listed for the evaluation panel and never fetches an index.
    if (!isDrawableComparisonCategory(category)) continue;
    const categoryDir = path.join(comparisonDir, category.dir);
    const indexFile = path.join(categoryDir, "comparison_index.json");
    if (!existsSync(indexFile)) {
      findings.push({ level: "error", category: category.key, message: `${indexFile} does not exist` });
      continue;
    }
    const index = readJson(indexFile);
    findings.push(...checkComparisonIndex(category, index));
    if (!isComparisonIndex(index)) continue;

    const czmlFiles = indexCzmlFiles(index);
    for (const file of [...czmlFiles, index.evaluationReport]) {
      if (!existsSync(path.join(categoryDir, file))) {
        findings.push({ level: "error", category: category.key, message: `${file} is referenced but missing on disk` });
      }
    }
    if (serverRoot) {
      const jsonFiles: Array<[string, string]> = [
        ["comparison_index.json", "comparison_index.json"],
        ["the evaluation report", index.evaluationReport],
      ];
      for (const [what, file] of jsonFiles) {
        const problem = await served(`${serverRoot}/${category.dir}/${file}`, "json");
        if (problem) findings.push({ level: "error", category: category.key, message: `server: ${what} ${problem}` });
      }
      for (const file of czmlFiles) {
        const problem = await served(`${serverRoot}/${category.dir}/${file}`, "czml");
        if (problem) findings.push({ level: "error", category: category.key, message: `server: ${file} ${problem}` });
      }
    }
  }
  return { listed: manifest.categories.length, trainingSets: 0, findings };
}

async function main(): Promise<number> {
  const options = parseArgs(process.argv.slice(2));
  let errors = 0;
  for (const airport of options.airports) {
    const comparison = await checkAirport(airport, options.server);
    const training = await checkTraining(airport, options.server);
    const findings = [...comparison.findings, ...training.findings];
    const errorCount = findings.filter((finding) => finding.level === "error").length;
    errors += errorCount;
    const verdict = errorCount === 0 ? "picker loads" : "picker BROKEN";
    const scope = options.server ? " (disk + server)" : " (disk only)";
    const listed = comparison.listed;
    const trained = training.trainingSets ? `, ${training.trainingSets} Training sets` : "";
    console.log(`${airport}: ${listed} categories listed${trained}, ${errorCount} errors, ${findings.length - errorCount} warnings — ${verdict}${scope}`);
    for (const finding of findings) {
      console.log(`  ${finding.level.toUpperCase().padEnd(5)} ${finding.category ? `[${finding.category}] ` : ""}${finding.message}`);
    }
  }
  return errors === 0 ? 0 : 1;
}

main().then(
  (code) => {
    process.exitCode = code;
  },
  (error) => {
    if (error instanceof UsageError) {
      if (error.message) console.error(error.message);
      console.error(USAGE);
      process.exitCode = error.message ? 2 : 0;
      return;
    }
    console.error(error);
    process.exitCode = 2;
  },
);

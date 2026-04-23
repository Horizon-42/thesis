export class JsonFetchError extends Error {
  readonly url: string;
  readonly status?: number;
  readonly missingAsset: boolean;

  constructor(
    message: string,
    options: { url: string; status?: number; missingAsset?: boolean },
  ) {
    super(message);
    this.name = "JsonFetchError";
    this.url = options.url;
    this.status = options.status;
    this.missingAsset = options.missingAsset ?? false;
  }
}

function looksLikeHtml(text: string): boolean {
  const trimmed = text.trimStart().toLowerCase();
  return trimmed.startsWith("<!doctype") || trimmed.startsWith("<html") || trimmed.startsWith("<");
}

export function isMissingJsonAsset(error: unknown): boolean {
  return (
    error instanceof JsonFetchError &&
    (error.missingAsset || error.status === 404)
  );
}

export async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  const contentType = response.headers.get("content-type") ?? "";

  if (!response.ok) {
    throw new JsonFetchError(`HTTP ${response.status} loading ${url}`, {
      url,
      status: response.status,
      missingAsset: response.status === 404,
    });
  }

  const text = await response.text();
  if (contentType.includes("text/html") || looksLikeHtml(text)) {
    throw new JsonFetchError(
      `Expected JSON from ${url}, but received HTML. The airport data file is probably missing.`,
      { url, missingAsset: true },
    );
  }

  try {
    return JSON.parse(text) as T;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new JsonFetchError(`Invalid JSON from ${url}: ${message}`, { url });
  }
}

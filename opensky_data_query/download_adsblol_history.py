#!/usr/bin/env python3
"""Download one day of ADSB.lol global history release assets.

The ADSB.lol globe history repository is published as daily GitHub releases.
Each daily release contains split tar assets for the whole globe. This script
downloads the assets for one date so downstream code can filter by airport
locally.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GITHUB_REPO = "adsblol/globe_history_2026"
GITHUB_API_ROOT = "https://api.github.com"
BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ReleaseChoice:
    """A GitHub release candidate with downloadable assets."""

    tag: str
    name: str
    html_url: str
    total_size: int
    assets: list[dict[str, Any]]


def parse_day(value: str) -> date:
    """Parse YYYY-MM-DD or YYYY.MM.DD date text."""
    for fmt in ("%Y-%m-%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("date must be YYYY-MM-DD or YYYY.MM.DD")


def candidate_tags(day: date, source: str) -> list[str]:
    """Return ADSB.lol release tags to try for a date and source preference."""
    prefix = f"v{day:%Y.%m.%d}-planes-readsb"
    replicas_by_source = {
        "auto": ("prod-0", "prod-1", "staging-0"),
        "prod": ("prod-0", "prod-1"),
        "staging": ("staging-0",),
    }
    return [f"{prefix}-{replica}" for replica in replicas_by_source[source]]


def github_json(url: str, *, token: str | None = None) -> Any:
    """Fetch and parse JSON from the GitHub API."""
    req = Request(url, method="GET")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "opensky-data-query-adsblol-downloader")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 404:
            return None
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API HTTP {e.code} for {url}: {body[:300]}") from e
    except URLError as e:
        raise RuntimeError(f"GitHub API network error for {url}: {e}") from e


def fetch_release(tag: str, *, token: str | None = None) -> ReleaseChoice | None:
    """Fetch a release by tag, returning None when the tag does not exist."""
    url = f"{GITHUB_API_ROOT}/repos/{GITHUB_REPO}/releases/tags/{tag}"
    data = github_json(url, token=token)
    if data is None:
        return None
    assets = [
        asset
        for asset in data.get("assets", [])
        if asset.get("browser_download_url") and asset.get("name")
    ]
    assets.sort(key=lambda item: str(item["name"]))
    return ReleaseChoice(
        tag=str(data.get("tag_name") or tag),
        name=str(data.get("name") or ""),
        html_url=str(data.get("html_url") or ""),
        total_size=sum(int(asset.get("size") or 0) for asset in assets),
        assets=assets,
    )


def choose_release(day: date, *, source: str, token: str | None = None) -> tuple[ReleaseChoice, list[ReleaseChoice]]:
    """Select the largest available release for a date/source preference."""
    found: list[ReleaseChoice] = []
    for tag in candidate_tags(day, source):
        release = fetch_release(tag, token=token)
        if release is not None and release.assets:
            found.append(release)
    if not found:
        tried = ", ".join(candidate_tags(day, source))
        raise RuntimeError(f"No downloadable ADSB.lol release found for {day.isoformat()}; tried {tried}")
    found.sort(key=lambda item: (item.total_size, item.tag), reverse=True)
    return found[0], found


def human_size(num_bytes: int) -> str:
    """Format a byte count for terminal output."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def download_asset(
    asset: dict[str, Any],
    *,
    dest_dir: Path,
    token: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download one release asset with resumable .part file support."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = str(asset["name"])
    url = str(asset["browser_download_url"])
    expected_size = int(asset.get("size") or 0)
    dest = dest_dir / name
    part = dest.with_name(f"{dest.name}.part")

    if dest.exists() and not overwrite:
        current_size = dest.stat().st_size
        if expected_size <= 0 or current_size == expected_size:
            print(f"[ADSB.lol] skip existing {name} ({human_size(current_size)})", flush=True)
            return dest
        raise RuntimeError(
            f"Existing file has unexpected size: {dest} "
            f"got {current_size}, expected {expected_size}. Use --overwrite to replace it."
        )

    if overwrite:
        dest.unlink(missing_ok=True)
        part.unlink(missing_ok=True)

    resume_at = part.stat().st_size if part.exists() else 0
    if expected_size > 0 and resume_at > expected_size:
        part.unlink()
        resume_at = 0

    req = Request(url, method="GET")
    req.add_header("User-Agent", "opensky-data-query-adsblol-downloader")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if resume_at > 0:
        req.add_header("Range", f"bytes={resume_at}-")

    print(f"[ADSB.lol] downloading {name} from byte {resume_at}", flush=True)
    try:
        with urlopen(req, timeout=60) as resp:
            append = resume_at > 0 and getattr(resp, "status", 200) == 206
            mode = "ab" if append else "wb"
            if resume_at > 0 and not append:
                resume_at = 0
            downloaded = resume_at
            last_report = time.monotonic()
            with part.open(mode) as f:
                while True:
                    chunk = resp.read(BUFFER_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 5.0:
                        print_download_progress(name, downloaded, expected_size)
                        last_report = now
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Download HTTP {e.code} for {name}: {body[:300]}") from e
    except URLError as e:
        raise RuntimeError(f"Download network error for {name}: {e}") from e

    final_size = part.stat().st_size
    if expected_size > 0 and final_size != expected_size:
        raise RuntimeError(f"Incomplete download for {name}: got {final_size}, expected {expected_size}")
    part.replace(dest)
    print(f"[ADSB.lol] downloaded {name} ({human_size(final_size)})", flush=True)
    return dest


def print_download_progress(name: str, downloaded: int, expected_size: int) -> None:
    """Print one concise progress line."""
    if expected_size > 0:
        pct = downloaded * 100.0 / expected_size
        print(f"[ADSB.lol] {name}: {pct:.1f}% ({human_size(downloaded)}/{human_size(expected_size)})", flush=True)
    else:
        print(f"[ADSB.lol] {name}: {human_size(downloaded)}", flush=True)


class SplitFileReader:
    """Read multiple split files as one sequential binary stream."""

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.index = 0
        self.current: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        chunks: list[bytes] = []
        remaining = size
        while size < 0 or remaining > 0:
            if self.current is None:
                if self.index >= len(self.paths):
                    break
                self.current = self.paths[self.index].open("rb")
                self.index += 1
            read_size = BUFFER_SIZE if size < 0 else min(remaining, BUFFER_SIZE)
            chunk = self.current.read(read_size)
            if not chunk:
                self.current.close()
                self.current = None
                continue
            chunks.append(chunk)
            if size > 0:
                remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None


def extract_split_tar(paths: list[Path], *, extract_dir: Path) -> int:
    """Extract downloaded split tar assets without writing a combined tar file."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    reader = SplitFileReader(paths)
    count = 0
    try:
        with tarfile.open(fileobj=reader, mode="r|*") as tar:
            for member in tar:
                target = (root / member.name).resolve()
                if not str(target).startswith(str(root) + os.sep) and target != root:
                    raise RuntimeError(f"Unsafe tar member path: {member.name}")
                tar.extract(member, path=root)
                count += 1
                if count % 10000 == 0:
                    print(f"[ADSB.lol] extracted {count} members", flush=True)
    finally:
        reader.close()
    print(f"[ADSB.lol] extracted {count} members to {extract_dir}", flush=True)
    return count


def write_manifest(
    *,
    path: Path,
    day: date,
    selected: ReleaseChoice,
    candidates: list[ReleaseChoice],
    asset_paths: list[Path],
    extracted_to: Path | None,
    extracted_count: int | None,
) -> None:
    """Write a small manifest for the local downloaded day."""
    manifest = {
        "schema_version": "adsblol-globe-history-download-v1",
        "repo": GITHUB_REPO,
        "date": day.isoformat(),
        "selected_release": {
            "tag": selected.tag,
            "name": selected.name,
            "html_url": selected.html_url,
            "total_size": selected.total_size,
        },
        "available_candidates": [
            {"tag": item.tag, "total_size": item.total_size, "asset_count": len(item.assets)}
            for item in candidates
        ],
        "assets": [
            {
                "name": str(asset.get("name")),
                "size": int(asset.get("size") or 0),
                "downloaded_path": str(path),
            }
            for asset, path in zip(selected.assets, asset_paths)
        ],
        "extracted_to": None if extracted_to is None else str(extracted_to),
        "extracted_count": extracted_count,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one ADSB.lol globe_history_2026 daily release")
    parser.add_argument("--date", required=True, type=parse_day, help="Date to download, e.g. 2026-04-19")
    parser.add_argument(
        "--source",
        choices=["auto", "prod", "staging"],
        default="auto",
        help="Release source preference. auto picks the largest available release for the date.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "outputs" / "adsblol_globe_history"),
        help="Directory for downloaded assets.",
    )
    parser.add_argument("--github-token", default=os.getenv("GITHUB_TOKEN"), help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files instead of reusing them.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected release/assets without downloading.")
    parser.add_argument("--extract", action="store_true", help="Extract the split tar after download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected, candidates = choose_release(args.date, source=args.source, token=args.github_token)
    output_root = Path(args.output_root)
    day_dir = output_root / f"{args.date:%Y.%m.%d}" / selected.tag
    assets_dir = day_dir / "assets"
    extract_dir = day_dir / "extracted"

    print(f"[ADSB.lol] selected release: {selected.tag}", flush=True)
    print(f"[ADSB.lol] release url: {selected.html_url}", flush=True)
    print(f"[ADSB.lol] total assets: {len(selected.assets)} ({human_size(selected.total_size)})", flush=True)
    for candidate in candidates:
        print(
            f"[ADSB.lol] candidate {candidate.tag}: "
            f"{len(candidate.assets)} assets, {human_size(candidate.total_size)}",
            flush=True,
        )
    for asset in selected.assets:
        print(f"[ADSB.lol] asset {asset['name']} ({human_size(int(asset.get('size') or 0))})", flush=True)

    if args.dry_run:
        return

    asset_paths = [
        download_asset(
            asset,
            dest_dir=assets_dir,
            token=args.github_token,
            overwrite=args.overwrite,
        )
        for asset in selected.assets
    ]
    extracted_count: int | None = None
    extracted_to: Path | None = None
    if args.extract:
        extracted_count = extract_split_tar(asset_paths, extract_dir=extract_dir)
        extracted_to = extract_dir

    manifest_path = day_dir / "download_manifest.json"
    write_manifest(
        path=manifest_path,
        day=args.date,
        selected=selected,
        candidates=candidates,
        asset_paths=asset_paths,
        extracted_to=extracted_to,
        extracted_count=extracted_count,
    )
    print(f"[ADSB.lol] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

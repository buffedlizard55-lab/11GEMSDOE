#!/usr/bin/env python3
"""Fetch and validate the public DrivenData GEMS leaderboard into a JSON snapshot.

This accesses only the public leaderboard page (not competition data or account pages). A
failed request, login redirect, or unexpected table leaves the existing snapshot untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SOURCE_URL = "https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/"
SCORE_RE = re.compile(r"(?<!\d)(0\.\d{1,6})(?!\d)")
RANK_RE = re.compile(r"^\s*#?\s*(\d+)\b")


class TableParser(HTMLParser):
    """Extract table rows and link text without depending on a third-party parser."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[dict]] = []
        self._table: list[dict] | None = None
        self._row: dict | None = None
        self._cell: dict | None = None
        self._anchor: dict | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "table":
            if self._table is None:
                self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = {"cells": [], "header": False}
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = {"text": [], "anchors": []}
            if tag == "th":
                self._row["header"] = True
        elif tag == "a" and self._cell is not None:
            self._anchor = {"href": attr.get("href", ""), "text": []}
        elif tag == "br" and self._cell is not None:
            self._cell["text"].append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor is not None and self._cell is not None:
            text = "".join(self._anchor["text"]).strip()
            if text:
                self._cell["anchors"].append({"href": self._anchor["href"], "text": text})
            self._anchor = None
        elif tag in {"th", "td"} and self._cell is not None and self._row is not None:
            self._cell["text"] = " ".join("".join(self._cell["text"]).split())
            self._row["cells"].append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row["cells"]:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"].append(data)
        if self._anchor is not None:
            self._anchor["text"].append(data)


def parse_leaderboard(html: str, source_url: str = SOURCE_URL, fetched_at: str | None = None) -> dict:
    parser = TableParser()
    parser.feed(html)
    parsed_rows: list[dict] = []
    header_found = False
    for table in parser.tables:
        rows = table
        rank_index = participant_index = score_index = None
        for row in rows:
            cells = row["cells"]
            cell_text = [cell["text"] for cell in cells]
            lowered = [x.lower() for x in cell_text]
            if any(x.strip().lower() == "rank" for x in cell_text) and any("participant" in x for x in lowered):
                header_found = True
                rank_index = next(i for i, x in enumerate(lowered) if x.strip() == "rank")
                participant_index = next(i for i, x in enumerate(lowered) if "participant" in x)
                score_index = next((i for i, x in enumerate(lowered) if "tversky" in x or ("score" in x and "shared" not in x)), None)
                continue
            if len(cells) < 4 or rank_index is None or participant_index is None:
                continue
            if max(rank_index, participant_index) >= len(cells):
                continue
            rank_match = RANK_RE.search(cell_text[rank_index])
            if score_index is not None and score_index < len(cell_text):
                score_match = SCORE_RE.search(cell_text[score_index])
            else:
                score_match = next((m for value in cell_text for m in [SCORE_RE.search(value)] if m), None)
            score_text = score_match.group(1) if score_match else None
            if not rank_match or score_text is None:
                continue
            # Prefer the participant cell's linked label; fall back to its visible team text.
            participant_cell = cells[participant_index]
            linked_names = [a["text"] for a in participant_cell["anchors"] if a["text"]]
            participant = linked_names[0] if linked_names else participant_cell["text"]
            participant = " ".join(participant.split())
            if not participant:
                continue
            parsed_rows.append({"rank": int(rank_match.group(1)), "participant": participant,
                                "score": score_text})
        if parsed_rows:
            break

    if not header_found or len(parsed_rows) < 5:
        raise ValueError(f"leaderboard table was not confidently parsed (header={header_found}, scored_rows={len(parsed_rows)})")
    parsed_rows.sort(key=lambda row: row["rank"])
    return {
        "source": source_url,
        "fetched_at_utc": fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "snapshot_scope": "Public leaderboard rows parsed from the official page; scores may change and are not private/final prize scores.",
        "snapshot_complete": True,
        "row_count": len(parsed_rows),
        "leader": parsed_rows[0],
        "rows": parsed_rows,
    }


def fetch(url: str = SOURCE_URL, timeout: int = 25) -> dict:
    parsed_source = urlparse(url)
    if (parsed_source.scheme != "https" or parsed_source.hostname not in {"www.drivendata.org", "drivendata.org"}
            or "leaderboard" not in parsed_source.path):
        raise ValueError("only the HTTPS public DrivenData leaderboard URL is allowed")
    request = Request(url, headers={"User-Agent": "11GEMSDOE-research-dashboard/1.0", "Accept": "text/html"})
    with urlopen(request, timeout=timeout) as response:
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        body = response.read(8 * 1024 * 1024 + 1)
    if len(body) > 8 * 1024 * 1024:
        raise ValueError("leaderboard response exceeded 8 MiB safety limit")
    final = urlparse(final_url)
    if final.scheme != "https" or final.hostname not in {"www.drivendata.org", "drivendata.org"} or "leaderboard" not in final.path:
        raise ValueError(f"unexpected redirect away from public leaderboard: {final_url}")
    if "html" not in content_type.lower():
        raise ValueError(f"expected HTML, got Content-Type {content_type!r}")
    html = body.decode("utf-8", errors="replace")
    return parse_leaderboard(html, source_url=final_url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/leaderboard.json"))
    parser.add_argument("--url", default=SOURCE_URL, help="official public leaderboard URL")
    args = parser.parse_args(argv)
    try:
        snapshot = fetch(args.url)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"ERROR: leaderboard snapshot not updated: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {args.output}: {snapshot['row_count']} scored rows; leader {snapshot['leader']['participant']} {snapshot['leader']['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

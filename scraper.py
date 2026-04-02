from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import feedparser
import requests

from settings import (
    BBC_FEEDS,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    SCRAPED_JSON,
)


@dataclass(frozen=True)
class NewsItem:
    title: str
    summary: str
    link: str
    text: str  # title + summary combined for NLP


# Parsing

def fetch_feed(label: str, url: str) -> list[NewsItem]:
    items: list[NewsItem] = []
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            if not title:
                continue
            combined = f"{title}. {summary}" if summary else title
            items.append(NewsItem(
                title=title,
                summary=summary,
                link=entry.get("link", ""),
                text=combined,
            ))
        print(f"[OK] Fetched {len(items)} articles")
    except Exception as e:
        print(f"[WARN] Failed to fetch {label} ({url}): {e}")
    return items


def scrape_all() -> list[dict]:
    all_items: list[NewsItem] = []
    for label, url in BBC_FEEDS.items():
        all_items.extend(fetch_feed(label, url))

    print(f"[INFO] Total scraped: {len(all_items)} articles")

    data = [asdict(item) for item in all_items]
    with open(SCRAPED_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {SCRAPED_JSON}")

    return data


# Entry

def main() -> None:
    scrape_all()


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic-ai-slim[anthropic,duckduckgo]",
#   "transformers",
#   "torch",
#   "pyarrow",
# ]
# ///
"""Search the web for vacation deals from Ottawa, label them with a HF model, save as Arrow.

Run: ANTHROPIC_API_KEY=... uv run deals.py
"""

import os
import statistics
import sys
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

MODEL = os.getenv("TRAVEL_MODEL", "anthropic:claude-sonnet-5-5")
ZS_MODEL = "facebook/bart-large-mnli"
LABEL_SETS = {  # field -> (hypothesis template, candidate labels)
    "trip_type": ("This is a {} vacation.", ["beach", "city", "ski", "nature and adventure", "cruise"]),
    "deal_kind": ("The deal type is {}.", ["flight only", "hotel only", "flight and hotel package", "all-inclusive package"]),
}


class Deal(BaseModel):
    destination: str = Field(description="City and country")
    price_cad: float | None = Field(description="Advertised price in CAD per person; null if not stated or not in CAD")
    what: str = Field(description="What the price buys, e.g. 'round-trip flight' or '7-night all-inclusive'")
    travel_dates: str | None = Field(description="Travel window as advertised")
    provider: str = Field(description="Airline, tour operator or site selling it")
    url: str = Field(description="URL exactly as it appeared in the search results")
    posted_on: str | None = Field(description="Date the deal was posted, if visible (YYYY-MM-DD)")
    summary: str = Field(description="One sentence describing the deal")


def search(today: str) -> list[Deal]:
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Caching, WebSearch

    agent = Agent(
        MODEL,
        name="ottawa_deals_agent",
        output_type=list[Deal],
        # Native search when the model has it, else DuckDuckGo. Caching: each search turn resends the history.
        capabilities=[WebSearch(local="duckduckgo"), Caching()],
        instructions=(
            "You find current vacation deals departing Ottawa (YOW). Run several searches: airline seat sales, "
            "all-inclusive packages, deal blogs and error fares from Ottawa. Only report deals you actually saw in "
            "search results, with the result's URL. Never invent or estimate prices. Skip deals that are expired "
            "or whose travel dates have passed."
        ),
    )
    result = agent.run_sync(f"Today is {today}. Find the best vacation deals from Ottawa, up to 25.")
    kept = verified(result.output, result.all_messages())
    print(f"{len(result.output)} deals extracted, {len(kept)} with URLs found in search results", file=sys.stderr)
    return kept


def verified(deals: list[Deal], messages: list) -> list[Deal]:
    """Guard against hallucinated deals: keep only URLs a search actually returned.

    part_kind is 'tool-return' (DuckDuckGo) or 'builtin-tool-return' (native search); the model's
    own tool-call args (where its output lives) are deliberately not counted.
    """
    seen = " ".join(
        str(p.content) for m in messages for p in m.parts if getattr(p, "part_kind", "").endswith("tool-return")
    )
    return [d for d in deals if d.url in seen]


def label(rows: list[dict]) -> None:
    from transformers import pipeline

    zs = pipeline("zero-shot-classification", model=ZS_MODEL)
    texts = [f"{r['destination']}: {r['what']}. {r['summary']}" for r in rows]
    for field, (template, labels) in LABEL_SETS.items():
        outs = zs(texts, candidate_labels=labels, hypothesis_template=template)
        for r, out in zip(rows, outs if isinstance(outs, list) else [outs]):
            r[field] = out["labels"][0]
            r[f"{field}_score"] = round(out["scores"][0], 3)


def insights(rows: list[dict]) -> str:
    priced = sorted((r for r in rows if r["price_cad"]), key=lambda r: r["price_cad"])
    lines = [f"{len(rows)} deals from Ottawa, {len(priced)} with a CAD price."]
    if priced:
        b = priced[0]
        lines.append(f"Cheapest: {b['destination']} at ${b['price_cad']:.0f} CAD ({b['what']}, {b['provider']}) {b['url']}")
    for field in LABEL_SETS:
        groups: dict[str, list[dict]] = {}
        for r in priced:
            groups.setdefault(r[field], []).append(r)
        lines.append(f"\nBy {field.replace('_', ' ')}:")
        for name, rs in sorted(groups.items(), key=lambda kv: kv[1][0]["price_cad"]):
            median = statistics.median(r["price_cad"] for r in rs)
            lines.append(f"- {name}: {len(rs)} deals, best {rs[0]['destination']} ${rs[0]['price_cad']:.0f}, median ${median:.0f}")
    shaky = sum(r[f"{f}_score"] < 0.5 for r in rows for f in LABEL_SETS)
    if shaky:
        lines.append(f"\n{shaky} labels scored under 0.5 confidence; check those by hand.")
    undated = sum(not r["posted_on"] for r in rows)
    if undated:
        lines.append(f"{undated} deals show no posted date; confirm they're still live before booking.")
    return "\n".join(lines)


def main() -> None:
    import pyarrow as pa
    import pyarrow.feather as feather

    today = date.today().isoformat()
    rows = [d.model_dump() | {"searched_at": today} for d in search(today)]
    if not rows:
        sys.exit("No verifiable deals found.")
    label(rows)
    out = Path("data")
    out.mkdir(exist_ok=True)
    feather.write_feather(pa.Table.from_pylist(rows), out / f"deals-{today}.arrow")
    text = insights(rows)
    (out / f"insights-{today}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

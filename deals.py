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
    depart_on: date | None = Field(description="First departure date the price applies to (YYYY-MM-DD); null if not stated")
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
            "You find current vacation deals departing Ottawa (YOW). Ottawa is cold from November to March, so "
            "always include sun all-inclusive packages (Mexico, Caribbean, Central America) from the tour operators "
            "selling from Ottawa: Sunwing, WestJet Vacations, Air Canada Vacations, Air Transat and Porter/Transat. "
            "Also search airline seat sales, deal blogs and error fares from Ottawa. Spread results across departure "
            "months over the next six months, including the Christmas/New Year and March Break peaks, so prices can "
            "be compared by time of year. Only report deals you actually saw in search results, with the result's "
            "URL. Never invent or estimate prices. Skip deals that are expired or whose travel dates have passed."
        ),
    )
    result = agent.run_sync(f"Today is {today}. Find the best vacation deals from Ottawa, up to 30.")
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


def travel_month(d: date | None) -> str:
    """Sortable time-of-year bucket. Christmas/New Year gets its own bucket: it prices far above early December."""
    if d is None:
        return "undated"
    # ponytail: fixed Dec 19-Jan 3 peak; March Break stays inside March, split it out if families need it
    if (d.month, d.day) >= (12, 19) or (d.month, d.day) <= (1, 3):
        return f"{d.year - (d.month == 1)}-12 holidays (Dec 19-Jan 3)"
    return f"{d:%Y-%m}"


def label(rows: list[dict]) -> None:
    from transformers import pipeline

    zs = pipeline("zero-shot-classification", model=ZS_MODEL)
    texts = [f"{r['destination']}: {r['what']}. {r['summary']}" for r in rows]
    for field, (template, labels) in LABEL_SETS.items():
        outs = zs(texts, candidate_labels=labels, hypothesis_template=template)
        for r, out in zip(rows, outs if isinstance(outs, list) else [outs]):
            r[field] = out["labels"][0]
            r[f"{field}_score"] = round(out["scores"][0], 3)


def grouped(priced: list[dict], field: str, chronological: bool = False) -> list[str]:
    """One line per value of field. priced must be sorted by price, so rs[0] is each group's cheapest."""
    groups: dict[str, list[dict]] = {}
    for r in priced:
        groups.setdefault(r[field], []).append(r)
    order = sorted(groups.items(), key=(lambda kv: kv[0]) if chronological else (lambda kv: kv[1][0]["price_cad"]))
    return [
        f"- {name}: {len(rs)} deals, best {rs[0]['destination']} ${rs[0]['price_cad']:.0f}, "
        f"median ${statistics.median(r['price_cad'] for r in rs):.0f}"
        for name, rs in order
    ]


def insights(rows: list[dict]) -> str:
    priced = sorted((r for r in rows if r["price_cad"]), key=lambda r: r["price_cad"])
    lines = [f"{len(rows)} deals from Ottawa, {len(priced)} with a CAD price."]
    if priced:
        b = priced[0]
        lines.append(f"Cheapest: {b['destination']} at ${b['price_cad']:.0f} CAD ({b['what']}, {b['provider']}) {b['url']}")
    for field in LABEL_SETS:
        lines.append(f"\nBy {field.replace('_', ' ')}:")
        lines += grouped(priced, field)
    all_incl = [r for r in priced if r["deal_kind"] == "all-inclusive package"]
    if all_incl:
        lines.append("\nAll-inclusive by departure month:")
        lines += grouped(all_incl, "travel_month", chronological=True)
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
    rows = [d.model_dump() | {"searched_at": today, "travel_month": travel_month(d.depart_on)} for d in search(today)]
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

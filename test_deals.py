"""Run: uv run --python 3.11 --with pydantic-ai-slim python test_deals.py"""

from datetime import date

from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from deals import Deal, insights, travel_month, verified

assert travel_month(date(2026, 12, 8)) == "2026-12"
assert travel_month(date(2026, 12, 19)) == "2026-12 holidays (Dec 19-Jan 3)"
assert travel_month(date(2027, 1, 3)) == "2026-12 holidays (Dec 19-Jan 3)"  # same season as the Dec dates
assert travel_month(date(2027, 1, 4)) == "2027-01"
assert travel_month(None) == "undated"
assert sorted(["2027-01", "undated", "2026-12 holidays (Dec 19-Jan 3)", "2026-12"]) == [
    "2026-12", "2026-12 holidays (Dec 19-Jan 3)", "2027-01", "undated"]  # buckets sort chronologically


def row(dest, price, trip, kind, score=0.9, posted="2026-11-01", depart=date(2026, 12, 8)):
    return dict(destination=dest, price_cad=price, what="round-trip flight", provider="X", url=f"https://x/{dest}",
                posted_on=posted, trip_type=trip, trip_type_score=score, deal_kind=kind, deal_kind_score=0.9,
                travel_month=travel_month(depart))


rows = [
    row("Cancun", 899.0, "beach", "all-inclusive package", depart=date(2026, 12, 29)),
    row("Punta Cana", 999.0, "beach", "all-inclusive package", score=0.3),
    row("Vancouver", 249.0, "city", "flight only", posted=None),
    row("Lisbon", None, "city", "flight only"),
]
text = insights(rows)
assert text.startswith("4 deals from Ottawa, 3 with a CAD price."), text
assert "Cheapest: Vancouver at $249 CAD" in text, text
assert "- beach: 2 deals, best Cancun $899, median $949" in text, text
assert text.index("- city:") < text.index("- beach:"), text  # groups sorted by best price
assert "1 labels scored under 0.5" in text, text
assert "1 deals show no posted date" in text, text
# All-inclusive section: flights excluded, months in date order (not price order).
ai = text.split("All-inclusive by departure month:\n")[1].split("\n\n")[0].splitlines()
assert ai == ["- 2026-12: 1 deals, best Punta Cana $999, median $999",
              "- 2026-12 holidays (Dec 19-Jan 3): 1 deals, best Cancun $899, median $899"], ai

# URL guard: a URL only in the model's own output (tool-call args) must not count as seen.
real, fake = (Deal(destination=d, price_cad=1.0, what="w", travel_dates=None, depart_on=None, provider="p",
                   url=f"https://{d}.example/x", posted_on=None, summary="s") for d in ("real", "fake"))
messages = [
    ModelResponse(parts=[ToolCallPart(tool_name="web_search", args={"query": "q"}, tool_call_id="1")]),
    ModelRequest(parts=[ToolReturnPart(tool_name="web_search", content=[{"href": real.url}], tool_call_id="1")]),
    ModelResponse(parts=[ToolCallPart(tool_name="final_result", args={"response": [real.model_dump(), fake.model_dump()]},
                                      tool_call_id="2")]),
]
assert verified([real, fake], messages) == [real]
print("ok")

"""Run: uv run --python 3.11 --with pydantic-ai-slim python test_deals.py"""

from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from deals import Deal, insights, verified


def row(dest, price, trip, kind, score=0.9, posted="2026-11-01"):
    return dict(destination=dest, price_cad=price, what="round-trip flight", provider="X", url=f"https://x/{dest}",
                posted_on=posted, trip_type=trip, trip_type_score=score, deal_kind=kind, deal_kind_score=0.9)


rows = [
    row("Cancun", 899.0, "beach", "all-inclusive package"),
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

# URL guard: a URL only in the model's own output (tool-call args) must not count as seen.
real, fake = (Deal(destination=d, price_cad=1.0, what="w", travel_dates=None, provider="p", url=f"https://{d}.example/x",
                   posted_on=None, summary="s") for d in ("real", "fake"))
messages = [
    ModelResponse(parts=[ToolCallPart(tool_name="web_search", args={"query": "q"}, tool_call_id="1")]),
    ModelRequest(parts=[ToolReturnPart(tool_name="web_search", content=[{"href": real.url}], tool_call_id="1")]),
    ModelResponse(parts=[ToolCallPart(tool_name="final_result", args={"response": [real.model_dump(), fake.model_dump()]},
                                      tool_call_id="2")]),
]
assert verified([real, fake], messages) == [real]
print("ok")

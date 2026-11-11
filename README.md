# travel

Weekly search for vacation deals from Ottawa (YOW).

1. A Pydantic AI agent searches the web and returns validated `Deal` objects. It uses the model's built-in web search, or DuckDuckGo if the model has none. Deals whose URL never showed up in the search results are dropped.
2. `facebook/bart-large-mnli` (zero-shot) labels each deal by `trip_type` and `deal_kind`, and records a confidence score for each label.
3. Rows are saved to `data/deals-YYYY-MM-DD.arrow` (Arrow IPC/Feather), with a text summary in `data/insights-YYYY-MM-DD.txt`.

```sh
ANTHROPIC_API_KEY=... uv run deals.py        # TRAVEL_MODEL=... to swap the LLM
uv run --python 3.11 --with pydantic-ai-slim python test_deals.py  # self-check, no key needed
```

Read the output with `pyarrow.feather.read_table(path)` or `polars.read_ipc(path)`.

`.github/workflows/deals.yml` runs every Monday and can also be started by hand. It needs the `ANTHROPIC_API_KEY` repo secret, writes the insights to the job summary and uploads `data/` as an artifact.

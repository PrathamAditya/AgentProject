"""Tool definitions (D15) + registration into the toolbox memory store (R5, D11).

3 relevant tools: paper_search, fetch_notes, get_current_time.
6 decoys with unrelated authored docstrings.
"""

from __future__ import annotations

import datetime
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from chunker import chunk_text
from config import ABSTRACT_CAP

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
KB_DIR = FIXTURES_DIR / "kb"


# --------------------------------------------------------------------------- #
# real tools
# --------------------------------------------------------------------------- #
def _arxiv_search(query: str) -> list[dict]:
    import arxiv

    search = arxiv.Search(query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
    out = []
    for r in search.results():
        out.append(
            {
                "arxiv_id": r.get_short_id() or "",
                "entry_id": r.entry_id,
                "title": (r.title or "").strip(),
                "authors": [a.name for a in (r.authors or [])],
                "published": str(r.published or ""),
                "abstract": (r.summary or "")[:ABSTRACT_CAP],
            }
        )
    return out


def _fixture_candidates(query: str) -> list[dict]:
    """Local fallback so the tool is usable offline (fixture corpus) with the same
    shape as the live arXiv tool."""
    candidates = []
    for md in sorted(KB_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        title_m = re.search(r"#\s+(.+)", text)
        author_m = re.search(r"\*\*Authors?:\*\*\s*(.+)", text)
        id_m = re.search(r"\*\*Id:\*\*\s*(\S+)", text)
        year_m = re.search(r"\((\d{4})\)", text)
        abstract = text[:ABSTRACT_CAP]
        candidates.append(
            {
                "arxiv_id": id_m.group(1) if id_m else "UNKNOWN",
                "entry_id": f"fixture://{md.name}",
                "title": title_m.group(1).strip() if title_m else md.stem,
                "authors": [a.strip() for a in (author_m.group(1).split(",") if author_m else [])],
                "published": year_m.group(1) if year_m else "unknown",
                "abstract": abstract,
            }
        )
    q = (query or "").lower()
    if q:
        matches = [c for c in candidates if any(
            w in (c["title"] + " " + c["abstract"]).lower() for w in q.split()
        )]
        if matches:
            candidates = matches
    return candidates


def paper_search(query: str):
    """Search arXiv for research papers matching a query.

    Returns a JSON array of candidates, each with arxiv_id, entry_id, title, authors,
    published, and abstract. Network required; falls back to the local fixture corpus
    when arXiv is unreachable.
    """
    try:
        results = _arxiv_search(query or "")
    except Exception:
        results = _fixture_candidates(query or "")
    return json.dumps(results, indent=2)


def fetch_notes(path: str = "kestrel-notes.md", ctx=None):
    """Read a fixture notes file, chunk it, and store each chunk into the knowledge
    base with source metadata (R14 search-and-store). Returns the full file text plus an
    oversized filler so the bounded-excerpt path is guaranteed to be exercised.

    The default target exercises the oversized-output path (>3,000 chars).
    """
    target = Path(path)
    if not target.is_absolute():
        target = KB_DIR / target.name
    if not target.exists():
        return f"File not found: {path}"
    text = target.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    num = len(chunks)
    embedder = ctx.get("embedder")
    for i, chunk in enumerate(chunks):
        emb = embedder.embed(chunk)
        ctx["manager"].knowledge_base.add_chunk(
            chunk,
            emb,
            source=target.name,
            chunk_id=i,
            num_chunks=num,
            title=target.stem,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
    # append the oversized filler to guarantee >3,000-char return (AC10)
    filler = (FIXTURES_DIR / "tools" / "oversized-output.txt").read_text(encoding="utf-8")
    return text + "\n\n" + filler


def get_current_time(ctx=None):
    """Return the current local date and time as an ISO string."""
    return datetime.datetime.now().astimezone().isoformat()


# --------------------------------------------------------------------------- #
# decoys (authored, unrelated)
# --------------------------------------------------------------------------- #
def knit_pattern_helper(gauge_stitches: int, rows_per_inch: float):
    """Help compute knitting gauge swatch measurements from stitches per inch and rows per inch."""
    return json.dumps({"gauge_stitches": gauge_stitches, "rows_per_inch": rows_per_inch})


def currency_convert(amount: float, from_currency: str, to_currency: str):
    """Convert an amount between two currencies using today's indicative rate."""
    return json.dumps({"amount": amount, "from": from_currency, "to": to_currency})


def recipe_scaler(servings: int, target_servings: int, ingredient_list: str):
    """Scale a recipe ingredient list from one serving count to another."""
    return json.dumps({"servings": servings, "target_servings": target_servings})


def translate_phrase(phrase: str, target_lang: str):
    """Translate a short everyday phrase into a target language."""
    return json.dumps({"phrase": phrase, "target_lang": target_lang})


def calendar_lookup(date: str):
    """Look up calendar metadata (day of week, holidays) for a given date."""
    return json.dumps({"date": date})


def sports_scores(league: str, team: str = ""):
    """Return recent sports scores for a league and optional team."""
    return json.dumps({"league": league, "team": team})


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    name: str
    func: callable
    docstring: str = ""
    parameters: dict = field(default_factory=dict)
    needs_context: bool = False

    def execute(self, args: dict, ctx: dict | None = None) -> str:
        kwargs = dict(args)
        if self.needs_context:
            kwargs["ctx"] = ctx
        return str(self.func(**kwargs))

    @property
    def signature(self) -> str:
        return json.dumps(self.parameters)


def _make_parameters(func) -> dict:
    sig = inspect.signature(func)
    props = {}
    required = []
    for name, param in sig.parameters.items():
        if name == "ctx":
            continue
        ann = param.annotation
        ptype = "string"
        if ann in (int, float):
            ptype = "number"
        elif ann is bool:
            ptype = "boolean"
        props[name] = {"type": ptype}
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif param.default is not None:
            props[name]["description"] = f"(default {param.default!r})"
    return {"type": "object", "properties": props, "required": required}


def _src(func) -> str:
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return func.__doc__ or ""


_TOOL_FUNCS = [
    paper_search,
    fetch_notes,
    get_current_time,
    knit_pattern_helper,
    currency_convert,
    recipe_scaler,
    translate_phrase,
    calendar_lookup,
    sports_scores,
]


def build_tools() -> list[Tool]:
    tools = []
    for f in _TOOL_FUNCS:
        needs_ctx = "ctx" in inspect.signature(f).parameters
        tools.append(
            Tool(
                name=f.__name__,
                func=f,
                docstring=inspect.getdoc(f) or "",
                parameters=_make_parameters(f),
                needs_context=needs_ctx,
            )
        )
    return tools


def register_tools(
    manager,
    llm,
    embedder,
    tools: list[Tool] | None = None,
    augment: bool = True,
) -> dict[str, bool]:
    """Register tools into the toolbox store. Returns {tool_name: inserted_bool}.
    Idempotent per tool name (R5); old names never duplicate."""
    results = {}
    for tool in tools or build_tools():
        description = tool.docstring
        queries = []
        if augment:
            description, queries = llm.augment_tool(tool.docstring, _src(tool.func))
        embedding_text = (
            f"{tool.name}\n{description}\n{tool.signature}\n" + "\n".join(queries)
        )
        emb = embedder.embed(embedding_text)
        inserted = manager.toolbox.register(
            tool_name=tool.name,
            description=description,
            signature=tool.signature,
            embedding_text=embedding_text,
            embedding=emb,
            synthetic_queries=queries,
        )
        results[tool.name] = inserted
    return results


TOOLS_BY_NAME = {t.name: t for t in build_tools()}

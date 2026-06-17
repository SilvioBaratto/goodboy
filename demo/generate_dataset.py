"""Generate a 1000-entry preference dataset via BAML (RLAIF-style).

Uses the Azure OpenAI client + .env borrowed from the ITAL-IA project's /api.
An LLM writes the (prompt, chosen, rejected) pairs instead of humans — this is
the "AI feedback" variant of RLHF the explainer ends on.

Output: demo/data/preferences.jsonl  (consumed by train_dpo.py)

Run:
    python demo/generate_dataset.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # make baml_client importable

# .env borrowed from the ITAL-IA api project (Azure OpenAI gpt-4.1 deployment).
ITALIA_ENV = Path("/Volumes/External SSD/ITAL-IA/api/.env")
OUT = HERE / "data" / "preferences.jsonl"

TARGET = 1000
BATCH = 10          # pairs requested per LLM call
WAVE = 12           # concurrent calls per wave
MAX_WAVES = 40
CONCURRENCY = 6     # max in-flight Azure calls

CATEGORIES = [
    "workplace and colleagues",
    "family and relatives",
    "friendships",
    "romantic relationships and breakups",
    "online and social media",
    "neighbors and shared spaces",
    "school, university, classmates",
    "money, debts and shared bills",
    "driving and traffic",
    "strangers and public situations",
    "roommates and flatmates",
    "customers, services and businesses",
]

VARIATIONS = [
    "small everyday annoyances",
    "a serious betrayal of trust",
    "being publicly embarrassed",
    "someone taking credit or stealing an idea",
    "repeated, ongoing disrespect",
    "feeling ignored or excluded",
    "being lied about or gossiped over",
    "money owed or unfairly lost",
    "blatant rudeness or aggression",
    "subtle passive-aggressive behavior",
]


def load_env(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE), no external deps."""
    if not path.exists():
        raise SystemExit(f"ITAL-IA .env not found at {path}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


def norm(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def valid(pair) -> bool:
    p, c, r = pair.prompt, pair.chosen, pair.rejected
    return (
        p.startswith("Question:")
        and "Answer:" in p
        and len(c.strip()) > 0
        and len(r.strip()) > 0
        and norm(c) != norm(r)
    )


def fix_space(text: str) -> str:
    return text if text.startswith(" ") else " " + text.lstrip()


async def main() -> None:
    load_env(ITALIA_ENV)
    # Import AFTER env is set so BAML resolves AZURE_OPENAI_* at runtime.
    from baml_client.async_client import b

    sem = asyncio.Semaphore(CONCURRENCY)
    pairs: dict[str, dict] = {}

    async def one_call(category: str, variation: str) -> list:
        async with sem:
            try:
                return await b.GeneratePreferenceBatch(  # type: ignore
                    category=category, count=BATCH, variation=variation
                )
            except Exception as exc:  # keep the wave going on a single failure
                print(f"  ! call failed ({category} / {variation}): {exc}")
                return []

    ci = vi = 0
    for wave in range(MAX_WAVES):
        if len(pairs) >= TARGET:
            break
        jobs = []
        for _ in range(WAVE):
            jobs.append(one_call(CATEGORIES[ci % len(CATEGORIES)],
                                 VARIATIONS[vi % len(VARIATIONS)]))
            ci += 1
            vi += 1  # rotate both so pairings keep shifting
        results = await asyncio.gather(*jobs)
        added = 0
        for batch in results:
            for pair in batch:
                if not valid(pair):
                    continue
                key = norm(pair.prompt)
                if key in pairs:
                    continue
                pairs[key] = {
                    "prompt": pair.prompt,
                    "chosen": fix_space(pair.chosen),
                    "rejected": fix_space(pair.rejected),
                }
                added += 1
        print(f"wave {wave + 1}: +{added} unique  (total {len(pairs)}/{TARGET})")

    rows = list(pairs.values())[:TARGET]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} preference pairs to {OUT}")
    if len(rows) < TARGET:
        print(f"WARNING: only {len(rows)} unique pairs (< {TARGET}). "
              f"Raise MAX_WAVES or add CATEGORIES/VARIATIONS.")


if __name__ == "__main__":
    asyncio.run(main())

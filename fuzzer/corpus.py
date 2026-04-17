"""
fuzzer/corpus.py — Corpus, Chain, and CorpusEntry.

The corpus is the fuzzer's memory. It maintains a priority queue of novel entries,
a deduplication seen-set, a (score, category) index for novelty checking and splice
selection, and a crash log for score-3 entries.

Novelty gate (two conditions — an entry is admitted if either is true):
  1. score > current maximum score in the corpus  (new high watermark)
  2. (score, category) pair not yet seen           (new failure mode at same level)

Score-0 entries are never admitted.

The Chain dataclass separates concerns:
  strategies — ordered list of semantic technique/tactic IDs the composer follows
  tool_spec  — unordered set of deterministic transform IDs offered as annotations

This split reflects the composite execution model: strategies go to the composer
as instructions; tool_spec is offered as callable annotation tools.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .scorer import ScoreResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


@dataclass
class Chain:
    """
    strategies: ordered — the composer applies them in sequence.
                e.g. ["tactic-roleplay", "researcher-persona", "lang-Russian"]

    tool_spec:  unordered — offered to the composer as annotation tools.
                e.g. ["base64", "unicode-confusable"]
                The composer decides if, where, and how many times to use them.
    """

    strategies: list[str]
    tool_spec: list[str]

    def seen_key(self) -> tuple:
        """
        Deduplication key.
        strategies is order-sensitive: [A, B] ≠ [B, A].
        tool_spec is order-insensitive: {base64, rot13} == {rot13, base64}.
        """
        return (tuple(self.strategies), frozenset(self.tool_spec))

    def is_empty(self) -> bool:
        return not self.strategies and not self.tool_spec

    def __repr__(self) -> str:
        s = "+".join(self.strategies) if self.strategies else "∅"
        t = "{" + ",".join(sorted(self.tool_spec)) + "}" if self.tool_spec else "∅"
        return f"Chain(strategies={s}, tools={t})"


# ---------------------------------------------------------------------------
# CorpusEntry
# ---------------------------------------------------------------------------


@dataclass
class CorpusEntry:
    entry_id: str
    seed_prompt: str          # which seed this mutation started from
    chain: Chain
    mutated_prompt: str       # exact string sent to the target
    score: ScoreResult
    depth: int                # mutation depth from seed (0 = direct seed mutation)
    parent_id: Optional[str] = None

    # heapq comparison: higher score first, then shallower depth
    def __lt__(self, other: CorpusEntry) -> bool:
        return (-self.score.score, self.depth) < (-other.score.score, other.depth)

    def __le__(self, other: CorpusEntry) -> bool:
        return (-self.score.score, self.depth) <= (-other.score.score, other.depth)

    def __repr__(self) -> str:
        return (
            f"CorpusEntry(id={self.entry_id[:8]}, "
            f"score={self.score.score}, depth={self.depth}, "
            f"chain={self.chain!r})"
        )


def make_entry(
    seed_prompt: str,
    chain: Chain,
    mutated_prompt: str,
    score: ScoreResult,
    depth: int,
    parent_id: Optional[str] = None,
) -> CorpusEntry:
    """Factory that generates a fresh entry_id."""
    return CorpusEntry(
        entry_id=str(uuid.uuid4()),
        seed_prompt=seed_prompt,
        chain=chain,
        mutated_prompt=mutated_prompt,
        score=score,
        depth=depth,
        parent_id=parent_id,
    )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


class Corpus:
    """
    Async-safe corpus. All mutating methods acquire a lock.

    Internally:
      _entries  — min-heap by (-score, depth); heapq = max-score, min-depth first
      _seen     — set of chain.seen_key() for deduplication
      _index    — dict[(score, category)] → [CorpusEntry] for novelty + splicing
      _crashes  — append-only list of score-3 entries
    """

    def __init__(self) -> None:
        self._entries: list[CorpusEntry] = []
        self._seen: set[tuple] = set()
        self._index: dict[tuple[int, str], list[CorpusEntry]] = {}
        self._crashes: list[CorpusEntry] = []
        self._max_score: int = -1
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    async def add(self, entry: CorpusEntry) -> bool:
        """
        Attempt to add an entry. Returns True if admitted (novel), False if
        rejected (duplicate chain or non-novel score/category).

        Crashes (score=3) bypass both the chain-dedup and novelty checks —
        every crash is worth preserving regardless of whether the same chain
        or category was seen before (different seeds produce different prompts).
        """
        async with self._lock:
            if entry.score.score < 3 and entry.chain.seen_key() in self._seen:
                logger.debug("Duplicate chain — skipping %s", entry.entry_id[:8])
                return False
            if not self._is_novel(entry):
                logger.debug(
                    "Non-novel (score=%d, cat=%s) — skipping %s",
                    entry.score.score, entry.score.category, entry.entry_id[:8],
                )
                return False
            self._admit(entry)
            return True

    def _is_novel(self, entry: CorpusEntry) -> bool:
        """
        Novelty check — called under lock.

        Rule: any unique chain that produces score > 0 is worth keeping.
        Chain deduplication (seen_key) in add() prevents true duplicates.
        Corpus size is managed separately by cull().

        Score=0 is never admitted — no useful signal.
        Score=3 (crash) always admitted — every crash is worth saving.
        Score=1/2 admitted if the chain hasn't been seen before (enforced
        by the seen_key check in add() before this is even called).
        """
        return entry.score.score > 0

    def _admit(self, entry: CorpusEntry) -> None:
        """Unconditionally insert — must be called under lock."""
        self._seen.add(entry.chain.seen_key())
        heapq.heappush(self._entries, entry)

        sc_key = (entry.score.score, entry.score.category)
        self._index.setdefault(sc_key, []).append(entry)

        if entry.score.score > self._max_score:
            self._max_score = entry.score.score

        if entry.score.score == 3:
            self._crashes.append(entry)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    async def pick_parent(self) -> Optional[CorpusEntry]:
        """
        Weighted random selection. Weight = (score + 1)^2 / (depth + 1).
        Biases toward high-scoring shallow entries while still exploring.
        """
        async with self._lock:
            if not self._entries:
                return None
            weights = [
                (e.score.score + 1) ** 2 / (e.depth + 1)
                for e in self._entries
            ]
            return random.choices(self._entries, weights=weights, k=1)[0]

    async def get_top_k(self, k: int = 10) -> list[CorpusEntry]:
        """Top-k entries by (score desc, depth asc). Used by LLM-guided composer."""
        async with self._lock:
            return sorted(
                self._entries,
                key=lambda e: (-e.score.score, e.depth),
            )[:k]

    async def get_by_min_score(self, min_score: int) -> list[CorpusEntry]:
        """All entries at or above a score threshold. Used for splicing."""
        async with self._lock:
            return [e for e in self._entries if e.score.score >= min_score]

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def has_crash(self) -> bool:
        return bool(self._crashes)

    def crashes(self) -> list[CorpusEntry]:
        return list(self._crashes)

    def size(self) -> int:
        return len(self._entries)

    def max_score(self) -> int:
        return self._max_score

    def score_distribution(self) -> dict[int, int]:
        """Count of entries per score level."""
        dist: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for e in self._entries:
            dist[e.score.score] = dist.get(e.score.score, 0) + 1
        return dist

    # ------------------------------------------------------------------
    # Culling
    # ------------------------------------------------------------------

    async def cull(self) -> int:
        """
        Keep only the shallowest entry per (score, category) pair.
        Score-3 (crash) entries are always retained unconditionally.
        Returns number of entries removed.
        """
        async with self._lock:
            best: dict[tuple[int, str], CorpusEntry] = {}
            for entry in self._entries:
                if entry.score.score == 3:
                    continue
                sc_key = (entry.score.score, entry.score.category)
                if sc_key not in best or entry.depth < best[sc_key].depth:
                    best[sc_key] = entry

            before = len(self._entries)
            kept = list(best.values()) + list(self._crashes)
            self._entries = kept
            heapq.heapify(self._entries)

            # Rebuild index from survivors
            self._index = {}
            for entry in self._entries:
                sc_key = (entry.score.score, entry.score.category)
                self._index.setdefault(sc_key, []).append(entry)

            removed = before - len(self._entries)
            if removed:
                logger.info("Culled %d dominated entries (%d remain)", removed, len(self._entries))
            return removed

    # ------------------------------------------------------------------
    # Serialisation (checkpointing)
    # ------------------------------------------------------------------

    def to_dict_list(self) -> list[dict]:
        """Serialise all entries for writing to corpus.json."""
        return [_entry_to_dict(e) for e in self._entries]

    @classmethod
    def from_dict_list(cls, data: list[dict]) -> Corpus:
        """Restore from corpus.json. Bypasses novelty check — trusts saved state."""
        corpus = cls()
        for d in data:
            entry = _entry_from_dict(d)
            corpus._admit(entry)  # direct insert, no lock needed during restore
        logger.info("Restored corpus: %d entries, max_score=%d", corpus.size(), corpus.max_score())
        return corpus


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _entry_to_dict(e: CorpusEntry) -> dict:
    return {
        "entry_id": e.entry_id,
        "parent_id": e.parent_id,
        "seed_prompt": e.seed_prompt,
        "chain": {
            "strategies": e.chain.strategies,
            "tool_spec": e.chain.tool_spec,
        },
        "mutated_prompt": e.mutated_prompt,
        "score": {
            "score": e.score.score,
            "category": e.score.category,
            "reasoning": e.score.reasoning,
        },
        "depth": e.depth,
    }


def _entry_from_dict(d: dict) -> CorpusEntry:
    return CorpusEntry(
        entry_id=d["entry_id"],
        parent_id=d.get("parent_id"),
        seed_prompt=d["seed_prompt"],
        chain=Chain(
            strategies=d["chain"]["strategies"],
            tool_spec=d["chain"]["tool_spec"],
        ),
        mutated_prompt=d["mutated_prompt"],
        score=ScoreResult(
            score=d["score"]["score"],
            category=d["score"]["category"],
            reasoning=d["score"]["reasoning"],
        ),
        depth=d["depth"],
    )

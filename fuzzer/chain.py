"""
fuzzer/chain.py — UCB1-guided chain builder.

Maintains two independent UCB1 bandits:
  strategy_ucb — over technique/tactic/translation transform IDs
  tool_ucb     — over deterministic (encoding/obfuscation/structural) transform IDs

UCB1 formula:
  score(item) = avg_reward(item) + sqrt(2 * ln(total_pulls) / pulls(item))

  Rewards:  0.0 = chain discarded (non-novel or annotation failure)
            1.0 = chain admitted to corpus
            2.0 = chain produced a score-3 crash

Items never pulled receive infinite UCB1 score — explore-first ensures every
transform is tried before exploitation begins.

Chain construction (random mode):
  N strategies sampled (ordered, without replacement) from strategy_ucb
  M tools sampled (unordered, without replacement) from tool_ucb
  N ~ Geometric(p=0.5, min=1), M ~ Geometric(p=0.4, min=0)

Splicing:
  strategies from entry_a + tool_spec from entry_b
  The two layers are independent — any semantic approach works with any encoding.

Phase 1 helpers:
  all_single_strategy_chains() — one chain per strategy (for LLM-based transforms)
  all_single_tool_chains()     — one chain per tool (for deterministic transforms)
  These are used by the engine for the deterministic bootstrap phase.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from transforms import TRANSFORMS  # type: ignore[import]

from .corpus import Chain, CorpusEntry

logger = logging.getLogger(__name__)

# Reward constants
REWARD_DISCARD = 0.0
REWARD_ADMIT   = 1.0
REWARD_CRASH   = 2.0

# Which TRANSFORMS groups map to strategies vs tools
_STRATEGY_GROUPS = {"technique", "tactic"}
_TOOL_GROUPS     = {"encoding", "obfuscation", "structural"}


# ---------------------------------------------------------------------------
# UCB1 bandit
# ---------------------------------------------------------------------------


@dataclass
class _ItemStats:
    pulls: int = 0
    reward_sum: float = 0.0

    def avg_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls > 0 else 0.0


class UCB1:
    """Multi-armed bandit with UCB1 selection policy."""

    def __init__(self, items: list[str]) -> None:
        self._items: list[str] = list(items)
        self._stats: dict[str, _ItemStats] = {item: _ItemStats() for item in items}
        self._total_pulls: int = 0

    def ucb_score(self, item: str) -> float:
        s = self._stats.get(item)
        if s is None or s.pulls == 0:
            return float("inf")
        exploitation = s.avg_reward()
        exploration  = math.sqrt(2.0 * math.log(max(1, self._total_pulls)) / s.pulls)
        return exploitation + exploration

    def sample_one(self) -> Optional[str]:
        """Return the item with the highest UCB1 score. Ties broken randomly."""
        if not self._items:
            return None
        max_s = max(self.ucb_score(i) for i in self._items)
        candidates = [i for i in self._items if self.ucb_score(i) >= max_s]
        return random.choice(candidates)

    def sample_n_ordered(self, n: int) -> list[str]:
        """
        Sample n unique items without replacement, preserving UCB1-greedy order.
        Used for strategies (order matters).
        """
        available = list(self._items)
        result: list[str] = []
        for _ in range(min(n, len(available))):
            max_s = max(self.ucb_score(i) for i in available)
            candidates = [i for i in available if self.ucb_score(i) >= max_s]
            chosen = random.choice(candidates)
            result.append(chosen)
            available.remove(chosen)
        return result

    def sample_n_unordered(self, n: int) -> list[str]:
        """
        Sample n unique items, return sorted (order-insensitive).
        Used for tool_spec.
        """
        return sorted(self.sample_n_ordered(n))

    def update(self, items: list[str], reward: float) -> None:
        """Record the reward for all items that appeared in the chain."""
        for item in items:
            if item in self._stats:
                self._stats[item].pulls += 1
                self._stats[item].reward_sum += reward
        if items:
            self._total_pulls += 1

    def to_dict(self) -> dict:
        return {
            "total_pulls": self._total_pulls,
            "stats": {
                item: {"pulls": s.pulls, "reward_sum": s.reward_sum}
                for item, s in self._stats.items()
            },
        }

    @classmethod
    def from_dict(cls, items: list[str], data: dict) -> UCB1:
        ucb = cls(items)
        ucb._total_pulls = data.get("total_pulls", 0)
        for item, sd in data.get("stats", {}).items():
            if item in ucb._stats:
                ucb._stats[item].pulls      = sd.get("pulls", 0)
                ucb._stats[item].reward_sum = sd.get("reward_sum", 0.0)
        return ucb


# ---------------------------------------------------------------------------
# ChainBuilder
# ---------------------------------------------------------------------------


class ChainBuilder:
    """
    Builds mutation chains using UCB1-guided sampling.

    strategies_pool — technique + tactic IDs from TRANSFORMS + translation IDs
    tools_pool      — encoding + obfuscation + structural IDs (deterministic only)
    """

    def __init__(
        self,
        transform_groups: list[str],
        languages: list[str],
    ) -> None:
        self.strategies_pool, self.tools_pool = _partition_transforms(
            transform_groups, languages
        )
        self._strategy_ucb = UCB1(self.strategies_pool)
        self._tool_ucb     = UCB1(self.tools_pool)

        logger.info(
            "ChainBuilder: %d strategies, %d tools",
            len(self.strategies_pool), len(self.tools_pool),
        )

    # ------------------------------------------------------------------
    # Random chain construction
    # ------------------------------------------------------------------

    def random_chain(self) -> Optional[Chain]:
        """
        Sample a random chain via UCB1.
          N strategies (ordered) ~ Geometric(p=0.5), min=1
          M tools (unordered)    ~ Geometric(p=0.4), min=0

        Returns None if either pool is empty and can't satisfy min constraints.
        """
        if not self.strategies_pool and not self.tools_pool:
            logger.warning("Both strategy and tool pools are empty")
            return None

        n = _geometric(p=0.5, lo=1, hi=len(self.strategies_pool)) if self.strategies_pool else 0
        m = _geometric(p=0.4, lo=0, hi=len(self.tools_pool))     if self.tools_pool else 0

        strategies = self._strategy_ucb.sample_n_ordered(n)
        tools      = self._tool_ucb.sample_n_unordered(m)

        return Chain(strategies=strategies, tool_spec=tools)

    # ------------------------------------------------------------------
    # Splicing
    # ------------------------------------------------------------------

    def splice(self, entry_a: CorpusEntry, entry_b: CorpusEntry) -> Chain:
        """
        Combine strategies from entry_a with tool_spec from entry_b.
        The two layers are independent — any semantic framing + any encoding toolkit.
        """
        return Chain(
            strategies=list(entry_a.chain.strategies),
            tool_spec=list(entry_b.chain.tool_spec),
        )

    # ------------------------------------------------------------------
    # UCB1 weight update
    # ------------------------------------------------------------------

    def update(self, chain: Chain, admitted: bool, crashed: bool) -> None:
        """Call after each iteration to update bandit weights."""
        reward = REWARD_CRASH if crashed else (REWARD_ADMIT if admitted else REWARD_DISCARD)
        if chain.strategies:
            self._strategy_ucb.update(chain.strategies, reward)
        if chain.tool_spec:
            self._tool_ucb.update(chain.tool_spec, reward)

    # ------------------------------------------------------------------
    # Phase 1 helpers
    # ------------------------------------------------------------------

    def all_single_strategy_chains(self) -> list[Chain]:
        """
        One chain per strategy — used in phase 1 to exhaust LLM-based transforms.
        Each chain has one strategy and no tools (the strategy alone is the mutation).
        """
        return [Chain(strategies=[s], tool_spec=[]) for s in self.strategies_pool]

    def all_single_tool_chains(self) -> list[Chain]:
        """
        One chain per tool — used in phase 1 to exhaust deterministic transforms.
        These bypass the composer entirely (applied directly to the seed prompt).
        """
        return [Chain(strategies=[], tool_spec=[t]) for t in self.tools_pool]

    # ------------------------------------------------------------------
    # Phase 3 — LLM-guided proposal
    # ------------------------------------------------------------------

    async def llm_propose(
        self,
        composer: "Composer",  # avoid circular import; resolved at runtime
        top_k_entries: list,
        seed_intent: str,
        target_context: str = "",
    ) -> Optional[Chain]:
        """
        Ask the composer to propose a chain based on score history.

        Validates the returned strategy/tool IDs against our pools.
        Falls back to random_chain() if the proposal is unusable.
        Returns a Chain, or None if both proposal and fallback fail.
        """
        from .composer import Composer as _Composer  # local import avoids circular dep

        proposal = await composer.propose_chain(
            top_k_entries=top_k_entries,
            strategies_pool=self.strategies_pool,
            tools_pool=self.tools_pool,
            seed_intent=seed_intent,
            target_context=target_context,
        )

        if proposal is None:
            logger.debug("llm_propose: composer returned None — falling back to random")
            return self.random_chain()

        raw_strategies = proposal.get("strategies") or []
        raw_tools      = proposal.get("tools") or []
        rationale      = proposal.get("rationale", "")

        # Validate — silently drop any ID not in our pools
        strategies = [s for s in raw_strategies if s in self.strategies_pool]
        tools      = [t for t in raw_tools      if t in self.tools_pool]

        if not strategies and not tools:
            logger.debug(
                "llm_propose: no valid IDs in proposal (strategies=%r, tools=%r) "
                "— falling back to random",
                raw_strategies, raw_tools,
            )
            return self.random_chain()

        logger.info(
            "llm_propose: strategies=%s tools=%s rationale=%r",
            strategies, tools, rationale[:80],
        )
        return Chain(strategies=strategies, tool_spec=sorted(tools))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "strategy_ucb": self._strategy_ucb.to_dict(),
            "tool_ucb":     self._tool_ucb.to_dict(),
        }

    def restore_weights(self, data: dict) -> None:
        """Load saved UCB1 state from checkpoint."""
        if "strategy_ucb" in data:
            self._strategy_ucb = UCB1.from_dict(self.strategies_pool, data["strategy_ucb"])
        if "tool_ucb" in data:
            self._tool_ucb = UCB1.from_dict(self.tools_pool, data["tool_ucb"])
        logger.info("Restored UCB1 weights (total_pulls strategy=%d, tool=%d)",
                    self._strategy_ucb._total_pulls, self._tool_ucb._total_pulls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _partition_transforms(
    transform_groups: list[str],
    languages: list[str],
) -> tuple[list[str], list[str]]:
    """
    Split configured groups into (strategies, tools).

    Strategies: technique + tactic groups (LLM-dependent) + language translations
    Tools:      encoding + obfuscation + structural groups (deterministic only)
                noise (structural, requires_llm=True) is excluded from both pools
    """
    strategies: list[str] = []
    tools:      list[str] = []

    for tid, t in TRANSFORMS.items():
        if t.group not in transform_groups:
            continue
        if t.group in _STRATEGY_GROUPS:
            strategies.append(tid)
        elif t.group in _TOOL_GROUPS and not t.requires_llm:
            tools.append(tid)
        # requires_llm structural (noise) → skip from both pools

    # Translation strategies (dynamic, not in TRANSFORMS)
    if "translation" in transform_groups:
        for lang in languages:
            strategies.append(f"lang-{lang}")

    return strategies, tools


def _geometric(p: float, lo: int, hi: int) -> int:
    """
    Draw from a Geometric(p) distribution clamped to [lo, hi].
    Mean = 1/p. Most draws are small; occasionally deeper chains are produced.
    """
    if hi <= 0:
        return 0
    n = 0
    while True:
        n += 1
        if random.random() < p or n >= hi:
            break
    return max(lo, min(n, hi))
